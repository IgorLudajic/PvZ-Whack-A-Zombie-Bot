# -*- coding: utf-8 -*-
"""
Petlja prave partije: kaptura ekrana -> percepcija (YOLO + HSV + HUD) ->
Snapshot -> RL politika -> humanizovano izvršenje. Vrti se dok partija
ne bude dobijena ili izgubljena.

Kraj partije se detektuje na dva načina:
1. šablonima pobede/poraza (moneybag/gameover) - brzi put,
2. nestankom HUD-a (ikona sunca) na >= HUD_MISSING_END_SECS - završni
   ekran prekriva HUD; frejm se tada snima u stats/ radi klasifikacije
   (i za sečenje svežih šablona ako su postojeći zastareli).
"""

import os
import time
from datetime import datetime

import cv2

import config
from bot.actions import decode
from bot.state import encode_observation, build_action_mask
from bot.sim.simulator import EpisodeStats
from bot.vision.capture import ScreenCapture
from bot.vision.detector import YoloDetector
from bot.vision.sun_finder import find_suns
from bot.vision.hud_reader import HudReader
from bot.vision.mower_watcher import MowerWatcher
from bot.vision.tracker import TargetTracker
from bot.vision.template_match import (
    find_template, load_template_scale, scale_template,
)
from bot.game.state_builder import GridMapper, build_snapshot
from bot.control.humanizer import Humanizer
from bot.control.executor import ActionExecutor

NOMINAL_LEVEL_DURATION = 230.0  # s - za time_progress komponentu stanja
MINE_ARM_TIME = 15.0            # s - Potato Mine od sadnje do aktivacije
END_CHECK_EVERY = 5             # provera moneybag/gameover svake N-te iteracije
MIN_PLAY_BEFORE_END = 15.0      # s - pre ovoga ne proglašavamo kraj po HUD-u


class RealGameSession:
    def __init__(self, policy):
        self.policy = policy
        self.screen = ScreenCapture()
        self.screen.detect_game_region()  # crne trake oko 4:3 slike
        self.detector = YoloDetector()
        self.hud = HudReader(self.screen)
        self.mowers = MowerWatcher(self.screen)
        self.tracker = TargetTracker()
        self.mapper = GridMapper(self.screen)
        self.humanizer = Humanizer(self.screen.bbox)
        self.stats = EpisodeStats()
        self.executor = ActionExecutor(
            self.humanizer, self.hud, self.mapper, self.tracker, self.stats)

        # svež šablon džaka (isečen sa ove mašine) ima prednost nad v1.0
        new_bag = os.path.join(config.ASSETS_DIR, "moneybag_new.png")
        self._tpl_win_raw = cv2.imread(new_bag if os.path.exists(new_bag)
                                       else config.TEMPLATE_WIN)
        self._tpl_loss_raw = cv2.imread(config.TEMPLATE_LOSS)
        self._tpl_loss_house_raw = cv2.imread(config.TEMPLATE_LOSS_HOUSE)
        self._tpl_ingame_raw = cv2.imread(
            os.path.join(config.ASSETS_DIR, "ingame_check.png"))
        self.tpl_win = self.tpl_loss = self.tpl_ingame = None
        self._last_zombie_time = time.time()
        self._last_bag_hunt = 0.0

        self._danger_seen = set()
        # interni registar posađenih mina: (red, kolona) -> vreme sadnje
        # (YOLO nema klasu za minu, ali mi znamo šta smo posadili)
        self._mines = {}
        # cooldown po karti: sprečava spam sadnje dok se sadnja verifikuje
        # i posle neuspešne sadnje (karta pri kraju punjenja izgleda svetlo
        # pa vizuelni čitač ume da je proglasi spremnom prerano)
        self._card_block_until = {}
        # prozor u kom promena brojača sunca znači potvrđeno kupljenje
        self._collect_window_until = 0.0
        # posle klika na sunca: pauza kupljenja dok pokupljena lete ka brojaču
        self._sun_blind_until = 0.0
        # crna lista "grobova": neuspela sadnja Grave Bustera znači da na
        # tom polju NIJE grob (pravi grob bi sadnju primio) - detekcije
        # groba na tom polju se ignorišu narednih 120 s
        self._grave_blacklist = {}
        # polja skoro otpisanih mina: štite se od lažnih detekcija još 15 s
        # (mina/tragovi vizuelno ostaju i posle otpisa iz registra)
        self._recent_mine_spots = {}

        # skalirani opseg površine sunca za ovu rezoluciju
        game_scale = self.screen.gh / 600.0
        self._sun_area = (config.SUN_AREA_MIN_NATIVE * game_scale ** 2,
                          config.SUN_AREA_MAX_NATIVE * game_scale ** 2)

        self._video = None
        if config.DEBUG_RECORD_VIDEO:
            os.makedirs(config.STATS_DIR, exist_ok=True)
            path = os.path.join(
                config.STATS_DIR,
                f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            self._video = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), 5,
                (self.screen.width // 2, self.screen.height // 2))
            print(f"[DEBUG] Snimam pregled partije u {path}")

    # ---------------------------------------------------------------- helpers
    def _game_crop(self, frame):
        s = self.screen
        return frame[s.gy:s.gy + s.gh, s.gx:s.gx + s.gw]

    def _match_end_template(self, crop_half, template):
        """Traži šablon kraja partije u upola smanjenom okviru igre."""
        if template is None:
            return None
        tpl = cv2.resize(template, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if tpl.shape[0] >= crop_half.shape[0] or tpl.shape[1] >= crop_half.shape[1]:
            return None
        res = cv2.matchTemplate(crop_half, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= config.TEMPLATE_MATCH_THRESHOLD:
            s = self.screen
            h, w = tpl.shape[:2]
            return (s.gx + (max_loc[0] + w // 2) * 2,
                    s.gy + (max_loc[1] + h // 2) * 2)
        return None

    def _hud_visible(self, frame):
        """Brza provera prisustva HUD-a: ikona sunca u gornjem levom uglu
        okvira igre (marker snimljen na ovoj mašini, pouzdan)."""
        if self.tpl_ingame is None:
            return True
        s = self.screen
        corner = frame[s.gy:s.gy + int(0.25 * s.gh), s.gx:s.gx + int(0.20 * s.gw)]
        res = cv2.matchTemplate(corner, self.tpl_ingame, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= 0.70

    def _classify_end(self):
        """Kada HUD nestane, partija je gotova. Sačeka se da se završna
        kinematika razvije, pa se klasifikuje: kinematika "zombiji ulaze u
        kuću" = PORAZ; kraj bez nje = POBEDA (partija ima samo dva ishoda;
        ručni prekid je poseban slučaj i ne stiže dovde)."""
        time.sleep(4.0)
        frame = self.screen.grab_bgr()
        os.makedirs(config.STATS_DIR, exist_ok=True)
        path = os.path.join(
            config.STATS_DIR,
            f"end_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(path, frame)

        if self._tpl_loss_house_raw is not None:
            found, _, _, _, val = find_template(
                frame, self._tpl_loss_house_raw, threshold=0.70, scale=1.0)
            if found:
                print("[AGENT] Poraz - zombiji su ušli u kuću.")
                return "LOSS"
        if self._tpl_loss_raw is not None:
            found, _, _, _, _ = find_template(frame, self._tpl_loss_raw, threshold=0.70)
            if found:
                print("[AGENT] Poraz - THE ZOMBIES ATE YOUR BRAINS.")
                return "LOSS"
        print("[AGENT] 🏆 USPEŠNO ZAVRŠENA PARTIJA - POBEDA!")
        return "WIN"

    def _verify_last_plant(self, cards_by_name, now):
        """Sadnja je uspela samo ako je karta potamnela (ušla u punjenje).
        Ako je posle sadnje karta i dalje svetla, sadnja NIJE prošla (npr.
        polje zauzeto) - vraćamo sunce/statistiku, a za minu blokiramo to
        polje u registru da politika ne pokušava u krug."""
        lp = self.executor.last_plant
        if lp is None:
            return
        card, cell, t_plant = lp
        if now - t_plant < 0.9:
            return  # karta možda još nije osvežena na ekranu
        self.executor.last_plant = None
        if cards_by_name.get(card, False):
            cost = config.PLANT_COSTS[card]
            self.executor.sun_ledger += cost
            self.stats.sun_spent -= cost
            self.stats.plants_used[card] = max(0, self.stats.plants_used[card] - 1)
            # karta na hlađenje - bez ponovnog pokušaja narednih par sekundi
            self._card_block_until[card] = now + 4.0
            print(f"[AGENT] Sadnja {card} na {cell} NIJE uspela "
                  f"(karta i dalje svetla) - pauza za tu kartu 4 s.")
            if card == "potato_mine":
                # blokiraj polje kao (naoružanu) minu - tamo nešto već stoji
                self._mines[cell] = now - MINE_ARM_TIME
            elif card == "gravebuster":
                # pravi grob bi buster primio -> na tom polju NIJE grob
                self._grave_blacklist[cell] = now + 120.0
                print(f"[AGENT] Polje {cell} NIJE grob - na crnoj listi 120 s.")
            elif card == "ice":
                self.executor.freeze_until = 0.0  # zamrzavanje se nije desilo

    def _is_probably_mine(self, tr, now):
        """'Meta' na registrovanoj mini je sama mina (YOLO je ume proglasiti
        zombijem) - pravi zombi HODA. Tik na centru mine (<0.35 ćelije) se
        filtrira BEZ obzira na starost tracka: YOLO treperenje pravi svež
        track na mini svaki put, pa uslov zrelosti ne sme biti obavezan."""
        if not self._mines:
            return False
        zr, _ = self.mapper.zombie_cell(tr)
        cf = self.mapper.col_float(tr.cx)
        for (mr, mc) in self._mines:
            if mr != zr:
                continue
            d = abs(cf - mc)
            if d < 0.35:
                return True
            if d < 0.55 and abs(tr.vx) < 10 and now - tr.created > 1.2:
                return True
        return False

    def _hunt_money_bag(self, suns_px):
        """Kraj nivoa: džak se pojavljuje na mestu poslednjeg ubistva, sa
        žutom strelicom iznad (nju sun-detektor vidi kao 'sunce')."""
        targets = []
        if self.executor.last_kill_px is not None:
            targets.append(self.executor.last_kill_px)
        # klik ISPOD žutih blobova - strelica lebdi iznad džaka
        drop = 0.35 * self.mapper.cell_h
        targets.extend((sx, sy + drop) for (sx, sy, _r) in suns_px[:2])
        while len(targets) < 4:  # dopuni pretragom preko sredine travnjaka
            c = 3 + (len(targets) * 2) % 5
            cx_, cy_ = self.mapper.cell_center_px(2, c)
            targets.append((cx_ - self.screen.left, cy_ - self.screen.top))
        for (tx, ty) in targets[:4]:
            self.humanizer.move_to(self.screen.left + tx, self.screen.top + ty)
            self.humanizer.click()
            time.sleep(0.15)

    def _update_danger_stats(self, targets):
        for tr in targets:
            cf = self.mapper.col_float(tr.cx)
            self.stats.closest_approach = min(self.stats.closest_approach, cf)
            if cf < 1.0 and tr.id not in self._danger_seen:
                self._danger_seen.add(tr.id)
                self.stats.danger_near_count += 1

    def _mines_snapshot(self, now):
        """Ažurira registar mina (mina nestaje kad zombi stane NA nju -
        eksplodirala je ili je pojedena) i vraća [(r, c, armed)].
        Uslov mora biti |pozicija - kolona| <= prag: raniji uslov "levo od
        mine" je brisao sve mine desno od bilo kog zombija u redu."""
        for (r, c), planted_at in list(self._mines.items()):
            for tr in self.tracker.alive(now):
                zr, _ = self.mapper.zombie_cell(tr)
                if zr == r and abs(self.mapper.col_float(tr.cx) - c) <= 0.4:
                    armed = now - planted_at >= MINE_ARM_TIME
                    del self._mines[(r, c)]
                    self._recent_mine_spots[(r, c)] = now
                    if armed:
                        # eksplozija: mete u dometu su leševi - ne gađati ih
                        for tr2 in self.tracker.alive(now):
                            zr2, _ = self.mapper.zombie_cell(tr2)
                            if zr2 == r and abs(self.mapper.col_float(tr2.cx) - c) <= 0.7:
                                tr2.dying_until = now + 1.0
                    break
        return [(r, c, now - t >= MINE_ARM_TIME)
                for (r, c), t in self._mines.items()]

    def _record_debug(self, frame, action_desc):
        if self._video is None:
            return
        small = cv2.resize(frame, (self.screen.width // 2, self.screen.height // 2))
        cv2.putText(small, action_desc, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        self._video.write(small)

    # ------------------------------------------------------------------- play
    def play(self):
        """Igra partiju do kraja; vraća EpisodeStats."""
        # šabloni na razmeri koju je navigator otkrio (skaliranje tek sada -
        # navigacija se dešava POSLE konstrukcije sesije)
        tpl_scale = load_template_scale(default=1.0)
        self.tpl_win = scale_template(self._tpl_win_raw, tpl_scale)
        self.tpl_loss = scale_template(self._tpl_loss_raw, tpl_scale)
        self.tpl_ingame = scale_template(self._tpl_ingame_raw, tpl_scale)

        print("[AGENT] Partija počinje - RL politika preuzima kontrolu.")
        start = time.time()
        frame_idx = 0
        hud_missing_since = None
        # HUD se u novoj partiji pojavljuje TEK kada se pokupi prvo sunce -
        # detekcija kraja po nestanku HUD-a sme da se aktivira tek pošto je
        # HUD viđen (2 uzastopna frejma, protiv YOLO/šuma treperenja)
        hud_seen_streak = 0
        hud_armed = False

        try:
            while True:
                now = time.time()
                elapsed = now - start
                if elapsed > config.GAME_HARD_TIMEOUT:
                    self.stats.result = "TIMEOUT"
                    break

                frame = self.screen.grab_bgr()
                frame_idx += 1
                crop = self._game_crop(frame)

                # kraj partije - šabloni (brzi put, na pola rezolucije)
                if frame_idx % END_CHECK_EVERY == 0:
                    crop_half = cv2.resize(crop, None, fx=0.5, fy=0.5,
                                           interpolation=cv2.INTER_AREA)
                    win_pos = self._match_end_template(crop_half, self.tpl_win)
                    if win_pos is not None:
                        print("[AGENT] 🏆 DŽAK UHVAĆEN - USPEŠNO ZAVRŠENA PARTIJA!")
                        self.humanizer.move_to(self.screen.left + win_pos[0],
                                               self.screen.top + win_pos[1])
                        self.humanizer.click()
                        self.stats.result = "WIN"
                        break
                    if self._match_end_template(crop_half, self.tpl_loss) is not None:
                        self.stats.result = "LOSS"
                        break

                # kraj partije - nestanak HUD-a (završni ekran ga prekriva).
                # HUD ne postoji dok se ne pokupi prvo sunce, pa se ova
                # provera aktivira tek pošto je HUD pouzdano viđen.
                if self._hud_visible(frame):
                    hud_seen_streak += 1
                    if hud_seen_streak >= 2:
                        hud_armed = True
                    hud_missing_since = None
                else:
                    hud_seen_streak = 0
                    if hud_armed and elapsed > MIN_PLAY_BEFORE_END:
                        if hud_missing_since is None:
                            hud_missing_since = now
                        elif now - hud_missing_since >= config.HUD_MISSING_END_SECS:
                            print("[AGENT] HUD je nestao - partija je završena.")
                            self.stats.result = self._classify_end()
                            break

                # percepcija (YOLO i sunca na isečenom okviru igre)
                moving_dets, grave_dets = self.detector.detect(
                    crop, offset=(self.screen.gx, self.screen.gy))
                self.tracker.update(moving_dets, now)
                # gornja margina 0.15 preskače CEO HUD panel (brojač + karte);
                # donja 0.965 = sama ivica travnjaka: 5. red sunaca prolazi,
                # a žuti natpis "Whack a Zombie" ispod (centar ~0.985) ne
                suns_px = find_suns(
                    crop,
                    top_margin_px=int(0.15 * self.screen.gh),
                    bottom_margin_px=int(0.965 * self.screen.gh),
                    area_min=self._sun_area[0], area_max=self._sun_area[1],
                    offset=(self.screen.gx, self.screen.gy))
                # NAŠA posađena mina ume YOLO-u da liči na grob, a HSV filteru
                # na sunce (izrasla mina je žućkasta!) - detekcije oko mina se
                # izbacuju iz OBA kanala, po RASTOJANJU (blob uz ivicu ćelije
                # beži u susednu). Skoro otpisana polja ostaju zaštićena 15 s.
                self._recent_mine_spots = {
                    cell: t for cell, t in self._recent_mine_spots.items()
                    if now - t < 15.0}
                protected = set(self._mines.keys()) | set(self._recent_mine_spots.keys())
                if protected:
                    def near_mine(px, py, frac):
                        for (mr, mc) in protected:
                            mx, my = self.mapper.cell_center_px(mr, mc)
                            if (abs(px - (mx - self.screen.left)) < frac * self.mapper.cell_w
                                    and abs(py - (my - self.screen.top)) < frac * self.mapper.cell_h):
                                return True
                        return False

                    suns_px = [s for s in suns_px if not near_mine(s[0], s[1], 0.65)]
                    grave_dets = [
                        d for d in grave_dets
                        if not near_mine(d.cx, d.cy + 0.2 * (d.y2 - d.y1), 0.8)]

                # grobovi se NE stvaraju u prve 3 kolone - sve detekcije tamo
                # su lažne (najčešće naša mina ili tragovi na travi)
                grave_dets = [
                    d for d in grave_dets
                    if self.mapper.cell_of_px(
                        d.cx, d.cy + 0.2 * (d.y2 - d.y1))[1] >= config.GRAVE_MIN_COL]

                # crna lista: polja gde je buster odbijen NISU grobovi
                if self._grave_blacklist:
                    self._grave_blacklist = {
                        cell: t for cell, t in self._grave_blacklist.items()
                        if t > now}
                    grave_dets = [
                        d for d in grave_dets
                        if self.mapper.cell_of_px(
                            d.cx, d.cy + 0.2 * (d.y2 - d.y1))
                        not in self._grave_blacklist]

                # mete: mina koju YOLO vidi kao zombija se NE računa
                alive_now = [tr for tr in self.tracker.alive(now)
                             if not self._is_probably_mine(tr, now)]
                if alive_now:
                    self._last_zombie_time = now

                slot_states = self.hud.card_states(frame)
                # slot redosled (CARD_ORDER) -> kanonski (gravebuster, mina, led)
                by_name = dict(zip(config.CARD_ORDER, slot_states))
                self._verify_last_plant(by_name, now)

                # potvrda kupljenja sunca: brojač na HUD-u se promenio ubrzo
                # posle klika na sunce -> sunce je STVARNO pokupljeno
                if self.hud.sun_counter_changed(frame) and now <= self._collect_window_until:
                    self.executor.sun_ledger += config.SUN_VALUE
                    self.stats.suns_collected += 1
                card_ready = tuple(
                    by_name[name] and now >= self._card_block_until.get(name, 0.0)
                    for name in ("gravebuster", "potato_mine", "ice"))
                mowers_left = self.mowers.update(frame)
                self.stats.mowers_used = self.mowers.used_count
                self._update_danger_stats(alive_now)
                # adrenalin: više zombija -> brži pokreti (do 30%)
                self.humanizer.set_pressure(len(alive_now) / 10.0)

                # stanje -> politika
                snapshot, cell_tracks = build_snapshot(
                    self.mapper, alive_now, grave_dets, suns_px,
                    card_ready, mowers_left,
                    elapsed / NOMINAL_LEVEL_DURATION,
                    self.executor.sun_ledger, self.executor.freeze_remaining, now,
                    mines=self._mines_snapshot(now))
                obs = encode_observation(snapshot)
                mask = build_action_mask(snapshot)
                action = self.policy.act(obs, mask)
                kind, r, c = decode(action)

                # kraj nivoa: HUD stoji, zombija nema >= 6 s, partija odmakla -
                # džak sa novcem čeka (na mestu poslednjeg ubistva, sa žutom
                # strelicom iznad); hvatanje džaka pokreće završni ekran
                if (hud_armed and hud_missing_since is None
                        and elapsed > 120.0
                        and now - self._last_zombie_time >= 6.0
                        and now - self._last_bag_hunt >= 6.0):
                    self._last_bag_hunt = now

                    # partija je u ovom stanju već DOBIJENA (svi zombiji
                    # počišćeni, ostalo je samo hvatanje džaka) - posle 25 s
                    # bez zombija upiši pobedu i stani, ma šta bilo sa džakom
                    if now - self._last_zombie_time >= 25.0:
                        self.stats.result = "WIN"
                        print("[AGENT] 🏆 USPEŠNO ZAVRŠENA PARTIJA - POBEDA!")
                        break

                    # probaj da NAĐEŠ džak šablonom (multi-scale, blag prag -
                    # u ovoj fazi lažni klik ne košta ništa)
                    if self._tpl_win_raw is not None:
                        found, bx, by, bs, val = find_template(
                            crop, self._tpl_win_raw, threshold=0.60)
                        if found:
                            print("[AGENT] 🏆 DŽAK PRONAĐEN - kupim ga! "
                                  f"(poklapanje {val:.2f}, skala {bs})")
                            self.humanizer.move_to(
                                self.screen.left + self.screen.gx + bx,
                                self.screen.top + self.screen.gy + by)
                            self.humanizer.click()
                            time.sleep(1.2)
                            self.stats.result = "WIN"
                            print("[AGENT] 🏆 USPEŠNO ZAVRŠENA PARTIJA - POBEDA!")
                            break
                        print(f"[AGENT] Džak još nije nađen šablonom "
                              f"(najbolje {val:.2f}) - tražim klikovima...")
                    self._hunt_money_bag(suns_px)
                    continue

                # runtime brava: led se ne troši na malu grupu (politika je
                # trenirana na prag ~22, ovo je osigurač) - umesto toga udri
                # najlevljeg zombija
                if kind == "ice" and len(alive_now) < config.ICE_MIN_ZOMBIES:
                    if alive_now:
                        tr = min(alive_now, key=lambda t: t.cx)
                        ur, uc = self.mapper.zombie_cell(tr)
                        kind, r, c = "whack", ur, uc
                        cell_tracks[(ur, uc)] = tr
                    else:
                        kind, r, c = "wait", None, None
                # obrnuta brava: ogroman talas + spreman led = zamrzni ODMAH
                elif (kind != "ice" and card_ready[2]
                        and len(alive_now) >= config.ICE_FORCE_ZOMBIES):
                    kind, r, c = "ice", None, None
                    self.stats.actions["ice_forced"] = \
                        self.stats.actions.get("ice_forced", 0) + 1

                # slepilo za sunca u letu: pokupljena sunca putuju ka brojaču
                # ~0.7 s i NE jure se ponovo
                if now < self._sun_blind_until and kind == "collect_sun":
                    if alive_now:
                        tr = min(alive_now, key=lambda t: t.cx)
                        ur, uc = self.mapper.zombie_cell(tr)
                        kind, r, c = "whack", ur, uc
                        cell_tracks[(ur, uc)] = tr
                    else:
                        kind, r, c = "wait", None, None

                # prioritet sunca: ima sunca na ekranu i bezbedno je -> puna
                # kasa znači više mina/bustera/leda (sigurnosni sloj ispod
                # i dalje ima poslednju reč ako se pojavi opasnost)
                if (config.SUN_PRIORITY and suns_px
                        and now >= self._sun_blind_until
                        and kind not in ("collect_sun", "ice")):
                    closest = min(
                        (self.mapper.col_float(t.cx) for t in alive_now),
                        default=99.0)
                    if (closest >= config.SUN_PRIORITY_SAFE_COL
                            or (len(alive_now) <= config.SUN_PRIORITY_FEW_ZOMBIES
                                and closest >= config.SUN_PRIORITY_RELAXED_COL)):
                        kind, r, c = "collect_sun", None, None
                        self.stats.actions["sun_priority"] = \
                            self.stats.actions.get("sun_priority", 0) + 1

                # sigurnosni sloj: zombi nadomak kuće ima apsolutni prioritet;
                # pod navalom se zaštitni prag širi (kraj partije = najopasnije)
                if config.SAFETY_OVERRIDE:
                    threshold = config.SAFETY_COL_THRESHOLD
                    if len(alive_now) >= config.SAFETY_SWARM_COUNT:
                        threshold += config.SAFETY_SWARM_BONUS
                    urgent = None
                    for tr in alive_now:
                        cf = self.mapper.col_float(tr.cx)
                        if cf < threshold and (urgent is None or cf < urgent[0]):
                            urgent = (cf, tr)
                    if urgent is not None:
                        ur, uc = self.mapper.zombie_cell(urgent[1])
                        if (kind, r, c) != ("whack", ur, uc):
                            kind, r, c = "whack", ur, uc
                            cell_tracks[(ur, uc)] = urgent[1]
                            self.stats.actions["safety_override"] = \
                                self.stats.actions.get("safety_override", 0) + 1
                self.stats.actions[kind] = self.stats.actions.get(kind, 0) + 1
                self._record_debug(frame, f"{kind} {r},{c}" if r is not None else kind)

                # izvršenje
                self.humanizer.reaction_jitter()
                if kind == "whack":
                    self.executor.whack_cell(r, c, cell_tracks)
                    # usput pokupi sunce nadohvat ruke (kao čovek: udri pa
                    # zgrabi sunce pored) - ali ne dok pokupljena još lete
                    if (suns_px and time.time() >= self._sun_blind_until
                            and self.executor.opportunistic_sun(suns_px)):
                        self._collect_window_until = time.time() + 1.3
                        self._sun_blind_until = time.time() + config.SUN_BLIND_AFTER_COLLECT
                elif kind == "buster":
                    self.executor.plant("gravebuster", r, c)
                    self._card_block_until["gravebuster"] = time.time() + 1.5
                elif kind == "mine":
                    self.executor.plant("potato_mine", r, c)
                    self._mines[(r, c)] = time.time()
                    self._card_block_until["potato_mine"] = time.time() + 1.5
                elif kind == "ice":
                    self.executor.plant_ice(snapshot)
                    self._card_block_until["ice"] = time.time() + 1.5
                elif kind == "collect_sun":
                    self.executor.collect_sun(suns_px)
                    self._collect_window_until = time.time() + 1.3
                    self._sun_blind_until = time.time() + config.SUN_BLIND_AFTER_COLLECT
                else:
                    self.executor.wait()

        except KeyboardInterrupt:
            print("\n[AGENT] Prekinuto od strane korisnika.")
            self.stats.result = "INTERRUPTED"
        finally:
            self.stats.duration = time.time() - start
            if self._video is not None:
                self._video.release()

        return self.stats
