# -*- coding: utf-8 -*-
"""
Izvršni sloj: visokonivovsku akciju politike pretvara u humanizovane
poteze mišem. Ovde se sprovodi pravilo 1/2/3 udarca po tipu zombija.
"""

import random
import time

import config
from bot.state import find_free_cell
from bot.vision.tracker import KIND_OF_CLASS


class ActionExecutor:
    def __init__(self, humanizer, hud_reader, mapper, tracker, stats):
        self.human = humanizer
        self.hud = hud_reader
        self.mapper = mapper
        self.tracker = tracker
        self.stats = stats
        self.sun_ledger = float(config.START_SUN)
        self.freeze_until = 0.0
        # poslednja sadnja, za verifikaciju uspeha (karta mora da potamni)
        self.last_plant = None  # (card, (row, col), vreme)
        # mesto poslednjeg ubistva - tu se na kraju partije pojavljuje džak
        self.last_kill_px = None

    # ------------------------------------------------------------------ akcije
    FRAME_AGE = 0.20  # s - prosečna starost percepcije (kaptura + YOLO)

    def whack_cell(self, row, col, cell_tracks):
        track = cell_tracks.get((row, col))
        if track is None:
            return
        now = time.time()
        hits = track.hits_remaining(now)
        if hits <= 0:
            return
        # prediktivni nišan: koliko će meta odmaći dok miš stigne + starost
        # frejma; mlad track (<0.6 s) ima nestabilnu procenu brzine - bez leada
        ax, ay = track.aim_px
        abs_x = self.mapper.screen.left + ax
        abs_y = self.mapper.screen.top + ay
        vx = track.vx if now - track.created > 0.6 else 0.0
        lead = self.FRAME_AGE + self.human.expected_move_time(abs_x, abs_y)
        px, py = ax + vx * lead, ay
        # ne nišani levlje od travnjaka (traka kosačica nije meta) niti
        # iznad njega (klik u HUD "uzme" kartu -> zaglavljen paket!)
        px = max(px, self.mapper.left + 10)
        py = max(py, self.mapper.top + 12)
        self.human.move_to(self.mapper.screen.left + px, self.mapper.screen.top + py)
        gap_avg = (config.MULTI_CLICK_GAP_MIN + config.MULTI_CLICK_GAP_MAX) / 2
        for i in range(hits):
            self.human.click()
            self.stats.clicks += 1
            self.tracker.register_hit(track)
            if i < hits - 1:
                self.human.multi_click_gap()
                # meta hoda i tokom serije udaraca - ruka je prati
                if abs(vx) > 5:
                    self.human.micro_shift(vx * gap_avg, 0)
        if track.hits_remaining(time.time()) <= 0:
            self.stats.kills[track.kind] += 1
            self.last_kill_px = (px, py)

    def collect_sun(self, suns_px):
        """Klik na najbliži sun-blob; veliki blob (trojka sunaca) se
        "počisti" dodatnim klikovima preko cele grupe.

        NAPOMENA: registar sunca se NE uvećava ovde - pokupljeno sunce se
        priznaje tek kada se brojač na HUD-u stvarno promeni (real_env)."""
        if not suns_px:
            return
        import pyautogui
        sl, st = self.mapper.screen.left, self.mapper.screen.top
        cx, cy = pyautogui.position()
        # počisti do 3 najbliža blob-a u jednom potezu (kao čovek koji
        # pređe preko svih sunaca na travnjaku)
        targets = sorted(
            suns_px,
            key=lambda s: (s[0] + sl - cx) ** 2 + (s[1] + st - cy) ** 2)[:4]
        for (sx, sy, sr) in targets:
            self.human.move_to(sl + sx, st + sy)
            self.human.click()
            if sr >= 45:  # veći od pojedinačnog sunca -> grupa; pređi preko nje
                # pomaci su relativni: levo od centra, pa preko centra na desno
                for dx in (-0.7 * sr, 1.4 * sr):
                    time.sleep(random.uniform(0.05, 0.09))
                    self.human.micro_shift(dx, random.uniform(-6, 6))
                    self.human.click()

    def opportunistic_sun(self, suns_px, radius=320):
        """Usputno kupljenje: ako je sunce nadohvat ruke (blizu trenutne
        pozicije kursora), klikni ga u prolazu. Vraća True ako je kliknuto."""
        import pyautogui
        sl, st = self.mapper.screen.left, self.mapper.screen.top
        cx, cy = pyautogui.position()
        best = None
        best_d2 = radius * radius
        for (sx, sy, _sr) in suns_px:
            d2 = (sx + sl - cx) ** 2 + (sy + st - cy) ** 2
            if d2 < best_d2:
                best, best_d2 = (sx, sy), d2
        if best is None:
            return False
        self.human.move_to(sl + best[0], st + best[1])
        self.human.click()
        return True

    def plant(self, card, row, col):
        """card: 'gravebuster' | 'cherry' | 'ice'; klik na kartu pa na ćeliju."""
        slot = config.CARD_ORDER.index(card)
        cx, cy = self.hud.card_center_px(slot)
        self.human.move_to(cx, cy)
        self.human.click()
        time.sleep(random.uniform(0.10, 0.18))
        tx, ty = self.mapper.cell_center_px(row, col)
        self.human.move_to(tx, ty)
        self.human.click()

        self.sun_ledger = max(0.0, self.sun_ledger - config.PLANT_COSTS[card])
        self.stats.sun_spent += config.PLANT_COSTS[card]
        self.stats.plants_used[card] += 1
        self.last_plant = (card, (row, col), time.time())
        if card == "ice":
            self.freeze_until = time.time() + 5.5
        time.sleep(random.uniform(0.10, 0.20))

        # osiguranje: ako sadnja nije uspela (zauzeto polje, kriva detekcija),
        # paket semena ostaje na kursoru i blokirao bi čekić - desni klik ga
        # otkazuje, a ne radi ništa ako je sadnja prošla
        import pyautogui
        pyautogui.rightClick(_pause=False)
        time.sleep(random.uniform(0.05, 0.10))

    def plant_ice(self, snapshot):
        cell = find_free_cell(snapshot)
        if cell is not None:
            self.plant("ice", cell[0], cell[1])

    def wait(self):
        time.sleep(random.uniform(0.10, 0.20))

    @property
    def freeze_remaining(self):
        return max(0.0, self.freeze_until - time.time())
