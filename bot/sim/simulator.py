# -*- coding: utf-8 -*-
"""
Simulator mini-igre "Whack-a-Zombie" za trening RL agenta.

Modelira: grobove i talase zombija (običan/čunj/kanta = 1/2/3 udarca),
ispadanje sunca pri ubistvu, sve tri biljke (Potato Mine, Grave Buster,
Ice-shroom) sa cenom i punjenjem, kosačice i uslove pobede/poraza.
Potato Mine se aktivira ~15 s posle sadnje i eksplodira kada zombi
nagazi (uloga "rezervne kosačice"); nenaoružanu minu zombi pojede.

Ključna ideja: svaka akcija agenta traje onoliko koliko bi trajala čoveku
(vreme reakcije + Fittsov zakon za pokret miša + trajanje klikova), a svet
teče tokom izvršavanja akcije. Politika tako uči strategiju koja je
izvodljiva ljudskom brzinom - u špicevima talasa čekić fizički nije
dovoljan, pa se isplati štedeti sunce za Cherry Bomb / Ice-shroom.

Parametri spawna i tajminga su aproksimacija prave igre i nasumično se
variraju po epizodi (domain randomization) kako bi politika bila robusna
na razliku simulator-igra.
"""

import math
import random
from dataclasses import dataclass, field

import config
from config import ROWS, COLS
from bot.actions import decode, ACTION_WAIT
from bot.state import Snapshot, ZombieInfo, HITS_BY_KIND, find_free_cell

WORLD_TICK = 0.05  # s - rezolucija simulacije unutar trajanja jedne akcije


# ----------------------------------------------------------------------------
# Nagrade (obrazloženje u radu: gusta nagrada za ubistva/resurse, kazne
# vezane za približavanje kući daju bliži signal od samog poraza)
# ----------------------------------------------------------------------------
R_KILL = 0.10
R_SUN = 0.05
R_BUSTED_GRAVE = 0.30
R_MISS = -0.05
R_DANGER_MID = -0.20    # zombi prešao u levu trećinu (kolona < 3)
R_DANGER_NEAR = -0.40   # zombi nadomak kuće (kolona < 1)
R_MOWER = -5.0
R_LOSS = -20.0
R_WIN = 10.0


@dataclass
class SimZombie:
    kind: str
    row: int
    x: float                 # pozicija u jedinicama kolona (8 = desno, <0 = kuća)
    rise_remaining: float
    hits_remaining: int
    speed: float             # kolona/s
    passed_mid: bool = False
    passed_near: bool = False


@dataclass
class SimPlant:
    kind: str                # 'ice' (jednokratni efekat sa tajmerom)
    row: int
    col: int
    timer: float             # vreme do efekta


@dataclass
class SimMine:
    row: int
    col: int
    arm_remaining: float     # >0 dok se aktivira; <=0 = naoružana


@dataclass
class SimSun:
    row: int
    col: int
    expires_at: float


@dataclass
class EpisodeStats:
    duration: float = 0.0
    result: str = ""
    kills: dict = field(default_factory=lambda: {"zombie": 0, "conehead": 0, "buckethead": 0})
    clicks: int = 0
    suns_collected: int = 0
    sun_spent: int = 0
    plants_used: dict = field(default_factory=lambda: {"gravebuster": 0, "potato_mine": 0, "ice": 0})
    mowers_used: int = 0
    actions: dict = field(default_factory=dict)
    closest_approach: float = float(COLS)  # najmanja x-pozicija zombija (bezbednosna margina)
    danger_near_count: int = 0             # koliko je zombija ušlo u kolonu < 1

    @property
    def total_kills(self):
        return sum(self.kills.values())


class WhackSimulator:
    """Okruženje u stilu reset()/step(), bez spoljnih zavisnosti (čist numpy/python)."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self):
        rng = self.rng
        self.t = 0.0
        self.sun_bank = float(rng.uniform(100, 175))
        self.mowers = [True] * ROWS
        self.zombies = []
        self.plants = []
        self.mines = []
        self.suns = []
        self.freeze_remaining = 0.0
        self.slow_remaining = 0.0
        self.done = False
        self.stats = EpisodeStats()

        # kursor počinje na sredini ekrana
        self._cursor = (config.NOMINAL_SCREEN_W * 0.5, config.NOMINAL_SCREEN_W * 0.3)

        # nesavršenost percepcije i nišanjenja (preslikava realan sistem:
        # YOLO povremeno propusti detekciju, klik u gužvi ume da promaši)
        self.p_click_miss = rng.uniform(0.05, 0.12)
        self.p_detect_miss = rng.uniform(0.02, 0.07)

        # domain randomization parametara po epizodi; sunce pada u TROJKAMA
        # (potvrđeno u pravoj igri), pa je verovatnoća ispuštanja niža
        self.sun_drop_prob = rng.uniform(0.09, 0.15)
        self.zombie_speed = {
            "zombie": rng.uniform(0.180, 0.240),
            "conehead": rng.uniform(0.170, 0.220),
            "buckethead": rng.uniform(0.140, 0.185),
        }
        self.rise_range = (rng.uniform(0.9, 1.2), rng.uniform(1.8, 2.4))
        self.mine_arm_time = rng.uniform(13.0, 16.0)
        self.recharge_time = {"gravebuster": 7.5, "potato_mine": 30.0, "ice": 30.0}
        self.recharge = {"gravebuster": 0.0, "potato_mine": 0.0, "ice": 0.0}

        # početni raspored grobova - u pravoj igri se grobovi NE stvaraju
        # u prve 3 kolone (potvrđeno posmatranjem)
        self.graves = {}  # (r, c) -> 'idle' | float (vreme do nestanka pod busterom)
        n_graves = rng.randint(10, 14)
        cells = [(r, c) for r in range(ROWS) for c in range(3, COLS)]
        rng.shuffle(cells)
        for rc in cells[:n_graves]:
            self.graves[rc] = "idle"

        self._build_spawn_schedule()
        return self._snapshot()

    def _build_spawn_schedule(self):
        """Raspored spawna: tri faze rastućeg intenziteta + 3 talasa (burst)."""
        rng = self.rng
        T = rng.uniform(200.0, 235.0)
        self.level_spawn_end = T
        schedule = []

        def phase(t0, t1, lo, hi):
            t = t0
            while t < t1:
                t += rng.uniform(lo, hi)
                if t < t1:
                    schedule.append(t)

        def burst(t0, count, spread):
            for _ in range(count):
                schedule.append(t0 + rng.uniform(0, spread))

        phase(0.0, 0.18 * T, 2.0, 3.0)
        phase(0.18 * T, 0.40 * T, 1.2, 1.8)
        burst(0.40 * T, rng.randint(10, 14), 5.0)      # talas 1
        phase(0.40 * T, 0.65 * T, 0.8, 1.2)
        burst(0.65 * T, rng.randint(14, 18), 6.0)      # talas 2
        phase(0.65 * T, 0.90 * T, 0.60, 0.90)
        burst(0.90 * T, rng.randint(26, 34), 7.0)      # finalni talas
        schedule.sort()

        # tip zombija zavisi od napretka nivoa
        self.spawn_queue = []
        for st in schedule:
            p = st / T
            cone_p = 0.10 + 0.25 * p
            bucket_p = 0.05 + 0.15 * p
            r = rng.random()
            if r < bucket_p:
                kind = "buckethead"
            elif r < bucket_p + cone_p:
                kind = "conehead"
            else:
                kind = "zombie"
            self.spawn_queue.append((st, kind))

        # novi grobovi iskaču sa talasima (kao u igri); pamtimo trenutke
        self.grave_wave_times = [0.40 * T, 0.65 * T, 0.90 * T]

    # ------------------------------------------------------------- geometrija
    def _cell_px(self, row, col):
        W = config.NOMINAL_SCREEN_W
        H = W * 9.0 / 16.0
        lw = (config.LAWN_RIGHT_FRAC - config.LAWN_LEFT_FRAC) * W
        lh = (config.LAWN_BOTTOM_FRAC - config.LAWN_TOP_FRAC) * H
        x = config.LAWN_LEFT_FRAC * W + (col + 0.5) * lw / COLS
        y = config.LAWN_TOP_FRAC * H + (row + 0.5) * lh / ROWS
        return x, y

    def _card_px(self, slot):
        W = config.NOMINAL_SCREEN_W
        H = W * 9.0 / 16.0
        l, t, w, h = config.CARD_SLOTS_FRAC[slot]
        return (l + w / 2) * W, (t + h / 2) * H

    def _move_time(self, target_px):
        """Fittsov zakon - vreme pokreta miša do mete."""
        d = math.hypot(target_px[0] - self._cursor[0], target_px[1] - self._cursor[1])
        t = config.FITTS_A + config.FITTS_B * math.log2(d / config.FITTS_TARGET_W + 1.0)
        if self.rng.random() < config.OVERSHOOT_PROB:
            t += self.rng.uniform(0.06, 0.14)  # promašaj pa korekcija
        self._cursor = target_px
        return t

    def _reaction(self):
        return self.rng.uniform(config.HUMAN_REACTION_MIN, config.HUMAN_REACTION_MAX)

    def _click_time(self):
        return self.rng.uniform(config.CLICK_HOLD_MIN, config.CLICK_HOLD_MAX)

    def _gap_time(self):
        return self.rng.uniform(config.MULTI_CLICK_GAP_MIN, config.MULTI_CLICK_GAP_MAX)

    # ------------------------------------------------------------------ world
    def _advance(self, duration):
        """Pomera svet za 'duration' sekundi; vraća akumuliranu nagradu
        iz pasivnih događaja (kazne za opasnost, kosačice, poraz)."""
        reward = 0.0
        remaining = duration
        while remaining > 1e-9 and not self.done:
            dt = min(WORLD_TICK, remaining)
            remaining -= dt
            self.t += dt

            # zamrzavanje / usporavanje
            speed_mult = 1.0
            if self.freeze_remaining > 0:
                self.freeze_remaining -= dt
                speed_mult = 0.0
                if self.freeze_remaining <= 0:
                    self.slow_remaining = 5.0
            elif self.slow_remaining > 0:
                self.slow_remaining -= dt
                speed_mult = 0.5

            # spawn
            while self.spawn_queue and self.spawn_queue[0][0] <= self.t:
                _, kind = self.spawn_queue.pop(0)
                self._spawn_zombie(kind)

            # novi grobovi sa talasima
            while self.grave_wave_times and self.grave_wave_times[0] <= self.t:
                self.grave_wave_times.pop(0)
                self._add_wave_graves()

            # biljke
            for p in list(self.plants):
                p.timer -= dt
                if p.timer <= 0:
                    reward += self._trigger_plant(p)
                    self.plants.remove(p)

            # mine: aktivacija + okidanje/gutanje pri nagazu
            for m in list(self.mines):
                if m.arm_remaining > 0:
                    m.arm_remaining -= dt
                stepped = [z for z in self.zombies
                           if z.row == m.row and z.rise_remaining <= 0
                           and abs(z.x - m.col) <= 0.35]
                if not stepped:
                    continue
                self.mines.remove(m)
                if m.arm_remaining <= 0:
                    # eksplozija - ubija sve na ćeliji mine
                    for z in [z for z in self.zombies
                              if z.row == m.row and abs(z.x - m.col) <= 0.6]:
                        self.zombies.remove(z)
                        self.stats.kills[z.kind] += 1
                        reward += R_KILL
                # nenaoružanu minu zombi pojede - propala investicija

            # grobovi koje Grave Buster jede
            for rc, state in list(self.graves.items()):
                if state != "idle":
                    new_t = state - dt
                    if new_t <= 0:
                        del self.graves[rc]
                        reward += R_BUSTED_GRAVE
                    else:
                        self.graves[rc] = new_t

            # zombiji
            for z in list(self.zombies):
                if z.rise_remaining > 0:
                    if speed_mult > 0:
                        z.rise_remaining -= dt
                    continue
                z.x -= z.speed * speed_mult * dt
                self.stats.closest_approach = min(self.stats.closest_approach, z.x)
                if not z.passed_mid and z.x < 3.0:
                    z.passed_mid = True
                    reward += R_DANGER_MID
                if not z.passed_near and z.x < 1.0:
                    z.passed_near = True
                    self.stats.danger_near_count += 1
                    reward += R_DANGER_NEAR
                if z.x < -0.6:
                    if self.mowers[z.row]:
                        self.mowers[z.row] = False
                        self.stats.mowers_used += 1
                        reward += R_MOWER
                        self.zombies = [o for o in self.zombies if o.row != z.row]
                    else:
                        self.done = True
                        self.stats.result = "LOSS"
                        reward += R_LOSS
                        break

            # sunca nestaju
            self.suns = [s for s in self.suns if s.expires_at > self.t]

            # punjenje karata
            for k in self.recharge:
                self.recharge[k] = max(0.0, self.recharge[k] - dt)

            # pobeda: nema više spawna i teren je čist
            if (not self.done and not self.spawn_queue
                    and not self.zombies and self.t > self.level_spawn_end):
                self.done = True
                self.stats.result = "WIN"
                reward += R_WIN

            # sigurnosni limit
            if not self.done and self.t > config.GAME_HARD_TIMEOUT:
                self.done = True
                self.stats.result = "TIMEOUT"
                reward += R_LOSS / 2

        return reward

    def _spawn_zombie(self, kind):
        idle = [rc for rc, st in self.graves.items() if st == "idle"]
        if not idle:
            return
        r, c = self.rng.choice(idle)
        lo, hi = self.rise_range
        self.zombies.append(SimZombie(
            kind=kind, row=r, x=float(c),
            rise_remaining=self.rng.uniform(lo, hi),
            hits_remaining=HITS_BY_KIND[kind],
            speed=self.zombie_speed[kind] * self.rng.uniform(0.9, 1.1),
        ))

    def _add_wave_graves(self):
        free = [(r, c) for r in range(ROWS) for c in range(3, COLS)
                if (r, c) not in self.graves]
        self.rng.shuffle(free)
        target_min = 6
        need = max(target_min - len(self.graves), 0) + 2
        for rc in free[:need]:
            self.graves[rc] = "idle"

    def _trigger_plant(self, plant):
        if plant.kind == "ice":
            self.freeze_remaining = 4.5
        return 0.0

    # ------------------------------------------------------------------- step
    def step(self, action_id):
        """Izvršava akciju; vraća (snapshot, reward, done, info).
        info['dt'] je realno trajanje akcije (za SMDP diskontovanje)."""
        assert not self.done, "Epizoda je završena - pozovi reset()"
        kind, r, c = decode(action_id)
        self.stats.actions[kind] = self.stats.actions.get(kind, 0) + 1
        reward = 0.0
        t0 = self.t

        if kind == "whack":
            reward += self._do_whack(r, c)
        elif kind == "buster":
            reward += self._do_plant_card("gravebuster", r, c)
        elif kind == "mine":
            reward += self._do_plant_card("potato_mine", r, c)
        elif kind == "ice":
            snap = self._snapshot()
            cell = find_free_cell(snap)
            if cell is not None:
                reward += self._do_plant_card("ice", cell[0], cell[1])
            else:
                reward += self._advance(0.2) + R_MISS
        elif kind == "collect_sun":
            reward += self._do_collect_sun()
        else:  # wait
            reward += self._advance(0.25)

        dt = self.t - t0
        info = {"dt": max(dt, 1e-3), "stats": self.stats}
        if self.done:
            self.stats.duration = self.t
            if not self.stats.result:
                self.stats.result = "LOSS"
        return self._snapshot(), reward, self.done, info

    def _find_whack_target(self, r, c):
        """Meta u ćeliji (r, c); čekić 'prati' metu pa tolerišemo pomeraj
        do pola kolone. Bira se najlevlja (najopasnija) u ćeliji."""
        best = None
        for z in self.zombies:
            if z.row == r and abs(z.x - c) <= 0.7:
                if best is None or z.x < best.x:
                    best = z
        return best

    def _do_whack(self, r, c):
        reward = self._advance(self._reaction() + self._move_time(self._cell_px(r, c)))
        if self.done:
            return reward
        target = self._find_whack_target(r, c)
        if target is None:
            return reward + R_MISS

        hits = target.hits_remaining
        for i in range(hits):
            reward += self._advance(self._click_time())
            self.stats.clicks += 1
            if self.done:
                return reward
            if target not in self.zombies:
                break  # ubila ga višnja/kosačica u međuvremenu
            if self.rng.random() < self.p_click_miss:
                continue  # promašaj - klik potrošen, zombi neoštećen
            target.hits_remaining -= 1
            if target.hits_remaining <= 0:
                self.zombies.remove(target)
                self.stats.kills[target.kind] += 1
                reward += R_KILL
                if self.rng.random() < self.sun_drop_prob:
                    # zombi ispušta TROJKU sunaca (kao u pravoj igri)
                    col = max(0, min(COLS - 1, int(round(target.x))))
                    for _ in range(3):
                        self.suns.append(SimSun(target.row, col, self.t + 8.0))
                break
            if i < hits - 1:
                reward += self._advance(self._gap_time())
                if self.done:
                    return reward
        return reward

    def _do_plant_card(self, card, r, c):
        cost = config.PLANT_COSTS[card]
        slot = config.CARD_ORDER.index(card)
        # klik na kartu
        reward = self._advance(
            self._reaction() + self._move_time(self._card_px(slot)) + self._click_time()
        )
        if self.done:
            return reward
        if self.recharge[card] > 0 or self.sun_bank < cost:
            return reward + R_MISS
        # klik na ćeliju
        reward += self._advance(self._move_time(self._cell_px(r, c)) + self._click_time())
        if self.done:
            return reward

        mine_cells = {(m.row, m.col) for m in self.mines}
        if card == "gravebuster":
            if self.graves.get((r, c)) != "idle":
                return reward + R_MISS
            self.graves[(r, c)] = 3.0  # buster jede grob 3 s
            self.stats.plants_used["gravebuster"] += 1
        elif card == "potato_mine":
            if (r, c) in self.graves or (r, c) in mine_cells:
                return reward + R_MISS
            self.mines.append(SimMine(r, c, self.mine_arm_time))
            self.stats.plants_used["potato_mine"] += 1
        elif card == "ice":
            if (r, c) in self.graves or (r, c) in mine_cells:
                return reward + R_MISS
            self.plants.append(SimPlant("ice", r, c, 1.0))
            self.stats.plants_used["ice"] += 1

        self.sun_bank -= cost
        self.stats.sun_spent += cost
        self.recharge[card] = self.recharge_time[card]
        return reward

    def _do_collect_sun(self):
        if not self.suns:
            return self._advance(0.2) + R_MISS
        target = min(self.suns, key=lambda s: self._dist_to_cell(s.row, s.col))
        reward = self._advance(
            self._reaction() + self._move_time(self._cell_px(target.row, target.col))
            + self._click_time()
        )
        if target in self.suns:
            self.suns.remove(target)
            self.sun_bank += config.SUN_VALUE
            self.stats.suns_collected += 1
            reward += R_SUN
        else:
            reward += R_MISS  # sunce isteklo dok je miš putovao
        return reward

    def _dist_to_cell(self, r, c):
        px = self._cell_px(r, c)
        return math.hypot(px[0] - self._cursor[0], px[1] - self._cursor[1])

    # --------------------------------------------------------------- snapshot
    def _card_ready(self, card):
        return self.recharge[card] <= 0 and self.sun_bank >= config.PLANT_COSTS[card]

    def _snapshot(self):
        zombies = []
        for z in self.zombies:
            if self.rng.random() < self.p_detect_miss:
                continue  # YOLO "flicker" - zombi nevidljiv u ovom frejmu
            col = max(0, min(COLS - 1, int(round(z.x))))
            zombies.append(ZombieInfo(
                row=z.row, col=col, kind=z.kind,
                hits_remaining=z.hits_remaining,
                rising=z.rise_remaining > 0,
            ))
        return Snapshot(
            zombies=zombies,
            graves={rc for rc, st in self.graves.items()
                    if st == "idle" and self.rng.random() > self.p_detect_miss},
            suns=[(s.row, s.col) for s in self.suns],
            mines=[(m.row, m.col, m.arm_remaining <= 0) for m in self.mines],
            sun_bank=self.sun_bank,
            card_ready=(
                self._card_ready("gravebuster"),
                self._card_ready("potato_mine"),
                self._card_ready("ice"),
            ),
            mowers_left=sum(self.mowers),
            time_progress=self.t / self.level_spawn_end,
            freeze_remaining=max(self.freeze_remaining, 0.0),
        )
