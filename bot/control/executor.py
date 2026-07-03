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

    # ------------------------------------------------------------------ akcije
    def whack_cell(self, row, col, cell_tracks):
        track = cell_tracks.get((row, col))
        if track is None:
            return
        now = time.time()
        hits = track.hits_remaining(now)
        if hits <= 0:
            return
        ax, ay = track.aim_px
        self.human.move_to(self.mapper.screen.left + ax, self.mapper.screen.top + ay)
        for i in range(hits):
            self.human.click()
            self.stats.clicks += 1
            self.tracker.register_hit(track)
            if i < hits - 1:
                self.human.multi_click_gap()
        if track.hits_remaining(time.time()) <= 0:
            self.stats.kills[track.kind] += 1

    def collect_sun(self, suns_px):
        if not suns_px:
            return
        import pyautogui
        cx, cy = pyautogui.position()
        sx, sy = min(suns_px, key=lambda s: (s[0] + self.mapper.screen.left - cx) ** 2
                                            + (s[1] + self.mapper.screen.top - cy) ** 2)
        self.human.move_to(self.mapper.screen.left + sx, self.mapper.screen.top + sy)
        self.human.click()
        self.sun_ledger += config.SUN_VALUE
        self.stats.suns_collected += 1

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
        time.sleep(random.uniform(0.15, 0.30))

    @property
    def freeze_remaining(self):
        return max(0.0, self.freeze_until - time.time())
