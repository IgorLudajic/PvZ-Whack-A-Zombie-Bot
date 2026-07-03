# -*- coding: utf-8 -*-
"""
Humanizovano upravljanje mišem.

Cilj: posmatrač ne može da razlikuje agenta od čoveka. Elementi:
- trajanje pokreta po Fittsovom zakonu (isto kao u simulatoru - politika
  je trenirana baš sa ovim tajminzima),
- blago zakrivljena Bezier putanja umesto prave linije,
- minimum-jerk profil brzine (ubrzanje pa usporavanje, kao ljudska ruka),
- povremeni mali overshoot sa korekcijom,
- varijabilno trajanje klika i pauza između uzastopnih udaraca.
"""

import math
import random
import time

import pyautogui

import config

pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = True


def _min_jerk(t):
    """Normalizovan minimum-jerk profil pozicije, t u [0, 1]."""
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


class Humanizer:
    def __init__(self, screen_bbox):
        self.bbox = screen_bbox  # (left, top, width, height)

    # ------------------------------------------------------------------ util
    def _clamp(self, x, y):
        l, t, w, h = self.bbox
        return (max(l + 5, min(x, l + w - 5)), max(t + 5, min(y, t + h - 5)))

    def fitts_duration(self, dist):
        d = config.FITTS_A + config.FITTS_B * math.log2(dist / config.FITTS_TARGET_W + 1.0)
        return d + random.uniform(0.0, 0.04)

    # ----------------------------------------------------------------- moves
    def move_to(self, x, y):
        x, y = self._clamp(x, y)
        x0, y0 = pyautogui.position()
        dist = math.hypot(x - x0, y - y0)
        if dist < 3:
            return

        if dist > 250 and random.random() < config.OVERSHOOT_PROB:
            # mali promašaj pa korekcija
            over = random.uniform(0.04, 0.10)
            ox = x + (x - x0) * over + random.uniform(-8, 8)
            oy = y + (y - y0) * over + random.uniform(-8, 8)
            self._glide(x0, y0, *self._clamp(ox, oy), self.fitts_duration(dist))
            x0, y0 = pyautogui.position()
            self._glide(x0, y0, x, y, random.uniform(0.08, 0.14))
        else:
            self._glide(x0, y0, x, y, self.fitts_duration(dist))

    def _glide(self, x0, y0, x1, y1, duration):
        """Pokret duž blago zakrivljene Bezier krive sa minimum-jerk tempom."""
        dist = math.hypot(x1 - x0, y1 - y0)
        # kontrolne tačke: normalno odstupanje proporcionalno dužini pokreta
        perp = (-(y1 - y0) / max(dist, 1), (x1 - x0) / max(dist, 1))
        bend1 = random.gauss(0, 0.06 * dist)
        bend2 = random.gauss(0, 0.04 * dist)
        c1 = (x0 + (x1 - x0) * 0.3 + perp[0] * bend1,
              y0 + (y1 - y0) * 0.3 + perp[1] * bend1)
        c2 = (x0 + (x1 - x0) * 0.7 + perp[0] * bend2,
              y0 + (y1 - y0) * 0.7 + perp[1] * bend2)

        steps = max(int(duration * 90), 4)
        t_start = time.perf_counter()
        for i in range(1, steps + 1):
            s = _min_jerk(i / steps)
            u = 1 - s
            px = (u ** 3 * x0 + 3 * u ** 2 * s * c1[0]
                  + 3 * u * s ** 2 * c2[0] + s ** 3 * x1)
            py = (u ** 3 * y0 + 3 * u ** 2 * s * c1[1]
                  + 3 * u * s ** 2 * c2[1] + s ** 3 * y1)
            pyautogui.moveTo(int(px), int(py), _pause=False)
            target_t = duration * i / steps
            sleep = t_start + target_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
        pyautogui.moveTo(int(x1), int(y1), _pause=False)

    # ---------------------------------------------------------------- clicks
    def click(self):
        pyautogui.mouseDown(_pause=False)
        time.sleep(random.uniform(config.CLICK_HOLD_MIN, config.CLICK_HOLD_MAX))
        pyautogui.mouseUp(_pause=False)

    def multi_click_gap(self):
        time.sleep(random.uniform(config.MULTI_CLICK_GAP_MIN, config.MULTI_CLICK_GAP_MAX))

    def reaction_jitter(self):
        """Mali nasumični zastoj - razbija mašinski ravnomeran ritam petlje."""
        time.sleep(random.uniform(0.02, 0.08))
