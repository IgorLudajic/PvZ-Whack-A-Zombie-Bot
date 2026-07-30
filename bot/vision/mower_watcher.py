# -*- coding: utf-8 -*-
"""
Praćenje kosačica: uska traka na levoj ivici svake staze.

Na početku partije snimi se referentni izgled trake za svaku stazu;
kada korelacija sa referencom padne ispod praga (kosačica krenula ili
je nema), staza se obeležava kao potrošena. Broj aktiviranih kosačica
je ključna metrika kvaliteta igre (cilj: 0).
"""

import time

import cv2
import numpy as np

import config
from config import ROWS

SUSTAINED_LOW_SECS = 4.0  # promena mora TRAJATI (zombi u prolazu != kosačica)


class MowerWatcher:
    def __init__(self, screen):
        self.screen = screen
        self.baselines = None
        self.fired = [False] * ROWS
        self.low_since = [None] * ROWS

        lawn_top = screen.gy + config.LAWN_TOP_FRAC * screen.gh
        lawn_bottom = screen.gy + config.LAWN_BOTTOM_FRAC * screen.gh
        row_h = (lawn_bottom - lawn_top) / ROWS
        x = int(screen.gx + config.MOWER_STRIP_LEFT_FRAC * screen.gw)
        w = int(config.MOWER_STRIP_WIDTH_FRAC * screen.gw)
        self.rois = []
        for r in range(ROWS):
            y = int(lawn_top + r * row_h + row_h * config.MOWER_ROW_OFFSET_FRAC)
            h = int(row_h * config.MOWER_ROW_HEIGHT_FRAC)
            self.rois.append((x, y, w, h))

    def capture_baseline(self, frame_bgr):
        self.baselines = [self._roi_gray(frame_bgr, roi) for roi in self.rois]
        self.fired = [False] * ROWS

    def _roi_gray(self, frame_bgr, roi):
        x, y, w, h = roi
        return cv2.cvtColor(frame_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

    def update(self, frame_bgr):
        """Vraća broj preostalih kosačica; usput registruje aktivacije.
        Kosačica se vodi kao potrošena tek kada promena TRAJE (zombi koji
        pregazi zonu promeni sliku samo nakratko)."""
        if self.baselines is None:
            self.capture_baseline(frame_bgr)
            return ROWS
        now = time.time()
        for r in range(ROWS):
            if self.fired[r]:
                continue
            current = self._roi_gray(frame_bgr, self.rois[r])
            base = self.baselines[r]
            if current.shape != base.shape or base.size == 0:
                continue
            res = cv2.matchTemplate(current, base, cv2.TM_CCOEFF_NORMED)
            corr = float(res.max())
            if corr < config.MOWER_CHANGE_THRESHOLD:
                if self.low_since[r] is None:
                    self.low_since[r] = now
                elif now - self.low_since[r] >= SUSTAINED_LOW_SECS:
                    self.fired[r] = True
            else:
                self.low_since[r] = None
        return ROWS - sum(self.fired)

    @property
    def used_count(self):
        return sum(self.fired)
