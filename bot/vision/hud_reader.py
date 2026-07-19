# -*- coding: utf-8 -*-
"""
Čitanje HUD-a: spremnost karata sa semenima.

Karta u PvZ-u je obojena i svetla samo kada je i napunjena i priuštiva;
dok se puni ili nema dovoljno sunca, prekrivena je tamno-sivim slojem.
Zato je prosečna saturacija + svetlina ROI-ja pouzdan indikator spremnosti,
bez potrebe za OCR-om brojača sunca.
"""

import cv2
import numpy as np

import config


class HudReader:
    def __init__(self, screen):
        self.screen = screen
        self.card_rois = [screen.frac_roi(f) for f in config.CARD_SLOTS_FRAC]
        self._counter_roi = screen.frac_roi(config.SUN_COUNTER_FRAC)
        self._counter_prev = None

    def card_states(self, frame_bgr):
        """Vraća tuple bool-ova po SLOTU (redosled config.CARD_ORDER) - da li
        je karta spremna za sadnju (napunjena + priuštiva = svetla).
        Prag svetline je poseban po slotu, kalibrisan iz snimaka spremnih
        karata (~85% izmerene svetline; tamni sloj obara svetlinu ~50%)."""
        states = []
        for slot, (x, y, w, h) in enumerate(self.card_rois):
            roi = frame_bgr[y:y + h, x:x + w]
            if roi.size == 0:
                states.append(False)
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            val = float(np.mean(hsv[:, :, 2]))
            states.append(val >= config.CARD_READY_VALUE_PER_SLOT[slot])
        return tuple(states)

    def card_center_px(self, slot):
        x, y, w, h = self.card_rois[slot]
        return (self.screen.left + x + w // 2, self.screen.top + y + h // 2)

    def sun_counter_changed(self, frame_bgr):
        """True ako se prikaz brojača sunca promenio od prošlog poziva.
        Brojač se menja SAMO kada se sunce doda ili potroši - promena posle
        klika na sunce je pouzdana potvrda da je sunce stvarno pokupljeno
        (bez OCR-a: poredi se donja trećina ROI-ja gde je sam broj)."""
        x, y, w, h = self._counter_roi
        strip = frame_bgr[y + int(h * 0.60):y + h, x:x + w]
        if strip.size == 0:
            return False
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        prev, self._counter_prev = self._counter_prev, gray
        if prev is None or prev.shape != gray.shape:
            return False
        diff = float(np.mean(cv2.absdiff(prev, gray)))
        return diff > 4.0
