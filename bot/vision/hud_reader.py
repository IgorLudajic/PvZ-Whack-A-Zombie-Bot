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

    def card_states(self, frame_bgr):
        """Vraća tuple bool-ova (gravebuster, cherry, ice) - da li je karta
        spremna za sadnju (napunjena + priuštiva)."""
        states = []
        for (x, y, w, h) in self.card_rois:
            roi = frame_bgr[y:y + h, x:x + w]
            if roi.size == 0:
                states.append(False)
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            sat = float(np.mean(hsv[:, :, 1]))
            val = float(np.mean(hsv[:, :, 2]))
            states.append(sat >= config.CARD_READY_SATURATION
                          and val >= config.CARD_READY_VALUE)
        return tuple(states)

    def card_center_px(self, slot):
        x, y, w, h = self.card_rois[slot]
        return (self.screen.left + x + w // 2, self.screen.top + y + h // 2)
