# -*- coding: utf-8 -*-
"""Kaptura ekrana (primarni monitor) preko mss-a."""

import ctypes

import cv2
import mss
import numpy as np

try:
    ctypes.windll.user32.SetProcessDPIAware()
except AttributeError:
    pass


class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()
        mon = self.sct.monitors[1]
        self.left, self.top = mon["left"], mon["top"]
        self.width, self.height = mon["width"], mon["height"]
        self.bbox = (self.left, self.top, self.width, self.height)
        # okvir igre unutar frejma (podrazumevano ceo ekran; vidi detect_game_region)
        self.gx, self.gy, self.gw, self.gh = 0, 0, self.width, self.height

    def grab_bgr(self) -> np.ndarray:
        img = np.array(self.sct.grab(self.sct.monitors[1]))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def detect_game_region(self):
        """Detektuje okvir igre (4:3 slika između crnih traka na 16:9
        monitoru). Pozvati dok je igra na ekranu, pre gradnje ROI-jeva."""
        from bot.vision.game_region import detect_game_region
        frame = self.grab_bgr()
        self.gx, self.gy, self.gw, self.gh = detect_game_region(frame)
        print(f"[VISION] Okvir igre: x={self.gx}, y={self.gy}, "
              f"{self.gw}x{self.gh} (monitor {self.width}x{self.height})")
        return (self.gx, self.gy, self.gw, self.gh)

    def frac_roi(self, frac):
        """(l, t, w, h) frakcije OKVIRA IGRE -> piksel koordinate frejma."""
        l, t, w, h = frac
        return (int(self.gx + l * self.gw), int(self.gy + t * self.gh),
                int(w * self.gw), int(h * self.gh))
