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

    def grab_bgr(self) -> np.ndarray:
        img = np.array(self.sct.grab(self.sct.monitors[1]))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def frac_roi(self, frac):
        """(l, t, w, h) frakcije ekrana -> piksel koordinate (x, y, w, h)."""
        l, t, w, h = frac
        return (int(l * self.width), int(t * self.height),
                int(w * self.width), int(h * self.height))
