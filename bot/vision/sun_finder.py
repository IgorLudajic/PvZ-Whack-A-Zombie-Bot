# -*- coding: utf-8 -*-
"""HSV detekcija sunca (žute konture odgovarajuće površine)."""

import cv2
import numpy as np

import config

_LOWER = np.array(config.SUN_HSV_LOWER)
_UPPER = np.array(config.SUN_HSV_UPPER)


def find_suns(frame_bgr, top_margin_px=100):
    """Vraća listu (x, y) centara sunca u pikselima ekrana.
    Gornja margina isključuje HUD (brojač sunca je takođe žut)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _LOWER, _UPPER)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    suns = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if config.SUN_AREA_MIN < area < config.SUN_AREA_MAX:
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            if cy < top_margin_px:
                continue
            suns.append((cx, cy))
    return suns
