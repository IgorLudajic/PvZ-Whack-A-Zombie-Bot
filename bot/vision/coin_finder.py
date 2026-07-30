# -*- coding: utf-8 -*-
"""
Detekcija srebrnih novčića i dijamanata (bonus pokupljivi predmeti).

Srebrni novčić: svetao, gotovo bez boje (niska saturacija), okrugao -
kružnost filtrira beli tekst i efekte. Dijamant: živa cijan-plava boja,
jedina takva na noćnom travnjaku. Oboje se kupe istim potezom kao sunca;
"slepilo" posle klika sprečava jurenje dok lete ka kasi.
"""

import math

import cv2
import numpy as np

SILVER_LOWER = (0, 0, 175)
SILVER_UPPER = (179, 45, 255)
DIAMOND_LOWER = (88, 120, 150)
DIAMOND_UPPER = (112, 255, 255)


def find_pickups(frame_bgr, top_margin_px, bottom_margin_px,
                 area_min, area_max, offset=(0, 0)):
    """Vraća [(x, y, r)] centre novčića/dijamanata u px ekrana."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
    results = []

    for lower, upper, need_circular in (
            (SILVER_LOWER, SILVER_UPPER, True),      # novčić: mora biti okrugao
            (DIAMOND_LOWER, DIAMOND_UPPER, False)):  # dijamant: boja je dovoljna
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (area_min < area < area_max):
                continue
            if need_circular:
                perim = cv2.arcLength(cnt, True)
                if perim <= 0 or 4 * math.pi * area / (perim * perim) < 0.55:
                    continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            if cy < top_margin_px or cy > bottom_margin_px:
                continue
            results.append((cx + offset[0], cy + offset[1],
                            math.sqrt(area / math.pi)))
    return results
