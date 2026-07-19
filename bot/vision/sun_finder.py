# -*- coding: utf-8 -*-
"""HSV detekcija sunca (žute konture odgovarajuće površine).

Vraća i poluprečnik blob-a: zombiji ispuštaju sunca u TROJKAMA čiji se
sjaj stapa u jedan veliki blob - izvršni sloj tada prelazi klikovima
preko cele grupe umesto jednog klika u centar."""

import math

import cv2
import numpy as np

import config

_LOWER = np.array(config.SUN_HSV_LOWER)
_UPPER = np.array(config.SUN_HSV_UPPER)


def find_suns(frame_bgr, top_margin_px=100, bottom_margin_px=None,
              area_min=None, area_max=None, offset=(0, 0)):
    """Vraća listu (x, y, r) - centar i poluprečnik blob-a u px ekrana.

    area_min/area_max: opseg površine konture u px^2, skaliran na rezoluciju
    (pozivalac ih računa iz visine okvira igre; podrazumevano nativa x1).
    offset se dodaje koordinatama (kada je frame isečen okvir igre).
    Gornja margina isključuje HUD (brojač sunca je takođe žut); donja
    isključuje traku ispod travnjaka (žuti natpis "Whack a Zombie")."""
    if area_min is None:
        area_min = config.SUN_AREA_MIN_NATIVE
    if area_max is None:
        area_max = config.SUN_AREA_MAX_NATIVE

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _LOWER, _UPPER)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    suns = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area_min < area < area_max:
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            if cy < top_margin_px:
                continue
            if bottom_margin_px is not None and cy > bottom_margin_px:
                continue
            r = math.sqrt(area / math.pi)
            suns.append((cx + offset[0], cy + offset[1], r))
    return suns
