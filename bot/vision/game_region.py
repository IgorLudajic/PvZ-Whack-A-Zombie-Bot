# -*- coding: utf-8 -*-
"""
Automatska detekcija okvira igre unutar snimka ekrana.

PvZ je 4:3 igra: u fullscreen režimu na 16:9 monitoru renderuje se sa
crnim trakama levo i desno (pillarbox). Svi ROI-jevi u config.py su zato
frakcije DETEKTOVANE svetle oblasti (okvira igre), a ne celog monitora -
tako kalibracija radi na bilo kojoj rezoluciji i odnosu strana.
"""

import numpy as np

BRIGHTNESS_THRESHOLD = 16   # ispod ovoga kolona/red se smatra crnom trakom
MIN_REGION_FRACTION = 0.30  # detektovana oblast mora biti bar ovoliki deo ekrana


def detect_game_region(frame_bgr):
    """Vraća (x, y, w, h) svetle (ne-crne) oblasti frejma.

    Ako smislena oblast ne postoji (npr. igra nije na ekranu), vraća ceo
    frejm - ROI-jevi tada rade kao da nema crnih traka."""
    gray = frame_bgr.max(axis=2)
    h, w = gray.shape

    col_bright = np.percentile(gray, 95, axis=0)
    row_bright = np.percentile(gray, 95, axis=1)
    xs = np.flatnonzero(col_bright > BRIGHTNESS_THRESHOLD)
    ys = np.flatnonzero(row_bright > BRIGHTNESS_THRESHOLD)

    if len(xs) < w * MIN_REGION_FRACTION or len(ys) < h * MIN_REGION_FRACTION:
        return (0, 0, w, h)

    return (int(xs[0]), int(ys[0]),
            int(xs[-1] - xs[0] + 1), int(ys[-1] - ys[0] + 1))
