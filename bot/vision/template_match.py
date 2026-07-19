# -*- coding: utf-8 -*-
"""
Multi-scale template matching.

Šabloni u assets/ su snimljeni na jednoj rezoluciji/razmeri, a igra na
drugoj mašini može biti renderovana krupnije ili sitnije - matchTemplate
nije otporan na skaliranje. Zato se šablon traži na više razmera; kada se
prava razmera jednom pronađe, čuva se u calibration.json i dalje se koristi
direktno (i za detekciju pobede/poraza tokom partije).
"""

import json
import os

import cv2

import config

CALIBRATION_FILE = os.path.join(config.ROOT_DIR, "calibration.json")

DEFAULT_SCALES = [0.50, 0.60, 0.667, 0.75, 0.833, 0.90, 1.00,
                  1.10, 1.20, 1.333, 1.40, 1.50, 1.60, 1.75, 2.00]


def find_template(frame, template, threshold, scale=None):
    """Traži šablon u frejmu; ako je scale None, pretražuje sve razmere.

    Vraća (found, cx, cy, best_scale, best_val)."""
    candidates = [scale] if scale is not None else DEFAULT_SCALES
    best_val, best_cx, best_cy, best_s = 0.0, None, None, None

    for s in candidates:
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        t = template if abs(s - 1.0) < 1e-6 else cv2.resize(
            template, None, fx=s, fy=s, interpolation=interp)
        th, tw = t.shape[:2]
        if th >= frame.shape[0] or tw >= frame.shape[1]:
            continue
        res = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_cx = max_loc[0] + tw // 2
            best_cy = max_loc[1] + th // 2
            best_s = s

    return best_val >= threshold, best_cx, best_cy, best_s, best_val


def save_template_scale(scale):
    data = {}
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data["template_scale"] = scale
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_template_scale(default=1.0):
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, encoding="utf-8") as f:
                return float(json.load(f).get("template_scale", default))
        except Exception:
            pass
    return default


def scale_template(template, scale):
    if template is None or abs(scale - 1.0) < 1e-6:
        return template
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(template, None, fx=scale, fy=scale, interpolation=interp)
