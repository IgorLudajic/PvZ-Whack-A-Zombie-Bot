# -*- coding: utf-8 -*-
"""
Alat za proveru kalibracije: uslika ekran i nacrta preko njega sve ROI-jeve
iz config.py (mrežu travnjaka 5x9, slotove karata, trake kosačica, brojač
sunca). Pokrenuti dok je Whack-a-Zombie partija na ekranu, pa pogledati
calibration_preview.png - linije treba da se poklapaju sa elementima igre.

    python tools/calibrate_hud.py

Ako se ne poklapaju, korigovati frakcije u config.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

import config
from config import ROWS, COLS
from bot.vision.capture import ScreenCapture


def main():
    screen = ScreenCapture()
    screen.detect_game_region()
    frame = screen.grab_bgr()

    # okvir igre (detektovana svetla oblast između crnih traka)
    gx, gy, gw, gh = screen.gx, screen.gy, screen.gw, screen.gh
    cv2.rectangle(frame, (gx, gy), (gx + gw - 1, gy + gh - 1), (255, 0, 255), 3)
    cv2.putText(frame, f"OKVIR IGRE {gw}x{gh}", (gx + 10, gy + gh - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # mreža travnjaka
    left, top = gx + config.LAWN_LEFT_FRAC * gw, gy + config.LAWN_TOP_FRAC * gh
    right = gx + config.LAWN_RIGHT_FRAC * gw
    bottom = gy + config.LAWN_BOTTOM_FRAC * gh
    cell_w, cell_h = (right - left) / COLS, (bottom - top) / ROWS
    for i in range(COLS + 1):
        x = int(left + i * cell_w)
        cv2.line(frame, (x, int(top)), (x, int(bottom)), (0, 255, 0), 2)
    for j in range(ROWS + 1):
        y = int(top + j * cell_h)
        cv2.line(frame, (int(left), y), (int(right), y), (0, 255, 0), 2)

    # slotovi karata
    for i, frac in enumerate(config.CARD_SLOTS_FRAC):
        x, y, cw, ch = screen.frac_roi(frac)
        cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 0, 255), 3)
        cv2.putText(frame, config.CARD_ORDER[i], (x, y + ch + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # trake kosačica
    from bot.vision.mower_watcher import MowerWatcher
    for (x, y, mw, mh) in MowerWatcher(screen).rois:
        cv2.rectangle(frame, (x, y), (x + mw, y + mh), (255, 0, 0), 2)

    # brojač sunca
    x, y, cw, ch = screen.frac_roi(config.SUN_COUNTER_FRAC)
    cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 255, 255), 2)

    out = os.path.join(config.ROOT_DIR, "calibration_preview.png")
    cv2.imwrite(out, frame)
    print(f"Sačuvano: {out}")
    print("Zelena mreža = travnjak 5x9, crveno = karte, plavo = kosačice, "
          "žuto = brojač sunca.")
    print("Ako se ne poklapa sa igrom, podesi frakcije u config.py.")


if __name__ == "__main__":
    main()
