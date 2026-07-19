# -*- coding: utf-8 -*-
"""
Snimanje frejmova za sveže šablone navigacije.

Pokreni skriptu, odmah dovedi igru u prvi plan (glavni meni), pa POLAKO
prolazi kroz meni dok skripta snima ekran na svake 4 sekunde:

    Main Menu -> Mini-Games -> ikona "Whack a Zombie" -> popup New Game
    -> potvrda -> "Click to start!" -> pusti da partija krene (par sekundi)

Frejmovi se čuvaju u assets_capture/frame_XX.png; iz njih se seku novi
šabloni za assets/.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2

from bot.vision.capture import ScreenCapture

N_FRAMES = 16
INTERVAL = 4.0
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "assets_capture")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    screen = ScreenCapture()
    print(f"Snimam {N_FRAMES} frejmova na svake {INTERVAL:.0f} s "
          f"(ukupno {N_FRAMES * INTERVAL:.0f} s).")
    print("ODMAH klikni na igru i polako prolazi kroz meni!\n")
    for i in range(1, N_FRAMES + 1):
        time.sleep(INTERVAL)
        frame = screen.grab_bgr()
        path = os.path.join(OUT_DIR, f"frame_{i:02d}.png")
        cv2.imwrite(path, frame)
        print(f"  [{i:2d}/{N_FRAMES}] snimljen {os.path.basename(path)}")
    print(f"\nGotovo - frejmovi su u {OUT_DIR}")


if __name__ == "__main__":
    main()
