# -*- coding: utf-8 -*-
"""
PvZ WHACK-A-ZOMBIE RL AGENT v2.0

Tok programa:
  1. Deterministička navigacija kroz meni (meni je UI bez neizvesnosti -
     RL tu nema šta da uči, pa je skriptirana navigacija pouzdanija).
  2. RL politika (trenirana u simulatoru, vidi train_rl.py) igra partiju
     humanizovanim potezima dok ne pobedi ili izgubi.
  3. Upis kompletne statistike (stats/game_stats.csv + JSON po partiji).
  4. Program se gasi.

Pre prvog pokretanja:
  - istrenirati politiku:  python train_rl.py
  - proveriti kalibraciju: python tools/calibrate_hud.py
"""

import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pyautogui

import config
from navigator import PvZNavigator


def main():
    print("=" * 49)
    print("   PvZ WHACK-A-ZOMBIE RL AGENT v2.0")
    print("=" * 49)

    pyautogui.FAILSAFE = True  # mišem u gornji levi ugao = hitno gašenje

    if not os.path.exists(config.POLICY_PATH):
        print(f"\n[GREŠKA] Nema istrenirane politike: {config.POLICY_PATH}")
        print("Prvo pokreni:  python train_rl.py")
        sys.exit(1)
    if not os.path.exists(config.YOLO_MODEL_PATH):
        print(f"\n[GREŠKA] Nema YOLO modela: {config.YOLO_MODEL_PATH}")
        sys.exit(1)

    print("\n[INIT] Učitavam RL politiku i YOLO model...")
    from bot.agent.policy import TrainedPolicy
    from bot.game.real_env import RealGameSession
    from bot.stats.logger import StatsLogger

    policy = TrainedPolicy(config.POLICY_PATH)
    session = RealGameSession(policy)  # učitava i YOLO (warmup pre menija)
    logger = StatsLogger()

    print("\n--> Otvori igru na glavnom meniju (Main Menu) i ne diraj miš.")
    print("--> Počinjem za 3 sekunde...")
    time.sleep(3)

    nav = PvZNavigator()
    if not nav.start_whack_a_zombie():
        print("\n[GREŠKA] Navigacija kroz meni nije uspela. "
              "Proveri da li je igra u fokusu na glavnom meniju.")
        sys.exit(1)

    print("[INIT] Igra pokrenuta - predajem kontrolu agentu.\n")
    time.sleep(1.0)

    stats = session.play()
    logger.log_game(stats)

    print("\n[KRAJ] Program se gasi.")


if __name__ == "__main__":
    main()
