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

import argparse
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
    parser = argparse.ArgumentParser(description="PvZ Whack-a-Zombie RL agent")
    parser.add_argument("--games", type=int, default=1,
                        help="broj partija zaredom (za statistiku)")
    args = parser.parse_args()

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
    logger = StatsLogger()
    RealGameSession(policy)  # warmup YOLO pre menija (prva inferenca je spora)

    print("\n--> Otvori igru na glavnom meniju (Main Menu) i ne diraj miš.")
    print("--> Počinjem za 3 sekunde...")
    time.sleep(3)

    nav = PvZNavigator()
    for game_no in range(1, args.games + 1):
        if args.games > 1:
            print(f"\n{'=' * 49}\n   PARTIJA {game_no}/{args.games}\n{'=' * 49}")

        if game_no == 1:
            ok = nav.start_whack_a_zombie()
        else:
            # posle završene partije igra nudi dijaloge (Try Again / meni);
            # prvo probaj direktno kroz dijaloge, pa ceo put od menija
            ok = nav.prepare_game(timeout=25.0) or nav.start_whack_a_zombie()
        if not ok:
            print("\n[GREŠKA] Navigacija nije uspela - prekidam seriju.")
            break

        print("[INIT] Igra pokrenuta - predajem kontrolu agentu.\n")
        time.sleep(1.0)

        session = RealGameSession(policy)  # svež tracker/statistika po partiji
        stats = session.play()
        if stats.result == "INTERRUPTED":
            print("\n[KRAJ] Partija prekinuta ručno (Ctrl+C) - ništa se ne upisuje.")
            break
        logger.log_game(stats)

        if stats.result == "TIMEOUT":
            break
        if game_no < args.games:
            time.sleep(3.0)  # završni ekran/animacija pre sledeće navigacije

    print("\n[KRAJ] Program se gasi.")


if __name__ == "__main__":
    main()
