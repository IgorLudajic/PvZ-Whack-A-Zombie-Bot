# -*- coding: utf-8 -*-
"""
Finalna selekcija politike na NEZAVISNOM test setu.

Tokom treninga se najbolji checkpoint bira po kratkoj greedy evaluaciji
(24 epizode, fiksni seed) - dovoljno za vođenje treninga, ali podložno
overfittingu na taj uski set. Ova skripta presuđuje pošteno: svaki
kandidat se evaluira na velikom broju epizoda sa test-seedom koji NIJE
korišćen u treningu, pa se bira politika sa najboljom kombinacijom win
rate-a i (odsustva) kosačica. Pobednik se kopira u models/policy.pt.

    python tools/select_policy.py --episodes 150 --seed 555
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import config
from bot.sim.simulator import WhackSimulator
from bot.state import encode_observation, build_action_mask
from bot.agent.policy import TrainedPolicy

# Kandidati moraju biti trenirani na AKTUELNOJ verziji stanja (OBS_DIM);
# stariji checkpointi (runs/rl/v7-v10, stara bc) su nekompatibilni jer je
# mehanika Potato Mine promenila kodiranje stanja.
CANDIDATES = [
    "runs/rl/bc_mine/policy_best.pt",
    "runs/rl/dqn_mine/policy_best.pt",
]


def evaluate(path, episodes, seed):
    policy = TrainedPolicy(path)
    sim = WhackSimulator(seed=seed)  # isti seed => isti nizovi epizoda za sve
    wins, mowers, danger = 0, 0, 0
    for _ in range(episodes):
        snap = sim.reset()
        done = False
        while not done:
            obs = encode_observation(snap)
            mask = build_action_mask(snap)
            snap, _, done, info = sim.step(policy.act(obs, mask))
        st = info["stats"]
        wins += st.result == "WIN"
        mowers += st.mowers_used
        danger += st.danger_near_count
    n = episodes
    return {
        "win_rate": 100.0 * wins / n,
        "mowers": mowers / n,
        "danger": danger / n,
        "score": 100.0 * wins / n - 5.0 * mowers / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--candidates", nargs="*", default=CANDIDATES)
    parser.add_argument("--apply", action="store_true", default=True,
                        help="kopiraj pobednika u models/policy.pt (podrazumevano)")
    parser.add_argument("--no-apply", dest="apply", action="store_false")
    args = parser.parse_args()

    print(f"Test set: {args.episodes} epizoda, seed {args.seed} (neviđen u treningu)\n")
    print(f"{'kandidat':<32} {'win%':>7} {'kosač.':>8} {'opasnost':>9} {'score':>8}")
    print("-" * 70)

    results = []
    for path in args.candidates:
        full = os.path.join(config.ROOT_DIR, path)
        if not os.path.exists(full):
            print(f"{path:<32}  (nema fajla, preskačem)")
            continue
        r = evaluate(full, args.episodes, args.seed)
        results.append((path, r))
        print(f"{path:<32} {r['win_rate']:6.1f}% {r['mowers']:8.2f} "
              f"{r['danger']:9.2f} {r['score']:8.1f}", flush=True)

    if not results:
        print("Nema validnih kandidata.")
        return

    best_path, best = max(results, key=lambda kv: kv[1]["score"])
    print("-" * 70)
    print(f"\nPOBEDNIK: {best_path}")
    print(f"  win rate {best['win_rate']:.1f}% | kosačice {best['mowers']:.2f}/partiji "
          f"| score {best['score']:.1f}")

    if args.apply:
        os.makedirs(os.path.dirname(config.POLICY_PATH), exist_ok=True)
        shutil.copyfile(os.path.join(config.ROOT_DIR, best_path), config.POLICY_PATH)
        print(f"\nKopirano u {config.POLICY_PATH}")


if __name__ == "__main__":
    main()
