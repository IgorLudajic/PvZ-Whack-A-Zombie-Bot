# -*- coding: utf-8 -*-
"""
Evaluacija politike u simulatoru: win rate, korišćenje kosačica, biljaka itd.

    python evaluate_policy.py --episodes 200                 # RL politika
    python evaluate_policy.py --episodes 200 --baseline v1   # heuristika v1.0 (bez biljaka)
    python evaluate_policy.py --episodes 200 --baseline pro  # pro-heuristika (demonstrator)

Poređenje ovih izlaza je direktan materijal za tabelu u diplomskom radu.
"""

import argparse
import csv
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import config
from bot.sim.simulator import WhackSimulator
from bot.state import encode_observation, build_action_mask
from bot.agent.heuristic import HeuristicPolicy, ProHeuristicPolicy
from bot.actions import ACTION_WAIT


def run_eval(episodes, baseline, policy_path, seed, out_csv):
    if baseline == "v1":
        policy = HeuristicPolicy()
        name = "HEURISTIKA (v1.0, bez biljaka)"
    elif baseline == "pro":
        policy = ProHeuristicPolicy()
        name = "PRO-HEURISTIKA (demonstrator)"
    else:
        from bot.agent.policy import TrainedPolicy
        policy = TrainedPolicy(policy_path)
        name = f"RL POLITIKA ({os.path.basename(policy_path)})"
    use_baseline = baseline is not None

    sim = WhackSimulator(seed=seed)
    all_stats = []

    for ep in range(episodes):
        snap = sim.reset()
        done = False
        while not done:
            if use_baseline:
                action = policy.act_snapshot(snap)
                mask = build_action_mask(snap)
                if not mask[action]:
                    action = ACTION_WAIT
            else:
                obs = encode_observation(snap)
                mask = build_action_mask(snap)
                action = policy.act(obs, mask)
            snap, _, done, info = sim.step(action)
        all_stats.append(info["stats"])
        if (ep + 1) % 25 == 0:
            wr = 100 * np.mean([s.result == "WIN" for s in all_stats])
            print(f"  ... {ep + 1}/{episodes} epizoda, trenutni win rate {wr:.1f}%")

    summarize(name, all_stats, out_csv)


def summarize(name, stats, out_csv):
    n = len(stats)
    wins = sum(s.result == "WIN" for s in stats)
    mowers = [s.mowers_used for s in stats]
    no_mower_games = sum(m == 0 for m in mowers)

    print("\n" + "=" * 60)
    print(f"  EVALUACIJA: {name}  ({n} epizoda)")
    print("=" * 60)
    print(f"  Win rate:                {100 * wins / n:6.1f}%  ({wins}/{n})")
    print(f"  Partije bez kosačica:    {100 * no_mower_games / n:6.1f}%")
    print(f"  Kosačice po partiji:     {np.mean(mowers):6.2f}")
    print(f"  Najbliži prilaz kući:    {np.mean([s.closest_approach for s in stats]):6.2f} kolona (prosek)")
    print(f"  Zombija u kritičnoj zoni:{np.mean([s.danger_near_count for s in stats]):6.2f} po partiji")
    print(f"  Ubistva po partiji:      {np.mean([s.total_kills for s in stats]):6.1f}")
    print(f"  Klikova po partiji:      {np.mean([s.clicks for s in stats]):6.1f}")
    print(f"  Sakupljeno sunca:        {np.mean([s.suns_collected for s in stats]):6.1f}")
    print(f"  Grave Buster po partiji: {np.mean([s.plants_used['gravebuster'] for s in stats]):6.2f}")
    print(f"  Cherry Bomb po partiji:  {np.mean([s.plants_used['cherry'] for s in stats]):6.2f}")
    print(f"  Ice-shroom po partiji:   {np.mean([s.plants_used['ice'] for s in stats]):6.2f}")
    print("=" * 60)

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["episode", "result", "duration", "kills", "clicks",
                        "suns", "gravebusters", "cherries", "ices", "mowers",
                        "closest_approach", "danger_near"])
            for i, s in enumerate(stats):
                w.writerow([i + 1, s.result, f"{s.duration:.1f}", s.total_kills,
                            s.clicks, s.suns_collected, s.plants_used["gravebuster"],
                            s.plants_used["cherry"], s.plants_used["ice"], s.mowers_used,
                            f"{s.closest_approach:.2f}", s.danger_near_count])
        print(f"  Detalji: {out_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--baseline", choices=["v1", "pro"], default=None,
                        help="evaluiraj skriptiranu politiku umesto RL (v1 ili pro)")
    parser.add_argument("--policy", type=str, default=config.POLICY_PATH)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    run_eval(args.episodes, args.baseline, args.policy, args.seed, args.out)
