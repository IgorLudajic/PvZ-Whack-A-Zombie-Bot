# -*- coding: utf-8 -*-
"""
Behavior Cloning (čista imitacija) pro-heuristike.

Motivacija: pro-heuristika postiže ~100% u simulatoru, ali je ručno pravilo.
DQN uči iz nagrade i pati od nestabilnosti TD bootstrappinga (win rate
osciluje i degradira posle vrhunca). Čisto supervizovano kloniranje
demonstratora nema taj churn - mreža samo uči da preslika stanje u
ekspertovu akciju (maskirani cross-entropy). Daje gornju granicu kvaliteta
imitacije i, pošto koristi istu CNN arhitekturu i format checkpointa kao
DQN, direktno je uporedivo i učitljivo u main.py.

    python train_bc.py --samples 200000 --epochs 12

Najbolji model (po greedy evaluaciji u simulatoru) se snima u
runs/rl/bc/policy_best.pt; finalni izbor i dalje presuđuje tools/select_policy.py.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import config
from bot.sim.simulator import WhackSimulator
from bot.state import encode_observation, build_action_mask, OBS_DIM
from bot.actions import N_ACTIONS, ACTION_WAIT
from bot.agent.network import QNetwork, masked_argmax, NEG_INF
from bot.agent.heuristic import ProHeuristicPolicy


def collect_dataset(n_samples, seed):
    """Skuplja (obs, mask, expert_action) igrajući pro-heuristiku."""
    print(f"[BC] Skupljam {n_samples} demonstracija pro-heuristike...")
    expert = ProHeuristicPolicy()
    sim = WhackSimulator(seed=seed)
    obs_buf = np.zeros((n_samples, OBS_DIM), dtype=np.float32)
    mask_buf = np.zeros((n_samples, N_ACTIONS), dtype=bool)
    act_buf = np.zeros(n_samples, dtype=np.int64)

    snap = sim.reset()
    for i in range(n_samples):
        a = expert.act_snapshot(snap)
        mask = build_action_mask(snap)
        if not mask[a]:
            a = ACTION_WAIT
        obs_buf[i] = encode_observation(snap)
        mask_buf[i] = mask
        act_buf[i] = a
        snap, _, done, _ = sim.step(a)
        if done:
            snap = sim.reset()
    print("[BC] Dataset spreman.")
    return obs_buf, mask_buf, act_buf


def greedy_eval(net, episodes=30, seed=4242):
    sim = WhackSimulator(seed=seed)
    wins, mowers = 0, 0
    for _ in range(episodes):
        snap = sim.reset()
        done = False
        while not done:
            obs = torch.from_numpy(encode_observation(snap)).unsqueeze(0)
            mask = torch.from_numpy(build_action_mask(snap)).unsqueeze(0)
            with torch.no_grad():
                a = int(masked_argmax(net(obs), mask).item())
            snap, _, done, info = sim.step(a)
        wins += info["stats"].result == "WIN"
        mowers += info["stats"].mowers_used
    return 100.0 * wins / episodes, mowers / episodes


def train(n_samples, epochs, batch_size, lr, seed, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.POLICY_PATH), exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    obs, mask, act = collect_dataset(n_samples, seed)
    obs_t = torch.from_numpy(obs)
    mask_t = torch.from_numpy(mask)
    act_t = torch.from_numpy(act)

    net = QNetwork()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    best_score = -1e9
    n = n_samples
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = net(obs_t[idx])
            # nevalidne akcije van konkurencije
            logits = logits.masked_fill(~mask_t[idx], NEG_INF)
            loss = ce(logits, act_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        gw, gm = greedy_eval(net)
        score = gw - 5.0 * gm
        marker = ""
        if score >= best_score:
            best_score = score
            torch.save({"model": net.state_dict(), "obs_dim": OBS_DIM,
                        "n_actions": N_ACTIONS}, os.path.join(run_dir, "policy_best.pt"))
            marker = "  -> NAJBOLJA, sačuvana"
        print(f"[BC] epoha {epoch:2d}/{epochs} | gubitak {total_loss / n_batches:.4f} | "
              f"greedy win rate {gw:5.1f}% | kosačice/partiji {gm:.2f}{marker}", flush=True)

    torch.save({"model": net.state_dict(), "obs_dim": OBS_DIM,
                "n_actions": N_ACTIONS}, os.path.join(run_dir, "policy_last.pt"))
    print(f"\n[BC] Gotovo. Najbolji model: {os.path.join(run_dir, 'policy_best.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavior cloning pro-heuristike")
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=str, default=os.path.join(config.RUNS_RL_DIR, "bc"))
    args = parser.parse_args()
    print(f"[BC] uzoraka: {args.samples} | epoha: {args.epochs} | rezultati: {args.run_dir}")
    train(args.samples, args.epochs, args.batch_size, args.lr, args.seed, args.run_dir)
