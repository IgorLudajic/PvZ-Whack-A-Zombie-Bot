# -*- coding: utf-8 -*-
"""
Trening RL agenta (Double DQN) u simulatoru Whack-a-Zombie.

Pokretanje:
    python train_rl.py --steps 400000

Rezultati idu u runs/rl/<timestamp>/:
    episodes.csv  - statistika svake epizode (za grafike u radu)
    curve.png     - kriva učenja (nagrada + win rate)
    policy.pt     - checkpoint politike (najbolji i poslednji)
Finalna politika se kopira u models/policy.pt (koristi je main.py).
"""

import argparse
import csv
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import config
from bot.sim.simulator import WhackSimulator
from bot.state import encode_observation, build_action_mask
from bot.agent.dqn import DQNAgent, GAMMA_PER_SEC, NStepAccumulator
from bot.agent.heuristic import ProHeuristicPolicy
from bot.actions import ACTION_WAIT


def linear_epsilon(step, total, eps_start=0.4, eps_end=0.05, frac=0.25):
    decay_steps = total * frac
    if step >= decay_steps:
        return eps_end
    return eps_start + (eps_end - eps_start) * step / decay_steps


def greedy_eval(agent, episodes=10, seed=4242):
    """Periodična evaluacija BEZ istraživanja - meri stvarnu snagu politike.
    Po njoj se bira najbolji checkpoint (trening metrika je zagađena
    epsilon šumom i nestabilnošću TD bootstrappinga)."""
    sim = WhackSimulator(seed=seed)
    wins, mowers = 0, 0
    for _ in range(episodes):
        snap = sim.reset()
        done = False
        while not done:
            obs = encode_observation(snap)
            mask = build_action_mask(snap)
            action = agent.act(obs, mask, epsilon=0.0)
            snap, _, done, info = sim.step(action)
        st = info["stats"]
        wins += st.result == "WIN"
        mowers += st.mowers_used
    return 100.0 * wins / episodes, mowers / episodes


def seed_with_demonstrations(agent, sim, demo_steps):
    """Puni replay bafer tranzicijama heurističke politike (DQfD-lite).

    Demonstracije daju mreži primere razumnog ponašanja ("udaraj zombije
    blizu kuće") mnogo pre nego što bi ih epsilon-greedy istraživanje
    samo otkrilo u prostoru od 138 akcija."""
    print(f"[TRENING] Seedujem demo bafer sa {demo_steps} koraka pro-heuristike...")
    demo = ProHeuristicPolicy()
    acc = NStepAccumulator(agent.demo_buffer)  # trajni bafer - nikad se ne pregazi
    snap = sim.reset()
    obs = encode_observation(snap)
    mask = build_action_mask(snap)
    for _ in range(demo_steps):
        action = demo.act_snapshot(snap)
        if not mask[action]:
            action = ACTION_WAIT
        snap, reward, done, info = sim.step(action)
        obs2 = encode_observation(snap)
        mask2 = build_action_mask(snap)
        acc.push(obs, action, reward, obs2, done,
                 mask2, GAMMA_PER_SEC ** info["dt"])
        obs, mask = obs2, mask2
        if done:
            snap = sim.reset()
            obs = encode_observation(snap)
            mask = build_action_mask(snap)
    print("[TRENING] Seedovanje gotovo.")


def train(total_steps, seed, run_dir, demo_steps=25_000, init_from=None,
          lr=2e-4, eps_start=0.4, eps_end=0.05, eval_every=20_000,
          eval_episodes=12, demo_frac=0.25, sup_weight=1.0, deploy=True):
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.POLICY_PATH), exist_ok=True)

    sim = WhackSimulator(seed=seed)
    agent = DQNAgent(lr=lr, demo_frac=demo_frac, sup_weight=sup_weight, seed=seed)
    if init_from:
        agent.load(init_from)
        print(f"[TRENING] Nastavljam od checkpointa: {init_from}")

    if demo_steps > 0:
        seed_with_demonstrations(agent, sim, demo_steps)

    snap = sim.reset()
    obs = encode_observation(snap)
    mask = build_action_mask(snap)

    ep_reward = 0.0
    episode = 0
    recent_results = []
    best_score = -1e9
    warmup = 1000 if demo_steps > 0 else 3000
    t_start = time.time()

    ep_csv = open(os.path.join(run_dir, "episodes.csv"), "w", newline="", encoding="utf-8")
    ep_writer = csv.writer(ep_csv)
    ep_writer.writerow([
        "episode", "step", "result", "reward", "duration", "kills",
        "kills_zombie", "kills_cone", "kills_bucket", "clicks",
        "suns_collected", "gravebusters", "potato_mines", "ices", "mowers_used",
    ])

    acc = NStepAccumulator(agent.buffer)
    for step in range(1, total_steps + 1):
        eps = linear_epsilon(step, total_steps, eps_start, eps_end)
        action = agent.act(obs, mask, eps)
        snap2, reward, done, info = sim.step(action)
        obs2 = encode_observation(snap2)
        mask2 = build_action_mask(snap2)
        gamma_dt = GAMMA_PER_SEC ** info["dt"]
        acc.push(obs, action, reward, obs2, done, mask2, gamma_dt)
        ep_reward += reward
        obs, mask = obs2, mask2

        if step > warmup:
            agent.update()

        if step % 50_000 == 0:
            agent.save(os.path.join(run_dir, f"policy_step{step // 1000}k.pt"))

        # periodična greedy evaluacija - po njoj se bira najbolja politika
        if step % eval_every == 0 and step > warmup:
            gw, gm = greedy_eval(agent, episodes=eval_episodes)
            score = gw - 5.0 * gm  # kosačice obaraju ocenu (cilj: 0)
            marker = ""
            if score >= best_score:
                best_score = score
                agent.save(os.path.join(run_dir, "policy_best.pt"))
                marker = "  -> NAJBOLJA, sačuvana"
            print(f"[EVAL] korak {step:7d} | greedy win rate {gw:5.1f}% | "
                  f"kosačice/partiji {gm:.2f}{marker}", flush=True)

        if done:
            st = info["stats"]
            episode += 1
            recent_results.append(1 if st.result == "WIN" else 0)
            ep_writer.writerow([
                episode, step, st.result, f"{ep_reward:.2f}", f"{st.duration:.1f}",
                st.total_kills, st.kills["zombie"], st.kills["conehead"],
                st.kills["buckethead"], st.clicks, st.suns_collected,
                st.plants_used["gravebuster"], st.plants_used["potato_mine"],
                st.plants_used["ice"], st.mowers_used,
            ])
            ep_csv.flush()

            if episode % 10 == 0:
                wr = 100.0 * np.mean(recent_results[-50:])
                speed = step / (time.time() - t_start)
                print(f"[TRENING] ep {episode:4d} | korak {step:7d} | eps {eps:.2f} | "
                      f"win rate (50 ep) {wr:5.1f}% | nagrada {ep_reward:7.2f} | "
                      f"{speed:.0f} koraka/s", flush=True)

            snap = sim.reset()
            obs = encode_observation(snap)
            mask = build_action_mask(snap)
            ep_reward = 0.0

    ep_csv.close()
    agent.save(os.path.join(run_dir, "policy_last.pt"))

    # finalna politika = najbolja ako postoji, inače poslednja
    best_path = os.path.join(run_dir, "policy_best.pt")
    final_src = best_path if os.path.exists(best_path) else os.path.join(run_dir, "policy_last.pt")
    if deploy:
        shutil.copyfile(final_src, config.POLICY_PATH)
        print(f"\n[TRENING] Gotovo. Politika sačuvana u {config.POLICY_PATH}")
    else:
        print(f"\n[TRENING] Gotovo. Najbolja politika: {final_src} "
              f"(bez kopiranja u {config.POLICY_PATH} - koristi tools/select_policy.py)")

    plot_curves(run_dir)


def plot_curves(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_path = os.path.join(run_dir, "episodes.csv")
    rows = list(csv.DictReader(open(eps_path, encoding="utf-8")))
    if len(rows) < 5:
        return
    rewards = np.array([float(r["reward"]) for r in rows])
    wins = np.array([1.0 if r["result"] == "WIN" else 0.0 for r in rows])
    w = min(25, max(len(rows) // 10, 1))
    kernel = np.ones(w) / w
    smooth_r = np.convolve(rewards, kernel, mode="valid")
    smooth_w = np.convolve(wins, kernel, mode="valid") * 100

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(rewards, alpha=0.25, color="tab:blue")
    ax[0].plot(np.arange(w - 1, len(rows)), smooth_r, color="tab:blue")
    ax[0].set_title("Nagrada po epizodi")
    ax[0].set_xlabel("Epizoda"); ax[0].set_ylabel("Ukupna nagrada")
    ax[1].plot(np.arange(w - 1, len(rows)), smooth_w, color="tab:green")
    ax[1].set_title(f"Win rate (klizni prozor {w} ep.)")
    ax[1].set_xlabel("Epizoda"); ax[1].set_ylabel("%"); ax[1].set_ylim(0, 105)
    fig.tight_layout()
    out = os.path.join(run_dir, "curve.png")
    fig.savefig(out, dpi=130)
    print(f"[TRENING] Kriva učenja: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN trening u simulatoru")
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-steps", type=int, default=25_000,
                        help="koraka heuristike za seedovanje bafera (0 = bez)")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None,
                        help="checkpoint od kog se nastavlja (fino doterivanje)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eps-start", type=float, default=0.4)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--demo-frac", type=float, default=0.25,
                        help="udeo demonstracija u svakom batch-u")
    parser.add_argument("--sup-weight", type=float, default=1.0,
                        help="težina DQfD margin gubitka")
    parser.add_argument("--no-deploy", dest="deploy", action="store_false",
                        help="ne kopiraj najbolju politiku u models/policy.pt")
    args = parser.parse_args()

    run_dir = args.run_dir or os.path.join(
        config.RUNS_RL_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"[TRENING] Koraka: {args.steps} | demo: {args.demo_steps} | rezultati: {run_dir}")
    train(args.steps, args.seed, run_dir, args.demo_steps, args.init_from,
          args.lr, args.eps_start, args.eps_end, args.eval_every,
          args.eval_episodes, args.demo_frac, args.sup_weight, args.deploy)
