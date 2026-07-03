# -*- coding: utf-8 -*-
"""
Double DQN sa maskiranjem akcija i SMDP diskontovanjem.

Pošto akcije traju različito (čovekov pokret miša do daleke mete traje duže
od bliskog klika), tranzicije nose sopstveni diskontni faktor
gamma_dt = GAMMA_PER_SEC ** trajanje_akcije, umesto fiksne game po koraku.
"""

import numpy as np
import torch
import torch.nn as nn

from bot.state import OBS_DIM
from bot.actions import N_ACTIONS, decode
from bot.agent.network import QNetwork, masked_argmax

GAMMA_PER_SEC = 0.95
N_STEP = 3  # n-step returns: terminalne nagrade se prostiru n puta brže


class NStepAccumulator:
    """Skuplja tranzicije i emituje n-step povraćaje sa SMDP diskontom:
    G = r_t + γ_t r_{t+1} + γ_t γ_{t+1} r_{t+2} + ...  uz γ_acc = Π γ_i."""

    def __init__(self, buffer, n=N_STEP):
        self.buffer = buffer
        self.n = n
        self.pending = []  # [obs, action, G, gamma_acc]

    def push(self, obs, action, reward, next_obs, done, next_mask, gamma_dt):
        for item in self.pending:
            item[2] += item[3] * reward
            item[3] *= gamma_dt
        self.pending.append([obs, action, reward, gamma_dt])

        if done:
            for (o, a, g, g_acc) in self.pending:
                self.buffer.add(o, a, g, next_obs, True, next_mask, g_acc)
            self.pending.clear()
        elif len(self.pending) >= self.n:
            o, a, g, g_acc = self.pending.pop(0)
            self.buffer.add(o, a, g, next_obs, False, next_mask, g_acc)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.capacity = capacity
        self.size = 0
        self.idx = 0
        self.obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, N_ACTIONS), dtype=bool)
        self.gamma = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, next_mask, gamma_dt):
        i = self.idx
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.next_mask[i] = next_mask
        self.gamma[i] = gamma_dt
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.action[idx]),
            torch.from_numpy(self.reward[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.done[idx]),
            torch.from_numpy(self.next_mask[idx]),
            torch.from_numpy(self.gamma[idx]),
        )


class DQNAgent:
    """Double DQN sa trajnim demo baferom (DQfD-stil): demonstracije
    heuristike se nikad ne pregaze i čine fiksni deo svakog batch-a,
    što sprečava urušavanje politike kada agentovi sopstveni podaci
    istisnu demonstracije iz ring bafera."""

    def __init__(self, lr=2e-4, buffer_size=100_000, batch_size=128,
                 target_sync=2500, demo_frac=0.25, margin=0.1,
                 sup_weight=1.0, seed=0):
        self.margin = margin
        self.sup_weight = sup_weight
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.online = QNetwork()
        self.target = QNetwork()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_size)
        self.demo_buffer = ReplayBuffer(50_000)
        self.demo_frac = demo_frac
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.updates = 0

    def act(self, obs, mask, epsilon):
        if self.rng.random() < epsilon:
            return self._balanced_random(mask)
        with torch.no_grad():
            q = self.online(torch.from_numpy(obs).unsqueeze(0))
            a = masked_argmax(q, torch.from_numpy(mask).unsqueeze(0))
        return int(a.item())

    def _balanced_random(self, mask):
        """Nasumična akcija uniformna po TIPU akcije, pa tek onda po ćeliji.
        Bez ovoga 45 cherry ćelija dominira istraživanjem čim je višnja
        priuštiva, pa se sunce troši na nasumične sadnje."""
        valid = np.flatnonzero(mask)
        by_kind = {}
        for a in valid:
            by_kind.setdefault(decode(int(a))[0], []).append(int(a))
        kind = self.rng.choice(list(by_kind.keys()))
        return int(self.rng.choice(by_kind[kind]))

    def update(self):
        if self.buffer.size < self.batch_size:
            return None
        n_demo = int(self.batch_size * self.demo_frac) if self.demo_buffer.size > 0 else 0
        batch = self.buffer.sample(self.batch_size - n_demo, self.rng)
        if n_demo > 0:
            demo = self.demo_buffer.sample(n_demo, self.rng)
            batch = tuple(torch.cat([a, b]) for a, b in zip(batch, demo))
        obs, action, reward, next_obs, done, next_mask, gamma = batch

        with torch.no_grad():
            # Double DQN: online mreža bira akciju, target mreža je vrednuje
            next_q_online = self.online(next_obs)
            best_next = masked_argmax(next_q_online, next_mask)
            next_q_target = self.target(next_obs).gather(1, best_next.unsqueeze(1)).squeeze(1)
            y = reward + (1.0 - done) * gamma * next_q_target

        q_full = self.online(obs)
        q = q_full.gather(1, action.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q, y)

        # DQfD supervizovani margin gubitak na demo delu batch-a: demo akcija
        # mora biti bar za 'margin' bolja od svake druge - politika brzo
        # dostiže nivo demonstratora, a TD deo je dalje popravlja
        if n_demo > 0:
            q_demo = q_full[-n_demo:]
            a_demo = action[-n_demo:]
            margins = torch.full_like(q_demo, self.margin)
            margins.scatter_(1, a_demo.unsqueeze(1), 0.0)
            sup = (q_demo + margins).max(dim=1).values \
                - q_demo.gather(1, a_demo.unsqueeze(1)).squeeze(1)
            loss = loss + self.sup_weight * sup.clamp(min=0).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save({"model": self.online.state_dict(),
                    "obs_dim": OBS_DIM, "n_actions": N_ACTIONS}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.online.load_state_dict(ckpt["model"])
        self.target.load_state_dict(ckpt["model"])
