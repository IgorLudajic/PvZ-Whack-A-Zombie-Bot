# -*- coding: utf-8 -*-
"""Politika za izvršavanje (inference): učitava istreniranu Q-mrežu i
bira najbolju validnu akciju za dato stanje."""

import numpy as np
import torch

from bot.agent.network import QNetwork, masked_argmax


class TrainedPolicy:
    def __init__(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.net = QNetwork(obs_dim=ckpt["obs_dim"], n_actions=ckpt["n_actions"])
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        q = self.net(torch.from_numpy(obs).unsqueeze(0))
        a = masked_argmax(q, torch.from_numpy(mask).unsqueeze(0))
        return int(a.item())
