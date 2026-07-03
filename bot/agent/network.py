# -*- coding: utf-8 -*-
"""
Q-mreža sa konvolucionim jezgrom nad mrežom travnjaka 5x9.

Ključna ideja: akcije vezane za ćelije (udari/buster/višnja) dele težine
preko prostora - pravilo "zombi u ćeliji -> udari tu ćeliju" uči se JEDNOM
(1x1 konvolucione glave), umesto 45 puta posebno kao kod MLP-a. To daje
mnogo verniju imitaciju demonstracija i bolju generalizaciju.

Globalne akcije (Ice-shroom, pokupi sunce, čekaj) dobijaju glavu nad
agregiranim featurima cele table + skalarni deo stanja.
"""

import numpy as np
import torch
import torch.nn as nn

from config import ROWS, COLS
from bot.state import OBS_DIM, GRID_CHANNELS, N_SCALARS
from bot.actions import N_ACTIONS, N_CELLS

NEG_INF = -1e9


class QNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=64):
        super().__init__()
        assert obs_dim == OBS_DIM and n_actions == N_ACTIONS
        in_ch = GRID_CHANNELS + N_SCALARS  # skalari se šire po svim ćelijama
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
        )
        # po ćeliji: kanal 0 = whack, 1 = buster, 2 = cherry
        self.cell_head = nn.Conv2d(hidden, 3, 1)
        # globalno: [ice, collect_sun, wait]
        self.global_head = nn.Sequential(
            nn.Linear(hidden + N_SCALARS, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, obs):
        b = obs.shape[0]
        grid = obs[:, :GRID_CHANNELS * N_CELLS].view(b, GRID_CHANNELS, ROWS, COLS)
        scalars = obs[:, GRID_CHANNELS * N_CELLS:]
        scalar_planes = scalars.view(b, N_SCALARS, 1, 1).expand(b, N_SCALARS, ROWS, COLS)
        x = self.conv(torch.cat([grid, scalar_planes], dim=1))

        # (B, 3, R, C) -> (B, 3*45) u redosledu [whack 0..44, buster 45..89, cherry 90..134]
        q_cells = self.cell_head(x).flatten(start_dim=2).flatten(start_dim=1)

        pooled = x.mean(dim=(2, 3))
        q_global = self.global_head(torch.cat([pooled, scalars], dim=1))

        return torch.cat([q_cells, q_global], dim=1)


def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Argmax preko Q-vrednosti uz ignorisanje nevalidnih akcija.
    q_values: (B, A) float, mask: (B, A) bool."""
    q = q_values.masked_fill(~mask, NEG_INF)
    return q.argmax(dim=1)


def random_valid_action(mask: np.ndarray, rng) -> int:
    valid = np.flatnonzero(mask)
    return int(rng.choice(valid))
