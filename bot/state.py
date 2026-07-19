# -*- coding: utf-8 -*-
"""
Zajedničko kodiranje stanja: i simulator i prava igra svoje "viđenje sveta"
svode na isti Snapshot, koji se zatim pretvara u vektor opservacije za
neuronsku mrežu i masku validnih akcija.

Ovo je ključ sim-to-real transfera: politika trenirana u simulatoru vidi
identičan format ulaza kada igra pravu igru.
"""

from dataclasses import dataclass, field

import numpy as np

from config import ROWS, COLS
from bot.actions import (
    N_ACTIONS, N_CELLS, WHACK_BASE, BUSTER_BASE, MINE_BASE,
    ACTION_ICE, ACTION_COLLECT_SUN, ACTION_WAIT, rc_to_cell,
)

# Broj udaraca čekićem po tipu zombija
HITS_BY_KIND = {"zombie": 1, "conehead": 2, "buckethead": 3}

GRID_CHANNELS = 7
N_SCALARS = 8
OBS_DIM = ROWS * COLS * GRID_CHANNELS + N_SCALARS


@dataclass
class ZombieInfo:
    row: int
    col: int
    kind: str            # 'zombie' | 'conehead' | 'buckethead'
    hits_remaining: int  # koliko udaraca je još potrebno
    rising: bool = False


@dataclass
class Snapshot:
    """Apstraktno stanje sveta, nezavisno od toga da li dolazi iz simulatora
    ili iz YOLO/HSV percepcije prave igre."""
    zombies: list = field(default_factory=list)      # [ZombieInfo]
    graves: set = field(default_factory=set)         # {(row, col)}
    suns: list = field(default_factory=list)         # [(row, col)]
    mines: list = field(default_factory=list)        # [(row, col, armed)]
    sun_bank: float = 0.0
    card_ready: tuple = (False, False, False)        # (gravebuster, potato_mine, ice)
    mowers_left: int = ROWS
    time_progress: float = 0.0                       # 0..1 kroz trajanje nivoa
    freeze_remaining: float = 0.0                    # s preostalog zamrzavanja


_KIND_CHANNEL = {"zombie": 0, "conehead": 1, "buckethead": 2}


def encode_observation(s: Snapshot) -> np.ndarray:
    """Snapshot -> vektor float32 dimenzije OBS_DIM."""
    grid = np.zeros((GRID_CHANNELS, ROWS, COLS), dtype=np.float32)

    for z in s.zombies:
        if 0 <= z.row < ROWS and 0 <= z.col < COLS:
            ch = _KIND_CHANNEL[z.kind]
            grid[ch, z.row, z.col] = min(grid[ch, z.row, z.col] + 0.5, 1.0)
            grid[3, z.row, z.col] = max(grid[3, z.row, z.col], z.hits_remaining / 3.0)

    for (r, c) in s.graves:
        if 0 <= r < ROWS and 0 <= c < COLS:
            grid[4, r, c] = 1.0

    for (r, c) in s.suns:
        if 0 <= r < ROWS and 0 <= c < COLS:
            grid[5, r, c] = 1.0

    for (r, c, armed) in s.mines:
        if 0 <= r < ROWS and 0 <= c < COLS:
            grid[6, r, c] = 1.0 if armed else 0.5

    scalars = np.array([
        min(s.sun_bank / 300.0, 2.0),
        1.0 if s.card_ready[0] else 0.0,
        1.0 if s.card_ready[1] else 0.0,
        1.0 if s.card_ready[2] else 0.0,
        s.mowers_left / float(ROWS),
        min(s.time_progress, 1.0),
        min(len(s.zombies) / 15.0, 2.0),
        min(s.freeze_remaining / 10.0, 1.0),
    ], dtype=np.float32)

    return np.concatenate([grid.ravel(), scalars])


def build_action_mask(s: Snapshot) -> np.ndarray:
    """Vraća bool masku dimenzije N_ACTIONS - koje akcije imaju smisla sada.

    Napomena o priuštivosti biljaka: i simulator i vizuelni čitač HUD-a
    javljaju karticu kao 'ready' samo kada je napunjena I priuštiva, pa
    maska ne mora posebno da proverava stanje sunca.
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)

    occupied = set()
    for z in s.zombies:
        if 0 <= z.row < ROWS and 0 <= z.col < COLS:
            occupied.add((z.row, z.col))
            mask[WHACK_BASE + rc_to_cell(z.row, z.col)] = True

    if s.card_ready[0]:
        for (r, c) in s.graves:
            if 0 <= r < ROWS and 0 <= c < COLS:
                mask[BUSTER_BASE + rc_to_cell(r, c)] = True

    mine_cells = {(r, c) for (r, c, _) in s.mines}

    if s.card_ready[1]:
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) not in s.graves and (r, c) not in mine_cells:
                    mask[MINE_BASE + rc_to_cell(r, c)] = True

    if s.card_ready[2]:
        free_exists = any(
            (r, c) not in s.graves and (r, c) not in mine_cells
            for r in range(ROWS) for c in range(COLS)
        )
        if free_exists:
            mask[ACTION_ICE] = True

    if s.suns:
        mask[ACTION_COLLECT_SUN] = True

    mask[ACTION_WAIT] = True
    return mask


def find_free_cell(s: Snapshot, prefer_safe=True):
    """Bira slobodnu ćeliju za sadnju Ice-shroom-a (bez groba/mine; po
    mogućstvu desno i dalje od zombija, da sadnja ne smeta čekiću)."""
    mine_cells = {(r, c) for (r, c, _) in s.mines}
    candidates = [
        (r, c) for r in range(ROWS) for c in range(COLS)
        if (r, c) not in s.graves and (r, c) not in mine_cells
    ]
    if not candidates:
        return None
    if not prefer_safe:
        return candidates[0]
    zombie_cells = {(z.row, z.col) for z in s.zombies}
    def score(rc):
        r, c = rc
        near_zombie = 1 if rc in zombie_cells else 0
        return (near_zombie, -c)  # prvo bez zombija, pa što desnije
    return min(candidates, key=score)
