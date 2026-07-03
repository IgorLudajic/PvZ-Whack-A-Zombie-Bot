# -*- coding: utf-8 -*-
"""
Pretvaranje percepcije prave igre (trackovi, grobovi, sunca, HUD) u isti
Snapshot format koji proizvodi simulator - politika ne razlikuje izvore.
"""

from config import (
    ROWS, COLS, LAWN_LEFT_FRAC, LAWN_TOP_FRAC, LAWN_RIGHT_FRAC, LAWN_BOTTOM_FRAC,
)
from bot.state import Snapshot, ZombieInfo


class GridMapper:
    """Mapiranje piksela ekrana <-> ćelija travnjaka 5x9."""

    def __init__(self, screen):
        self.screen = screen
        self.left = LAWN_LEFT_FRAC * screen.width
        self.top = LAWN_TOP_FRAC * screen.height
        self.cell_w = (LAWN_RIGHT_FRAC - LAWN_LEFT_FRAC) * screen.width / COLS
        self.cell_h = (LAWN_BOTTOM_FRAC - LAWN_TOP_FRAC) * screen.height / ROWS

    def cell_of_px(self, x, y):
        col = int((x - self.left) / self.cell_w)
        row = int((y - self.top) / self.cell_h)
        return (max(0, min(ROWS - 1, row)), max(0, min(COLS - 1, col)))

    def col_float(self, x):
        """Kolona kao realan broj (za merenje bezbednosne margine)."""
        return (x - self.left) / self.cell_w - 0.5

    def cell_center_px(self, row, col):
        x = self.left + (col + 0.5) * self.cell_w
        y = self.top + (row + 0.5) * self.cell_h
        return (self.screen.left + x, self.screen.top + y)

    def zombie_cell(self, track):
        """Red po nogama (donji deo box-a), kolona po centru."""
        feet_y = track.cy + 0.32 * track.h
        return self.cell_of_px(track.cx, feet_y)


def build_snapshot(mapper, tracks, grave_dets, suns_px, card_ready,
                   mowers_left, time_progress, sun_bank, freeze_remaining, now):
    """Vraća (Snapshot, cell->track mapa za izvršni sloj)."""
    zombies = []
    cell_tracks = {}
    for tr in tracks:
        r, c = mapper.zombie_cell(tr)
        hits = tr.hits_remaining(now)
        if hits <= 0:
            continue
        zombies.append(ZombieInfo(row=r, col=c, kind=tr.kind, hits_remaining=hits))
        # u ćeliji pamtimo najlevlju (najopasniju) metu
        if (r, c) not in cell_tracks or tr.cx < cell_tracks[(r, c)].cx:
            cell_tracks[(r, c)] = tr

    graves = {mapper.cell_of_px(d.cx, d.cy + 0.2 * (d.y2 - d.y1)) for d in grave_dets}
    suns = [mapper.cell_of_px(x, y) for (x, y) in suns_px]

    snap = Snapshot(
        zombies=zombies,
        graves=graves,
        suns=suns,
        sun_bank=sun_bank,
        card_ready=card_ready,
        mowers_left=mowers_left,
        time_progress=time_progress,
        freeze_remaining=freeze_remaining,
    )
    return snap, cell_tracks
