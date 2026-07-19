# -*- coding: utf-8 -*-
"""
Skriptirane politike za poređenje i demonstracije.

HeuristicPolicy  - replika strategije v1.0 bota: uvek udari najlevljeg
                   zombija, sunce kupi kad je mirno, Grave Buster kad je
                   malo zombija; Potato Mine i Ice-shroom NE koristi.
                   Služi kao baseline u evaluaciji.

ProHeuristicPolicy - obogaćena verzija za generisanje demonstracija:
                   koristi i Potato Mine (odbrambena linija u levim
                   kolonama - "rezervna kosačica") i Ice-shroom (kad je
                   gužva), i kupi sunce čim nema akutne opasnosti.
                   Pokazuje RL agentu sve tipove akcija u smislenim
                   situacijama.
"""

from config import ROWS, COLS
from bot.actions import (
    WHACK_BASE, BUSTER_BASE, MINE_BASE, ACTION_ICE,
    ACTION_COLLECT_SUN, ACTION_WAIT, rc_to_cell,
)
from bot.state import Snapshot


class HeuristicPolicy:
    def act_snapshot(self, s: Snapshot) -> int:
        if s.zombies:
            target = min(s.zombies, key=lambda z: z.col)
            return WHACK_BASE + rc_to_cell(target.row, target.col)
        if s.suns:
            return ACTION_COLLECT_SUN
        if s.card_ready[0] and s.graves and len(s.zombies) <= 2:
            r, c = min(s.graves, key=lambda rc: rc[1])
            return BUSTER_BASE + rc_to_cell(r, c)
        return ACTION_WAIT


class ProHeuristicPolicy:
    """Ekspertska skripta: prioritet akutna odbrana, pa led u gužvi, pa
    minska odbrambena linija u levim kolonama, pa ekonomija (sunce, grobovi)."""

    # led tek kad je navala OZBILJNA - prerano trošenje ostavlja agenta
    # golog pred još većim talasom koji sledi (finalni talasi su najveći)
    ICE_ZOMBIE_THRESHOLD = 24
    MINE_COL = 1  # kolona odbrambene linije

    def act_snapshot(self, s: Snapshot) -> int:
        zombies = s.zombies
        leftmost = min(zombies, key=lambda z: z.col) if zombies else None

        # 1) akutna opasnost - udari najlevljeg odmah
        if leftmost is not None and leftmost.col <= 2:
            return WHACK_BASE + rc_to_cell(leftmost.row, leftmost.col)

        # 2) velika gužva - zamrzni sve
        if s.card_ready[2] and len(zombies) >= self.ICE_ZOMBIE_THRESHOLD:
            return ACTION_ICE

        # 3) grobovi su izvor svih zombija - Grave Buster na grob NAJBLIŽI
        #    KUĆI čim nema akutne opasnosti (manje grobova = manja navala)
        if s.card_ready[0] and s.graves and (leftmost is None or leftmost.col >= 5):
            r, c = min(s.graves, key=lambda rc: rc[1])
            return BUSTER_BASE + rc_to_cell(r, c)

        # 4) minska linija u red sa NAJVIŠE grobova (najveći priliv zombija)
        if s.card_ready[1] and (leftmost is None or leftmost.col >= 5):
            cell = self._mine_spot(s)
            if cell is not None:
                return MINE_BASE + rc_to_cell(cell[0], cell[1])

        # 5) pokupi sunce čim nema akutne opasnosti (finansira biljke)
        if s.suns and (leftmost is None or leftmost.col >= 3):
            return ACTION_COLLECT_SUN

        # 6) standardno čišćenje
        if leftmost is not None:
            return WHACK_BASE + rc_to_cell(leftmost.row, leftmost.col)

        return ACTION_WAIT

    def _mine_spot(self, s: Snapshot):
        """Bira (red, kolona) za minu: red sa najviše grobova (najveći
        priliv zombija), kolona 1-2 (uloga rezervne kosačice)."""
        mine_rows = {r for (r, _, _) in s.mines}
        graves_per_row = {}
        for (r, _c) in s.graves:
            graves_per_row[r] = graves_per_row.get(r, 0) + 1
        candidate_rows = sorted(
            (r for r in range(ROWS)
             if r not in mine_rows and graves_per_row.get(r, 0) > 0),
            key=lambda r: -graves_per_row[r],
        )
        mine_cells = {(r, c) for (r, c, _) in s.mines}
        for r in candidate_rows:
            for c in (self.MINE_COL, self.MINE_COL + 1):
                if (r, c) not in s.graves and (r, c) not in mine_cells:
                    return (r, c)
        return None
