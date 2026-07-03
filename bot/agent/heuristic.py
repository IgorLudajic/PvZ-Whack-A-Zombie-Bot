# -*- coding: utf-8 -*-
"""
Skriptirane politike za poređenje i demonstracije.

HeuristicPolicy  - replika strategije v1.0 bota: uvek udari najlevljeg
                   zombija, sunce kupi kad je mirno, Grave Buster kad je
                   malo zombija; Cherry Bomb i Ice-shroom NE koristi.
                   Služi kao baseline u evaluaciji.

ProHeuristicPolicy - obogaćena verzija za generisanje demonstracija:
                   koristi i Cherry Bomb (na najgušći klaster) i
                   Ice-shroom (kad je gužva), i kupi sunce čim nema
                   akutne opasnosti. Pokazuje RL agentu sve tipove
                   akcija u smislenim situacijama.
"""

from config import ROWS, COLS
from bot.actions import (
    WHACK_BASE, BUSTER_BASE, CHERRY_BASE, ACTION_ICE,
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
    """Ekspertska skripta: prioritet akutna odbrana, pa AoE biljke u gužvi,
    pa ekonomija (sunce, grobovi)."""

    ICE_ZOMBIE_THRESHOLD = 10
    CHERRY_CLUSTER_MIN = 5

    def act_snapshot(self, s: Snapshot) -> int:
        zombies = s.zombies
        leftmost = min(zombies, key=lambda z: z.col) if zombies else None

        # 1) akutna opasnost - udari najlevljeg odmah
        if leftmost is not None and leftmost.col <= 2:
            return WHACK_BASE + rc_to_cell(leftmost.row, leftmost.col)

        # 2) velika gužva - zamrzni sve
        if s.card_ready[2] and len(zombies) >= self.ICE_ZOMBIE_THRESHOLD:
            return ACTION_ICE

        # 3) gust klaster - višnja na njegov centar
        if s.card_ready[1] and zombies:
            best_cell, best_count = None, 0
            for r in range(ROWS):
                for c in range(COLS):
                    if (r, c) in s.graves:
                        continue
                    count = sum(1 for z in zombies
                                if abs(z.row - r) <= 1 and abs(z.col - c) <= 1)
                    if count > best_count:
                        best_cell, best_count = (r, c), count
            if best_cell is not None and best_count >= self.CHERRY_CLUSTER_MIN:
                return CHERRY_BASE + rc_to_cell(best_cell[0], best_cell[1])

        # 4) nema akutne opasnosti - pokupi sunce (finansira biljke)
        if s.suns and (leftmost is None or leftmost.col >= 4):
            return ACTION_COLLECT_SUN

        # 5) standardno čišćenje
        if leftmost is not None:
            return WHACK_BASE + rc_to_cell(leftmost.row, leftmost.col)

        # 6) mir - skloni grob (najlevlji je najopasniji izvor)
        if s.card_ready[0] and s.graves:
            r, c = min(s.graves, key=lambda rc: rc[1])
            return BUSTER_BASE + rc_to_cell(r, c)

        return ACTION_WAIT
