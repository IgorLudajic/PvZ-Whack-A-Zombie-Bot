# -*- coding: utf-8 -*-
"""
Zajednički diskretni prostor akcija za simulator i pravu igru.

Politika bira VISOKONIVOVSKU akciju (koju metu / koju biljku / koje polje),
a izvršni sloj (sim ili executor) rešava motoriku: putanju miša i tačan
broj klikova po tipu zombija (1 običan, 2 čunj, 3 kanta).
"""

from config import ROWS, COLS

N_CELLS = ROWS * COLS  # 45

# Raspored ID-jeva akcija:
#   [0, 45)    WHACK  - udari zombija u ćeliji (potreban broj udaraca)
#   [45, 90)   BUSTER - posadi Grave Buster na grob u ćeliji
#   [90, 135)  MINE   - posadi Potato Mine na ćeliju (aktivira se ~15 s,
#                       eksplodira kad zombi nagazi - "rezervna kosačica")
#   135        ICE    - posadi Ice-shroom (zamrzava sve zombije)
#   136        COLLECT_SUN - pokupi najbliže sunce
#   137        WAIT   - kratka pauza (nema validnih/poželjnih akcija)
WHACK_BASE = 0
BUSTER_BASE = N_CELLS
MINE_BASE = 2 * N_CELLS
ACTION_ICE = 3 * N_CELLS
ACTION_COLLECT_SUN = 3 * N_CELLS + 1
ACTION_WAIT = 3 * N_CELLS + 2
N_ACTIONS = 3 * N_CELLS + 3  # 138


def cell_to_rc(cell):
    return cell // COLS, cell % COLS


def rc_to_cell(row, col):
    return row * COLS + col


def decode(action_id):
    """Vraća (vrsta_akcije, red, kolona); za globalne akcije red/kolona su None."""
    if action_id < BUSTER_BASE:
        r, c = cell_to_rc(action_id - WHACK_BASE)
        return "whack", r, c
    if action_id < MINE_BASE:
        r, c = cell_to_rc(action_id - BUSTER_BASE)
        return "buster", r, c
    if action_id < ACTION_ICE:
        r, c = cell_to_rc(action_id - MINE_BASE)
        return "mine", r, c
    if action_id == ACTION_ICE:
        return "ice", None, None
    if action_id == ACTION_COLLECT_SUN:
        return "collect_sun", None, None
    return "wait", None, None


def describe(action_id):
    kind, r, c = decode(action_id)
    if r is None:
        return kind
    return f"{kind}({r},{c})"
