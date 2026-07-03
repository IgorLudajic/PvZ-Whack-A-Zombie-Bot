import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from bot.state import Snapshot, ZombieInfo, encode_observation, build_action_mask, OBS_DIM
from bot.actions import decode, N_ACTIONS, WHACK_BASE, BUSTER_BASE, CHERRY_BASE, ACTION_ICE, ACTION_COLLECT_SUN, ACTION_WAIT, rc_to_cell

# rucno konstruisan snapshot
s = Snapshot(
    zombies=[ZombieInfo(2, 4, "buckethead", 3), ZombieInfo(0, 1, "zombie", 1)],
    graves={(1, 3), (4, 8)},
    suns=[(2, 2)],
    sun_bank=200.0,
    card_ready=(True, True, False),
    mowers_left=5,
    time_progress=0.5,
    freeze_remaining=0.0,
)
obs = encode_observation(s)
mask = build_action_mask(s)
assert obs.shape == (OBS_DIM,)

# whack maska tacno na 2 celije
whacks = [i for i in range(WHACK_BASE, BUSTER_BASE) if mask[i]]
assert sorted(whacks) == sorted([rc_to_cell(2, 4), rc_to_cell(0, 1)]), whacks
# buster tacno na 2 groba
busters = [i - BUSTER_BASE for i in range(BUSTER_BASE, CHERRY_BASE) if mask[i]]
assert sorted(busters) == sorted([rc_to_cell(1, 3), rc_to_cell(4, 8)]), busters
# cherry svuda osim grobova (45-2=43)
cherries = [i for i in range(CHERRY_BASE, ACTION_ICE) if mask[i]]
assert len(cherries) == 43
# ice nije spreman
assert not mask[ACTION_ICE]
assert mask[ACTION_COLLECT_SUN] and mask[ACTION_WAIT]

# grid kanali: kanta na (2,4) -> kanal 2 i hits 3/3
grid = obs[:270].reshape(6, 5, 9)
assert grid[2, 2, 4] == 0.5 and grid[3, 2, 4] == 1.0
assert grid[0, 0, 1] == 0.5 and abs(grid[3, 0, 1] - 1/3) < 1e-6
assert grid[4, 1, 3] == 1.0 and grid[5, 2, 2] == 1.0
print("kodiranje stanja i maska: OK")

# dekoder akcija
assert decode(rc_to_cell(2, 4)) == ("whack", 2, 4)
assert decode(BUSTER_BASE + rc_to_cell(1, 3)) == ("buster", 1, 3)
assert decode(ACTION_ICE) == ("ice", None, None)
print("dekoder akcija: OK")

# determinizam simulatora sa istim seed-om
from bot.sim.simulator import WhackSimulator
s1 = WhackSimulator(seed=42); a = s1.reset()
s2 = WhackSimulator(seed=42); b = s2.reset()
o1, o2 = encode_observation(a), encode_observation(b)
assert np.array_equal(o1, o2)
print("determinizam simulatora: OK")
print("SVI TESTOVI PROSLI")
