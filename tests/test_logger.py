# -*- coding: utf-8 -*-
"""Integracioni test: simulatorski EpisodeStats mora čisto proći kroz
StatsLogger (CSV + JSON) - ista putanja se izvršava u pravoj partiji."""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import config
config.STATS_DIR = tempfile.mkdtemp(prefix="pvz_stats_")  # ne diraj prave podatke

from bot.sim.simulator import WhackSimulator
from bot.state import encode_observation, build_action_mask
from bot.agent.policy import TrainedPolicy
from bot.stats.logger import StatsLogger


def main():
    pol = TrainedPolicy(config.POLICY_PATH)
    sim = WhackSimulator(seed=123)
    snap = sim.reset()
    done = False
    while not done:
        a = pol.act(encode_observation(snap), build_action_mask(snap))
        snap, _, done, info = sim.step(a)
    st = info["stats"]

    logger = StatsLogger()
    json_path = logger.log_game(st)

    rows = list(csv.DictReader(open(logger.csv_path, encoding="utf-8")))
    assert len(rows) == 1 and os.path.exists(json_path)
    print("\nJSON:", os.path.basename(json_path))
    print("CSV kolone:", list(rows[0].keys()))
    print("action_distribution:", st.actions)
    print("INTEGRACIJA LOGGER<-STATS: OK")


if __name__ == "__main__":
    main()
