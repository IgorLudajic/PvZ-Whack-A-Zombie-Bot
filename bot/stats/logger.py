# -*- coding: utf-8 -*-
"""
Beleženje statistike partija za analizu u radu.

Po partiji: JSON sa svim detaljima (stats/games/...).
Zbirno: stats/game_stats.csv - po red za svaku partiju + win rate do tada.
"""

import csv
import json
import os
from datetime import datetime

import config

CSV_COLUMNS = [
    "timestamp", "result", "duration_s", "kills_total", "kills_zombie",
    "kills_conehead", "kills_buckethead", "clicks", "suns_collected",
    "sun_spent", "gravebusters", "cherries", "ices", "mowers_used",
    "danger_near_count", "win_rate_to_date",
]


class StatsLogger:
    def __init__(self):
        self.csv_path = os.path.join(config.STATS_DIR, "game_stats.csv")
        self.games_dir = os.path.join(config.STATS_DIR, "games")
        os.makedirs(self.games_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_COLUMNS)

    def log_game(self, stats):
        """stats: EpisodeStats (iz simulatora ili prave partije)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        win_rate = self._win_rate_with(stats.result)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                timestamp, stats.result, f"{stats.duration:.2f}",
                stats.total_kills, stats.kills["zombie"], stats.kills["conehead"],
                stats.kills["buckethead"], stats.clicks, stats.suns_collected,
                stats.sun_spent, stats.plants_used["gravebuster"],
                stats.plants_used["cherry"], stats.plants_used["ice"],
                stats.mowers_used, stats.danger_near_count, f"{win_rate:.1f}",
            ])

        json_path = os.path.join(
            self.games_dir, f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "result": stats.result,
                "duration_s": round(stats.duration, 2),
                "kills": stats.kills,
                "kills_total": stats.total_kills,
                "clicks": stats.clicks,
                "suns_collected": stats.suns_collected,
                "sun_spent": stats.sun_spent,
                "plants_used": stats.plants_used,
                "mowers_used": stats.mowers_used,
                "danger_near_count": stats.danger_near_count,
                "closest_approach_cols": round(stats.closest_approach, 2),
                "action_distribution": stats.actions,
            }, f, indent=2, ensure_ascii=False)

        self.print_summary(stats, win_rate)
        return json_path

    def _win_rate_with(self, new_result):
        wins, total = 1 if new_result == "WIN" else 0, 1
        if os.path.exists(self.csv_path):
            with open(self.csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    total += 1
                    if row["result"] == "WIN":
                        wins += 1
        return 100.0 * wins / total

    def print_summary(self, stats, win_rate):
        print("\n" + "=" * 56)
        print("  STATISTIKA PARTIJE")
        print("=" * 56)
        print(f"  Rezultat:           {stats.result}")
        print(f"  Trajanje:           {stats.duration:.1f} s")
        print(f"  Ubijeno zombija:    {stats.total_kills} "
              f"(običnih {stats.kills['zombie']}, čunjeva {stats.kills['conehead']}, "
              f"kanti {stats.kills['buckethead']})")
        print(f"  Klikova:            {stats.clicks}")
        print(f"  Sakupljeno sunca:   {stats.suns_collected} (potrošeno {stats.sun_spent})")
        print(f"  Biljke:             Grave Buster x{stats.plants_used['gravebuster']}, "
              f"Cherry Bomb x{stats.plants_used['cherry']}, "
              f"Ice-shroom x{stats.plants_used['ice']}")
        print(f"  Kosačice:           {stats.mowers_used} "
              f"{'(SAVRŠENO - nijedna!)' if stats.mowers_used == 0 else ''}")
        print(f"  Win rate (ukupno):  {win_rate:.1f}%")
        print("=" * 56)
