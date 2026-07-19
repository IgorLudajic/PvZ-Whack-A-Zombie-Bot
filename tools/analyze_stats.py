# -*- coding: utf-8 -*-
"""
Analiza statistike odigranih partija (stats/game_stats.csv) za diplomski rad.

    python tools/analyze_stats.py

Ispisuje zbirni izveštaj i pravi grafike u stats/analysis/:
  - win rate i korišćenje kosačica kroz partije,
  - raspodela upotrebe biljaka (šta agent smatra najpametnijom strategijom),
  - struktura ubistava po tipu zombija.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import config


def load_rows():
    path = os.path.join(config.STATS_DIR, "game_stats.csv")
    if not os.path.exists(path):
        print(f"Nema statistike: {path} - prvo odigraj partiju (python main.py).")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = load_rows()
    n = len(rows)
    wins = sum(r["result"] == "WIN" for r in rows)
    mowers = np.array([int(r["mowers_used"]) for r in rows])
    plants = {
        "Potato Mine": np.array([int(r["potato_mines"]) for r in rows]),
        "Grave Buster": np.array([int(r["gravebusters"]) for r in rows]),
        "Ice-shroom": np.array([int(r["ices"]) for r in rows]),
    }
    kills = {
        "običan (1 udarac)": np.array([int(r["kills_zombie"]) for r in rows]),
        "čunj (2 udarca)": np.array([int(r["kills_conehead"]) for r in rows]),
        "kanta (3 udarca)": np.array([int(r["kills_buckethead"]) for r in rows]),
    }

    print("=" * 60)
    print(f"  ANALIZA {n} ODIGRANIH PARTIJA")
    print("=" * 60)
    print(f"  Win rate:                  {100 * wins / n:6.1f}%  ({wins}/{n})")
    print(f"  Partije bez kosačica:      {100 * np.mean(mowers == 0):6.1f}%")
    print(f"  Kosačica po partiji:       {mowers.mean():6.2f}")
    print(f"  Prosečno trajanje:         {np.mean([float(r['duration_s']) for r in rows]):6.1f} s")
    print(f"  Prosečno ubistava:         {np.mean([int(r['kills_total']) for r in rows]):6.1f}")
    print("\n  Upotreba biljaka po partiji (naučena strategija):")
    for name, arr in plants.items():
        print(f"    {name:<14} {arr.mean():5.2f}  (koristi se u {100 * np.mean(arr > 0):.0f}% partija)")
    print("\n  Struktura ubistava:")
    total_kills = sum(arr.sum() for arr in kills.values())
    for name, arr in kills.items():
        print(f"    {name:<22} {arr.sum():6d}  ({100 * arr.sum() / max(total_kills, 1):.1f}%)")
    print("=" * 60)

    make_plots(rows, plants, kills, mowers)


def make_plots(rows, plants, kills, mowers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(config.STATS_DIR, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    n = len(rows)
    x = np.arange(1, n + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1) kumulativni win rate + kosačice
    ax = axes[0][0]
    wins_cum = np.cumsum([r["result"] == "WIN" for r in rows]) / x * 100
    ax.plot(x, wins_cum, color="tab:green", label="Win rate (kumulativno)")
    ax.bar(x, mowers * 10, color="tab:red", alpha=0.4, label="Kosačice x10")
    ax.set_xlabel("Partija"); ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.legend(); ax.set_title("Uspešnost i korišćenje kosačica")

    # 2) upotreba biljaka
    ax = axes[0][1]
    for name, arr in plants.items():
        ax.plot(x, arr, marker="o", ms=3, label=name)
    ax.set_xlabel("Partija"); ax.set_ylabel("Sadnji po partiji")
    ax.legend(); ax.set_title("Upotreba biljaka (strategija agenta)")

    # 3) struktura ubistava (prosek)
    ax = axes[1][0]
    names = list(kills.keys())
    means = [kills[k].mean() for k in names]
    ax.bar(names, means, color=["tab:green", "tab:orange", "tab:red"])
    ax.set_ylabel("Prosečno po partiji"); ax.set_title("Ubistva po tipu zombija")

    # 4) trajanje partija
    ax = axes[1][1]
    ax.plot(x, [float(r["duration_s"]) for r in rows], color="tab:blue", marker="o", ms=3)
    ax.set_xlabel("Partija"); ax.set_ylabel("s"); ax.set_title("Trajanje partija")

    fig.tight_layout()
    out = os.path.join(out_dir, "analysis.png")
    fig.savefig(out, dpi=130)
    print(f"\nGrafici: {out}")


if __name__ == "__main__":
    main()
