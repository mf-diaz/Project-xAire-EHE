"""
validation_overlays.pdf: (a) common-fund trajectory and (b) average contribution per round,
empirical vs Ensemble, per wealth treatment.

  * Empirical curves: ALL 77 real games (successful and unsuccessful), from the deposit.
  * Ensemble curves: the mean trajectory of the EXTENSIVE run (10,000 games per treatment, all games),
    i.e. the expected ensemble behaviour with sampling noise removed, from
    work/ensemble_extensive_trajectories.csv (simulate_extensive_run.py). The 77-game matched run used
    for Table 1 is too small to draw smooth curves (the Equal treatment has only six games).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _style
from xaire_paths import WORK, RESULTS, ROUND_COLS, load_participants
_style.apply()
plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})   # larger fonts for this figure only

R = ROUND_COLS
COL = {"EQUAL": "tab:blue", "UNEQUAL-L": "tab:green", "UNEQUAL-H": "tab:red"}
NAME = {"EQUAL": "Equal", "UNEQUAL-L": "Unequal-L", "UNEQUAL-H": "Unequal-H"}   # legend labels as in the text
SUMMARY = WORK / "ensemble_extensive_trajectories.csv"


def fund(df, t):
    return df[df.control_wealth == t].groupby("partida_id")[R].sum().cumsum(axis=1).mean(axis=0).values


def avg(df, t):
    return df[df.control_wealth == t][R].mean(axis=0).values


def main():
    if not SUMMARY.exists():
        raise SystemExit(f"Missing {SUMMARY}; run simulate_extensive_run.py first.")
    emp = load_participants()
    ens = pd.read_csv(SUMMARY)
    x = np.arange(1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, emp_fn, col, ylab, letter in [
        (axes[0], fund, "common_fund", "Common fund (MU)", "(a)"),
        (axes[1], avg, "avg_contribution", "Average contribution (MU)", "(b)"),
    ]:
        for t, c in COL.items():
            ax.plot(x, emp_fn(emp, t), "-o", color=c, lw=2.2, ms=6, label=f"{NAME[t]} empirical")
            ax.plot(x, ens[ens.control_wealth == t].sort_values("round")[col].values, "--s",
                    color=c, lw=1.7, ms=6, mfc="white", label=f"{NAME[t]} ensemble")
        ax.set_xlabel("Round"); ax.set_ylabel(ylab)
        ax.text(0.02, 0.95, letter, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")
    axes[0].axhline(120, color="grey", ls=":", lw=1)
    axes[0].legend(loc="lower right", frameon=True, fontsize=11.5)
    plt.tight_layout()
    plt.savefig(RESULTS / "validation_overlays.pdf", bbox_inches="tight")
    print("wrote", RESULTS / "validation_overlays.pdf")


if __name__ == "__main__":
    main()
