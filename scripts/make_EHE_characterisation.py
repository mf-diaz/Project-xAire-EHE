"""
EHE_characterisation.pdf, extensive run (10,000 games per treatment).

(a) Agreement pattern of the four heuristics in every per-round decision, all games
    (successful + unsuccessful): 1,620,000 decisions = 30,000 games x 6 players x 9 rounds
    (round 1 is seeded from the empirical first-round distribution, not by the vote).
(b) Common-fund DTW and nMSE (Ensemble vs empirical, ALL games, per treatment and then averaged over
    the three treatments) as a function of the number of simulated games per treatment; mean +/- 1 SD
    over 30 random sub-samples of the 10,000-game pools. Because the scores are per treatment, their
    scale differs from the pooled values in Table 1.

Reads work/decision_pattern_counts.json and work/convergence_curve.csv (simulate_extensive_run.py).
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import _style
from xaire_paths import WORK, RESULTS
_style.apply()

ORDER = [("4", "Unanimous\n(4-0)", "tab:green"), ("3-1", "Majority\n(3-1)", "tab:blue"),
         ("2-1-1", "Plurality\n(2-1-1)", "tab:purple"), ("2-2", "Tie\n(2-2)", "tab:red")]


def main():
    pc = json.load(open(WORK / "decision_pattern_counts.json"))
    cv = pd.read_csv(WORK / "convergence_curve.csv")
    tot = pc["total"]
    print(f"Panel (a): {tot} decisions", {k: round(100 * pc['counts'][k] / tot, 1) for k, _, _ in ORDER})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    vals = [100 * pc["counts"][k] / tot for k, _, _ in ORDER]
    bars = ax1.bar([l for _, l, _ in ORDER], vals, color=[c for _, _, c in ORDER], edgecolor="black", lw=0.8)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10)
    ax1.set_ylabel("Share of per-round decisions (%)"); ax1.set_ylim(0, 47)
    ax1.text(0.03, 0.95, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold", va="top")

    l1, = ax2.plot(cv.n_games, cv.dtw_mean, "-o", color="tab:blue", label="DTW (common fund)")
    ax2.fill_between(cv.n_games, cv.dtw_mean - cv.dtw_sd, cv.dtw_mean + cv.dtw_sd, color="tab:blue", alpha=0.2)
    ax2.set_xscale("log"); ax2.set_xlabel("Simulated games per treatment")
    ax2.set_ylabel("DTW (common fund)", color="tab:blue")
    ax3 = ax2.twinx()
    l2, = ax3.plot(cv.n_games, cv.nmse_mean, "-s", color="tab:red", label=r"nMSE$_{\mathrm{var}}$ (common fund)")
    ax3.fill_between(cv.n_games, cv.nmse_mean - cv.nmse_sd, cv.nmse_mean + cv.nmse_sd, color="tab:red", alpha=0.15)
    ax3.set_ylabel(r"nMSE$_{\mathrm{var}}$ (common fund)", color="tab:red")
    ax2.legend(handles=[l1, l2], loc="upper right")
    ax2.text(0.03, 0.95, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold", va="top")
    plt.tight_layout()
    plt.savefig(RESULTS / "EHE_characterisation.pdf", bbox_inches="tight")
    print("wrote", RESULTS / "EHE_characterisation.pdf")


if __name__ == "__main__":
    main()
