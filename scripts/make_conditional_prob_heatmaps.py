"""
Fig. 2 of the MethodsX manuscript (conditional_prob_heatmaps / Fig2.pdf): estimated
contribution probabilities of the four heuristics at 60 MU initial endowment.
Reads the four lookup tables in work/ (the same files the simulator samples from, built by
build_conditional_tables.py),
so the figure is the estimation step itself, not a separate computation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _style
from xaire_paths import WORK, FIGURES
_style.apply()

ENDOWMENT = 60
PANELS = [
    ("(a)", r"$H_{\mathrm{self}}$", "H0_Previous_contribution.csv", "Previous contribution"),
    ("(b)", r"$H_{\mathrm{peer}}$", "H1_Others_contribution.csv", "Avg. contribution of others"),
    ("(c)", r"$H_{\mathrm{fund}}$", "H2_Cumulative_contribution.csv", "Common fund (invested per group of six)"),
    ("(d)", r"$H_{\mathrm{budget}}$", "H3_Spending_coins.csv", "Percentage of initial endowment spent"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
for ax, (letter, title, fname, xlabel) in zip(axes.ravel(), PANELS):
    df = pd.read_csv(WORK / fname)
    sub = df[df["endowment_initial"] == ENDOWMENT]
    labels = [str(int(float(v))) if fname.startswith("H0") else str(v) for v in sub.iloc[:, 1]]
    probs = sub.iloc[:, 2:5].values.astype(float).T
    im = ax.imshow(probs, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["0", "2", "4"])
    ax.set_xlabel(xlabel); ax.set_ylabel("Player's contribution")
    ax.set_title(title, fontsize=11)
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold")
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            ax.text(j, i, f"{probs[i, j]:.2f}", ha="center", va="center", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
plt.tight_layout()
plt.savefig(FIGURES / "Fig3_conditional_probabilities.pdf", bbox_inches="tight", metadata={"CreationDate": None})
print("wrote", FIGURES / "Fig3_conditional_probabilities.pdf")
