"""
Estimates the four conditional-probability lookup tables (H0 self, H1 peer, H2 fund, H3 budget) from the
cleaned participant file, using all 462 participants (successful and unsuccessful games). The estimation
follows notebook 3 of the original thesis pipeline, without the plotting.

Writes to work/: H0_Previous_contribution.csv, H1_Others_contribution.csv,
H2_Cumulative_contribution.csv, H3_Spending_coins.csv
"""
import numpy as np
import pandas as pd
from xaire_paths import WORK, ROUND_COLS, load_participants

CHOICES = [0.0, 2.0, 4.0]


def h0(df):
    out = {}
    for e in df["endowment_initial"].unique():
        sub = df[df["endowment_initial"] == e]
        tm = pd.DataFrame(0, index=CHOICES, columns=CHOICES, dtype=float)
        for _, p in sub.iterrows():
            c = p[ROUND_COLS].values
            for i in range(len(c) - 1):
                tm.loc[c[i + 1], c[i]] += 1
        out[e] = tm.div(tm.sum(axis=0), axis=1).fillna(0).T
    t = pd.concat({e: m.reset_index() for e, m in out.items()}, names=["endowment_initial"])
    return t.reset_index(level=1, drop=True).rename(columns={"index": "Previous_Contribution"})


def h1(df):
    rows = []
    for pid, g in df.groupby("partida_id"):
        for e, ge in g.groupby("endowment_initial"):
            for _, p in ge.iterrows():
                for k in range(len(ROUND_COLS) - 1):
                    others = g[g["id"] != p["id"]][ROUND_COLS[k]].values
                    rows.append({"endowment_initial": e, "previous_avg_others": others.mean(),
                                 "current_contribution": p[ROUND_COLS[k + 1]]})
    d = pd.DataFrame(rows)
    bins, labels = [0, 1, 2, 3, float("inf")], ["[0,1)", "[1,2)", "[2,3)", "[3,4]"]
    d["previous_avg_others_binned"] = pd.cut(d["previous_avg_others"], bins=bins, labels=labels,
                                             include_lowest=True, right=False)
    return d.groupby(["endowment_initial", "previous_avg_others_binned"])["current_contribution"] \
            .value_counts(normalize=True).unstack()


def h2(df):
    rows = []
    for pid, g in df.groupby("partida_id"):
        for e, ge in g.groupby("endowment_initial"):
            for _, p in ge.iterrows():
                cum = 0
                for k in range(len(ROUND_COLS) - 1):
                    cum += g[ROUND_COLS[k]].sum()
                    rows.append({"endowment_initial": e, "previous_cumulative_contribution": cum,
                                 "current_contribution": p[ROUND_COLS[k + 1]]})
    d = pd.DataFrame(rows)
    bins = [0, 24, 48, 72, 96, 120, float("inf")]
    labels = ["[0,24)", "[24,48)", "[48,72)", "[72,96)", "[96,120)", "≥120"]
    d["previous_cumulative_contribution_binned"] = pd.cut(d["previous_cumulative_contribution"], bins=bins,
                                                          labels=labels, include_lowest=True, right=False)
    return d.groupby(["endowment_initial", "previous_cumulative_contribution_binned"])["current_contribution"] \
            .value_counts(normalize=True).unstack()


def h3(df):
    rows = []
    for _, p in df.iterrows():
        cum = 0
        for k in range(len(ROUND_COLS) - 1):
            cum += p[ROUND_COLS[k]]
            rows.append({"endowment_initial": p["endowment_initial"],
                         "spent_ratio": cum / p["endowment_initial"],
                         "current_contribution": p[ROUND_COLS[k + 1]]})
    d = pd.DataFrame(rows)
    bins, labels = [0, 3 / 10, 6 / 10, float("inf")], ["[0,30%)", "[30%,60%)", "≥60%"]
    d["spent_ratio_binned"] = pd.cut(d["spent_ratio"], bins=bins, labels=labels, include_lowest=True, right=False)
    m = d.groupby(["endowment_initial", "spent_ratio_binned"])["current_contribution"] \
         .value_counts(normalize=True).unstack()
    cat = pd.Categorical(labels, categories=labels, ordered=True)
    parts = []
    for e in m.index.levels[0]:
        x = m.loc[e].reindex(index=cat, columns=CHOICES, fill_value=0).fillna(0)
        x["endowment_initial"] = e
        parts.append(x.reset_index().set_index(["endowment_initial", "spent_ratio_binned"]))
    return pd.concat(parts).sort_index()


def main():
    df = load_participants()
    for name, fn in [("H0_Previous_contribution", h0), ("H1_Others_contribution", h1),
                     ("H2_Cumulative_contribution", h2), ("H3_Spending_coins", h3)]:
        fn(df).to_csv(WORK / f"{name}.csv", index=True)
        print("wrote", WORK / f"{name}.csv")


if __name__ == "__main__":
    main()
