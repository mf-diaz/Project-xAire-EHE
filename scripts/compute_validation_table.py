"""
Reproduces MethodsX Table 1 (EHE validation scorecard): H_self, H_peer, H_fund,
H_budget, Ensemble and Random, against the empirical xAire data.

Needs the matched-run synthetic data (simulate_matched_run.py) and the deposit in data/.
    python compute_validation_table.py

Writes results/table1.json and results/table1_rows.tex (the eight rows of Table 1, best model in bold).

AGGREGATION CONVENTIONS (they differ by row, on purpose)

  Payoff (JS)            per initial-endowment level, unweighted mean over the five
                         levels. functions.compute_payoff_similarity_metrics, unmodified.
                         Raw final payoff in MU, bins of 2 MU, natural logarithm,
                         so JS lies in [0, ln 2].
  Payoff (Wasserstein-1) per initial-endowment level, unweighted mean. Payoff is
                         normalised as final / (0.5 * initial endowment), so W1 is
                         dimensionless.
  Inequality (|dGini|),  per wealth treatment on the treatment's mean Lorenz curve of
  Inequality (Lorenz     final payoff, then unweighted mean over the three treatments.
  area)                  Gini = 1 - 2 * trapezoid(Lorenz curve).
  Average contribution   POOLED, as in thesis Table 7.3: the per-round mean contribution
  (nMSE, DTW)            of each endowment level (five curves of 10 rounds, in the order
                         30, 60, 48, 24, 40 MU) is concatenated and ONE nMSE and ONE DTW
                         are computed on the 50-point vectors. nMSE: functions.similarities_Average_all.
  Common fund            POOLED, as in thesis Table 7.4: the mean cumulative fund of
  (nMSE, DTW)            each wealth treatment (three curves) is concatenated into a
                         30-point vector. functions.similarities_Fund_DTWMSE_all.

  All columns use goal-reaching games only. nMSE = MSE / Var(empirical).

RANDOM COLUMN. A matched-size (77 games) baseline in which every contribution is drawn
uniformly from the feasible subset of {0,2,4} MU (a player cannot spend more than the
endowment left). One 77-game draw is noisy, so the reported value is the mean over
N_REP independent replicates with seeds (SEED, rep); the standard deviation and the
number of valid replicates are stored in results/table1.json. A replicate is valid if it
has at least one goal-reaching game in each treatment and each endowment level.

The script self-checks its light re-implementation of the pooled metrics against the
functions.py output for the Ensemble column before it runs the Random replicates.
"""
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from fastdtw import fastdtw
from xaire_paths import WORK, RESULTS, load_participants
from functions import (
    GAME_TYPES, ENDOWMENT_CONFIGS, N_PLAYERS, ROUNDS, GOAL, round_columns,
    lorenz_curve, nMSE_variance, compute_payoff_similarity_metrics,
    similarities_Average_all, similarities_Fund_DTWMSE_all,
)

SEED = 42
N_REP = 200
HYPS = ["H0", "H1", "H2", "H3", "Hybridisation"]
LABELS = {"H0": "H_self", "H1": "H_peer", "H2": "H_fund", "H3": "H_budget",
          "Hybridisation": "Ensemble", "Random": "Random"}
TREATMENTS = ["EQUAL", "UNEQUAL-H", "UNEQUAL-L"]          # groupby (alphabetical) order used in functions.py
ENDOW_ORDER = [30, 60, 48, 24, 40]                       # order of first appearance in the xAire data, as in functions.py
DIST = lambda a, b: abs(a - b)


def simulate_random(rng):
    records, game_id = [], 0
    for game_type, num_games in GAME_TYPES.items():
        for _ in range(num_games):
            game_id += 1
            endowments = ENDOWMENT_CONFIGS[game_type][:]
            left = list(endowments)
            history = np.zeros((N_PLAYERS, ROUNDS))
            for r in range(ROUNDS):
                for i in range(N_PLAYERS):
                    feasible = [c for c in (0, 2, 4) if c <= left[i]]
                    c = rng.choice(feasible)
                    history[i, r] = c
                    left[i] -= c
            reached = 1 if history.sum() >= GOAL else 0
            for i in range(N_PLAYERS):
                rec = {"partida_id": game_id, "endowment_initial": endowments[i],
                       "control_wealth": game_type, "endowment_current": left[i],
                       "goal_reached": reached}
                rec.update({f"R{r+1}": history[i, r] for r in range(ROUNDS)})
                records.append(rec)
    return pd.DataFrame(records)


def mean_lorenz(df):
    out = {}
    for cw, g in df.groupby("control_wealth"):
        out[cw] = np.mean([lorenz_curve(x["endowment_current"].values)[1]
                           for _, x in g.groupby("partida_id")], axis=0)
    return out


def gini_from_curve(y):
    return 1 - 2 * np.trapezoid(y, np.linspace(0, 1, len(y)))


def payoff_metrics(real, sim, js_real_sim):
    """JS (from functions), Wasserstein-1, per endowment, unweighted mean."""
    w1 = []
    for e in sorted(real["endowment_initial"].unique()):
        r = real.loc[real.endowment_initial == e, "normalized_payoff"]
        s = sim.loc[sim.endowment_initial == e, "normalized_payoff"]
        w1.append(wasserstein_distance(r, s))
    return float(np.mean(js_real_sim)), float(np.mean(w1))


def inequality_metrics(real_curves, real_gini, sim):
    sim_curves = mean_lorenz(sim)
    dg, area = [], []
    for cw in real_curves:
        yr, ys = real_curves[cw], sim_curves[cw]
        dg.append(abs(real_gini[cw] - gini_from_curve(ys)))
        area.append(np.trapezoid(np.abs(yr - ys), np.linspace(0, 1, len(yr))))
    # legacy pooled convention (concatenate the three curves, one trapezoid on 21 points)
    yrA = np.concatenate([real_curves[c] for c in TREATMENTS])
    ysA = np.concatenate([sim_curves[c] for c in TREATMENTS])
    x = np.linspace(0, 1, len(yrA))
    legacy = (abs(gini_from_curve(yrA) - gini_from_curve(ysA)), np.trapezoid(np.abs(yrA - ysA), x))
    return float(np.mean(dg)), float(np.mean(area)), legacy


def pooled_avg(real, sim):
    m1 = real[["endowment_initial"] + round_columns].groupby("endowment_initial").mean()
    m2 = sim[["endowment_initial"] + round_columns].groupby("endowment_initial").mean()
    a, b = [], []
    for e in ENDOW_ORDER:
        a.extend(m1.loc[e].values); b.extend(m2.loc[e].values)
    a, b = np.array(a), np.array(b)
    return nMSE_variance(a, b), fastdtw(a, b, dist=DIST)[0]


def pooled_fund(real, sim):
    def mean_cum(df):
        rows = []
        for _, g in df.groupby("partida_id"):
            rows.append({"control_wealth": g["control_wealth"].iloc[0],
                         **dict(zip(round_columns, np.cumsum(g[round_columns].sum().values)))})
        return pd.DataFrame(rows).groupby("control_wealth").mean()
    mr, ms = mean_cum(real), mean_cum(sim)
    a = np.concatenate([mr.loc[t, round_columns].values for t in mr.index])
    b = np.concatenate([ms.loc[t, round_columns].values for t in mr.index])
    return nMSE_variance(a, b), fastdtw(a, b, dist=DIST)[0]


def prep(df):
    df = df.copy()
    df["normalized_payoff"] = df["endowment_current"] / (0.5 * df["endowment_initial"])
    return df[df["goal_reached"] == 1]


def all_metrics(real, sim, real_curves, real_gini):
    js_dict = compute_payoff_similarity_metrics(real, {"x": sim})
    js = [js_dict[(e, "JSD", "x")] for e in sorted(real["endowment_initial"].unique())]
    js_m, w1_m = payoff_metrics(real, sim, js)
    dg, area, _ = inequality_metrics(real_curves, real_gini, sim)
    an, ad = pooled_avg(real, sim)
    fn, fd = pooled_fund(real, sim)
    return dict(JS=js_m, W1=w1_m, dGini=dg, LorenzArea=area, AvgNMSE=an, AvgDTW=ad, FundNMSE=fn, FundDTW=fd)


def write_latex_rows(results, cols):
    labels = [("Payoff distribution (JS)", "JS"), ("Payoff distribution (Wasserstein-1)", "W1"),
              (r"Inequality ($|\Delta$Gini$|$)", "dGini"), ("Inequality (Lorenz area)", "LorenzArea"),
              ("Average contribution (nMSE)", "AvgNMSE"), ("Average contribution (DTW)", "AvgDTW"),
              ("Common fund (nMSE)", "FundNMSE"), ("Common fund (DTW)", "FundDTW")]
    lines = []
    for label, key in labels:
        vals = [results[key][h] for h in cols]
        best = min(range(5), key=lambda i: vals[i])          # best of the five models, not Random
        cells = [(r"\textbf{%.3f}" if i == best else "%.3f") % v for i, v in enumerate(vals)]
        lines.append(f"{label:<36}& " + " & ".join(cells) + r" \\")
    (RESULTS / "table1_rows.tex").write_text("\n".join(lines) + "\n")


def main():
    real = prep(load_participants())
    real_curves = mean_lorenz(real)
    real_gini = {cw: gini_from_curve(y) for cw, y in real_curves.items()}
    sims = {h: prep(pd.read_csv(WORK / f"df_users_{h}.csv")) for h in HYPS}

    results = {k: {} for k in ["JS", "W1", "dGini", "LorenzArea", "AvgNMSE", "AvgDTW", "FundNMSE", "FundDTW"]}
    legacy = {}

    # model columns: thesis functions for the pooled rows, light code for the rest
    avg_res, _ = similarities_Average_all(real, sims)
    fund_res = similarities_Fund_DTWMSE_all(real, sims)
    for h in HYPS:
        m = all_metrics(real, sims[h], real_curves, real_gini)
        for k in results:
            results[k][h] = m[k]
        # override the pooled nMSE/DTW-fund with the thesis function output and cross-check
        assert abs(avg_res[("ALL", "average_nMSE_variance")][h] - m["AvgNMSE"]) < 1e-9
        assert abs(fund_res[("ALL", "fund_nMSE_variance")][h] - m["FundNMSE"]) < 1e-9
        assert abs(fund_res[("ALL", "fund_dtw")][h] - m["FundDTW"]) < 1e-9
        legacy[h] = inequality_metrics(real_curves, real_gini, sims[h])[2]

    # Random column: mean over N_REP replicates
    reps, n_valid = [], 0
    for rep in range(N_REP):
        rng = np.random.default_rng([SEED, rep])
        sim = prep(simulate_random(rng))
        if (set(sim["control_wealth"]) != set(TREATMENTS)
                or set(sim["endowment_initial"]) != set(real["endowment_initial"])):
            continue
        reps.append(all_metrics(real, sim, real_curves, real_gini))
    n_valid = len(reps)
    for k in results:
        vals = np.array([r[k] for r in reps])
        results[k]["Random"] = float(vals.mean())
        results[k]["Random_sd"] = float(vals.std(ddof=1))
    results["Random_valid_replicates"] = n_valid
    results["legacy_pooled_concatenated_curves"] = {h: {"dGini": v[0], "LorenzArea": v[1]} for h, v in legacy.items()}

    cols = HYPS + ["Random"]
    print(f"{'Metric':<28}" + "".join(f"{LABELS[h]:>12}" for h in cols))
    rows = [("Payoff (JS)", "JS"), ("Payoff (Wasserstein-1)", "W1"),
            ("Inequality (|dGini|)", "dGini"), ("Inequality (Lorenz area)", "LorenzArea"),
            ("Avg. contribution (nMSE)", "AvgNMSE"), ("Avg. contribution (DTW)", "AvgDTW"),
            ("Common fund (nMSE)", "FundNMSE"), ("Common fund (DTW)", "FundDTW")]
    for label, key in rows:
        print(f"{label:<28}" + "".join(f"{results[key][h]:>12.3f}" for h in cols))
    print(f"\nRandom = mean of {n_valid} valid replicates out of {N_REP}; per-row SD in results/table1.json")
    with open(RESULTS / "table1.json", "w") as f:
        json.dump(results, f, indent=2)
    write_latex_rows(results, cols)
    print("wrote", RESULTS / "table1.json", "and", RESULTS / "table1_rows.tex")


if __name__ == "__main__":
    main()
