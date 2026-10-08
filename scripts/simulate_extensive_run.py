"""
Extensive run: 30,000 games of the ensemble (10,000 per treatment), seed 42, as in notebook 5 of the
thesis pipeline. It is slow (tens of minutes on one core) and needs the lookup tables from
build_conditional_tables.py. The raw run (about 100 MB) is not stored; only the three small summaries
that the article uses are written to work/:

  ensemble_extensive_trajectories.csv   mean common-fund and average-contribution trajectory per treatment
  decision_pattern_counts.json          agreement pattern of the four heuristics in every per-round decision
  convergence_curve.csv                 common-fund DTW and nMSE against the data as a function of the number
                                        of simulated games per treatment (30 sub-samples of the run)

Use --save-run to also keep the full run as work/df_users_hybrid_sim.csv.
"""
import argparse, json, time
from collections import Counter
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from functions import simulate_games_dataframe_hybrid, convert_defaultdict_to_dict
from xaire_paths import WORK, ROUND_COLS, load_participants

SEED = 42
T = ["EQUAL", "UNEQUAL-L", "UNEQUAL-H"]
CHECKPOINTS = [10, 25, 50, 100, 250, 500, 1000, 2000]
N_REP = 30
FILES = {"H0": "H0_Previous_contribution.csv", "H1": "H1_Others_contribution.csv",
         "H2": "H2_Cumulative_contribution.csv", "H3": "H3_Spending_coins.csv"}


def fund(df, t):
    return df[df.control_wealth == t].groupby("partida_id")[ROUND_COLS].sum().cumsum(axis=1).mean(axis=0).values


def avg(df, t):
    return df[df.control_wealth == t][ROUND_COLS].mean(axis=0).values


def pattern_counts(counters):
    tot, n = Counter(), 0
    for c in counters:
        for rounds in c["contribution_variability"].values():
            for hd in rounds.values():
                k = min(len(hd[h]) for h in ["H0", "H1", "H2", "H3"])
                for i in range(k):
                    v = [hd[h][i] for h in ["H0", "H1", "H2", "H3"]]
                    cs = tuple(sorted(Counter(v).values(), reverse=True))
                    tot[{(4,): "4", (3, 1): "3-1", (2, 2): "2-2"}.get(cs, "2-1-1")] += 1
                    n += 1
    return {"total": n, "counts": dict(tot)}


def convergence(emp, pool):
    ef = {t: fund(emp, t) for t in T}
    ids = {t: pool[pool.control_wealth == t].partida_id.unique() for t in T}
    rng = np.random.default_rng(SEED)
    rows = []
    for n in CHECKPOINTS:
        D, N = [], []
        for _ in range(N_REP):
            d, m = [], []
            for t in T:
                ch = rng.choice(ids[t], size=min(n, len(ids[t])), replace=False)
                y = fund(pool[pool.partida_id.isin(ch)], t)
                d.append(fastdtw(ef[t], y, dist=lambda a, b: abs(a - b))[0])
                m.append(np.mean((ef[t] - y) ** 2) / np.var(ef[t]))
            D.append(np.mean(d)); N.append(np.mean(m))
        rows.append(dict(n_games=n, dtw_mean=np.mean(D), dtw_sd=np.std(D), nmse_mean=np.mean(N), nmse_sd=np.std(N)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-run", action="store_true")
    a = ap.parse_args()
    emp = load_participants()
    datasets = {h: pd.read_csv(WORK / f) for h, f in FILES.items()}
    t0 = time.time()
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    pool, _, cs, cu = simulate_games_dataframe_hybrid(datasets, "Hybridisation", rng)
    print(f"simulated {pool['partida_id'].nunique()} games in {(time.time() - t0) / 60:.1f} min", flush=True)
    if a.save_run:
        pool.to_csv(WORK / "df_users_hybrid_sim.csv", index=False)

    rows = [dict(control_wealth=t, round=k + 1, avg_contribution=avg(pool, t)[k], common_fund=fund(pool, t)[k])
            for t in T for k in range(10)]
    pd.DataFrame(rows).to_csv(WORK / "ensemble_extensive_trajectories.csv", index=False)
    pc = pattern_counts([convert_defaultdict_to_dict(cs), convert_defaultdict_to_dict(cu)])
    json.dump(pc, open(WORK / "decision_pattern_counts.json", "w"), indent=2)
    convergence(emp, pool).to_csv(WORK / "convergence_curve.csv", index=False)
    print("wrote the three summaries to", WORK, f"({(time.time() - t0) / 60:.1f} min in total)")


if __name__ == "__main__":
    main()
