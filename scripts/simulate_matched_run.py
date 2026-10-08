"""
Matched run: simulates the same number of games and treatment configurations as the real experiment
(77 games: 6 Equal, 27 Unequal-L, 44 Unequal-H) once for each single heuristic and once for the
ensemble ("Hybridisation" in the code), each with seed 42, as in notebook 4 of the thesis pipeline.
Needs the lookup tables from build_conditional_tables.py.

Writes to work/: df_users_H0.csv ... df_users_H3.csv, df_users_Hybridisation.csv
"""
import numpy as np
import pandas as pd
from functions import simulate_games_dataframe
from xaire_paths import WORK

SEED = 42
FILES = {"H0": "H0_Previous_contribution.csv", "H1": "H1_Others_contribution.csv",
         "H2": "H2_Cumulative_contribution.csv", "H3": "H3_Spending_coins.csv"}


def main():
    datasets = {h: pd.read_csv(WORK / f) for h, f in FILES.items()}
    for hypo in ["H0", "H1", "H2", "H3", "Hybridisation"]:
        np.random.seed(SEED)
        rng = np.random.default_rng(SEED)
        df = simulate_games_dataframe(datasets, hypo, rng)
        df.to_csv(WORK / f"df_users_{hypo}.csv", index=False)
        print("wrote", WORK / f"df_users_{hypo}.csv", df.shape)


if __name__ == "__main__":
    main()
