"""
Table 2 of the MethodsX article: empirical distribution of first-round contributions by initial
endowment, estimated on all 462 participants (successful and unsuccessful games). These frequencies
seed the first round of every synthetic player; functions.py stores them in `round1_probabilities`.
This script recomputes them from the deposit and checks that the stored values agree (to the six
decimals stored there), so the link between the data and the simulator is explicit.
"""
import sys
import numpy as np
from xaire_paths import load_participants
from functions import round1_probabilities


def main():
    df = load_participants()
    print(f"{'Endowment (MU)':<16}{'P(0)':>8}{'P(2)':>8}{'P(4)':>8}")
    ok = True
    for e in sorted(df["endowment_initial"].unique()):
        vc = df.loc[df["endowment_initial"] == e, "R1"].value_counts(normalize=True)
        p = [vc.get(0.0, 0.0), vc.get(2.0, 0.0), vc.get(4.0, 0.0)]
        print(f"{e:<16}{p[0]:>8.2f}{p[1]:>8.2f}{p[2]:>8.2f}")
        ok &= bool(np.allclose(p, round1_probabilities[int(e)], atol=1e-6))
    print("matches functions.round1_probabilities:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
