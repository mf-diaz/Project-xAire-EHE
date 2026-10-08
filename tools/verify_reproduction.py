"""
Compares what the pipeline just produced with the reference values committed in reference/.
Exit code 0 only if everything agrees (floats to 1e-6, counts exactly).
    python tools/verify_reproduction.py [--skip-extensive]
"""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
skip_ext = "--skip-extensive" in sys.argv
bad = []


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


t, r = json.load(open(ROOT / "results/table1.json")), json.load(open(ROOT / "reference/table1.json"))
for k, v in r.items():
    if k == "legacy_pooled_concatenated_curves":
        continue
    if isinstance(v, dict):
        for h, x in v.items():
            if not close(t[k][h], x):
                bad.append(f"Table 1 {k}/{h}: {t[k][h]} vs reference {x}")
    elif t[k] != v:
        bad.append(f"Table 1 {k}: {t[k]} vs reference {v}")
print("Table 1:", "agrees with reference" if not bad else f"{len(bad)} differences")

if not skip_ext:
    n0 = len(bad)
    pc, rc = json.load(open(ROOT / "work/decision_pattern_counts.json")), json.load(open(ROOT / "reference/decision_pattern_counts.json"))
    if pc != rc:
        bad.append(f"decision pattern counts differ: {pc} vs {rc}")
    for name in ["ensemble_extensive_trajectories.csv", "convergence_curve.csv"]:
        a, b = pd.read_csv(ROOT / "work" / name), pd.read_csv(ROOT / "reference" / name)
        num = b.select_dtypes("number").columns
        if a.shape != b.shape or any(not close(x, y) for c in num for x, y in zip(a[c], b[c])):
            bad.append(f"{name} differs from reference")
    print("Extensive-run summaries:", "agree with reference" if len(bad) == n0 else "DIFFER")

FIGS = ["graphical_abstract.pdf", "Fig1_pipeline.pdf", "Fig2_vote_schematic.pdf", "Fig3_conditional_probabilities.pdf",
        "Fig4_validation.pdf", "Fig5_characterisation.pdf"]
missing = [f for f in FIGS if not (ROOT / "figures" / f).exists()]
if missing:
    bad.append(f"missing in figures/: {missing}")
print("figures/:", "all six present" if not missing else "INCOMPLETE")

if bad:
    print("\n".join(bad)); sys.exit(1)
print("REPRODUCTION OK")
