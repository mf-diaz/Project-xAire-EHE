# xAire EHE: reproduction code for the MethodsX article

Code that regenerates every number, table and data-driven figure of the MethodsX article on the
Empirical Heuristic-Ensemble (EHE) model of a collective-risk dilemma, starting only from the public
Data in Brief deposit. No empirical data are stored in this repository.

## Input

Download the Data in Brief deposit from CORA.RDR (DOI: [[TO CONFIRM: DOI of the final deposit]]), unzip it
and copy its contents into `data/` (see `data/README.md`). Only `processed/participants.csv` and
`processed/games.csv` are read. In the deposit the wealth treatment (`control_wealth`) is stored per
game, so the scripts join it onto the participants through `partida_id`; they stop with a message if the
deposit does not contain the 462 participants, 77 games (6 Equal, 27 Unequal-L, 44 Unequal-H) and the
endowment structure of the three treatments.

## Run

```
pip install -r requirements.txt     # Python 3.11+; see requirements.txt if fastdtw fails to build
./run_all.sh                        # everything; the extensive run took about two hours (124 min) on one core
./run_all.sh --skip-extensive       # about two minutes: Table 1, Table 2, heatmaps
```

Outputs go to `results/`, intermediate files to `work/`; neither is tracked. The last step compares the
outcome with `reference/` and prints `REPRODUCTION OK` or lists the differences.

## Pipeline

| Step | Script | Produces |
|---|---|---|
| 1 | `build_conditional_tables.py` | The four conditional-probability lookup tables (H0 self, H1 peer, H2 fund, H3 budget), from all 462 participants |
| 2 | `simulate_matched_run.py` | The 77-game synthetic runs (seed 42) of each heuristic and of the ensemble |
| 3 | `compute_validation_table.py` | Table 1 (`results/table1.json`, `results/table1_rows.tex`), including the random baseline |
| 4 | `compute_first_round_table.py` | Table 2; checks it against the first-round probabilities stored in `functions.py` |
| 5 | `make_conditional_prob_heatmaps.py` | Heatmap figure of the lookup tables |
| 6 | `simulate_extensive_run.py` | The 30,000-game ensemble run (10,000 per treatment, seed 42), reduced to three small summaries |
| 7 | `make_validation_overlays.py`, `make_EHE_characterisation.py` | The validation and characterisation figures |

`scripts/functions.py` is the simulation and metric library of the original thesis pipeline, unmodified.
`xaire_paths.py` holds the input and output locations and the checks on the deposit. The two schematic
figures of the article (pipeline and vote diagrams) are hand-drawn and have no code.

## Metric conventions

They differ by row, as in the thesis. Payoff JS (natural logarithm, bounded by ln 2) and Wasserstein-1
(on payoff normalised by half the endowment) are computed per endowment level and averaged. The Gini
difference and the Lorenz area are computed per treatment on the mean Lorenz curve and averaged over the
three treatments. The two dynamic indicators are pooled: the five endowment curves (order 30, 60, 48, 24,
40 MU, 50 points) or the three treatment curves (Equal, Unequal-H, Unequal-L, 30 points) are
concatenated and one nMSE (MSE over the variance of the data) and one FastDTW distance (radius 1) are
computed. DTW grows with series length, so compare it across models within a row only. Table 1 uses
goal-reaching games only. The random baseline is the mean of independent 77-game runs with actions drawn
uniformly from the feasible subset of {0, 2, 4}; replicates without a goal-reaching game in some treatment
or endowment level are discarded (195 of 200 are valid).

## Reproduction status

Starting from the deposit alone, the lookup tables and the five matched-run synthetic datasets are
identical, to the last digit, to the ones of the original thesis pipeline. Table 1 matches the article in all
eight rows. Table 2 matches except one cell: for 48 MU the share contributing 2 MU is 60/108 = 0.5556, printed
here as 0.56 and in the article as 0.55 (the article rounds so that the row sums to 1.00). Tested with Python 3.11, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0 and
matplotlib 3.10.9. Reference values are in `reference/`. Figures are regenerated with the same content,
but PDF files embed their creation date, so they are not byte-identical between runs.

## Safeguards

```
git config core.hooksPath .githooks     # once per clone
python tools/check_repo_clean.py        # also run by the hook before every commit
```

The check fails if raw or processed participant files, `.pkl` caches, files above 25 MB, CSV files with
identifier or free-text columns, anything under `data/`, `work/` or `results/`, or notebooks with outputs
are tracked.

## License

CC0 1.0 (`LICENSE`).
