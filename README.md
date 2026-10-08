# xAire EHE: reproduction code for the MethodsX article

Code that regenerates every number, table and data-driven figure of the MethodsX article on the
Empirical Heuristic-Ensemble (EHE) model of a collective-risk dilemma, starting only from the public
Data in Brief deposit. No empirical data are stored in this repository.

## Input

Download the deposit from CORA.RDR, DOI [10.34810/DATA3672](https://doi.org/10.34810/DATA3672), and unzip
it inside `data/`. The downloaded zip unpacks to `Readme.md` and a folder `xAire_CRD_dataset/`, so that
`data/xAire_CRD_dataset/processed/participants.csv` exists (see `data/WHERE_TO_PUT_THE_DEPOSIT.md`). Only
`processed/participants.csv` and `processed/games.csv` are read. In the deposit the wealth treatment
(`control_wealth`) is stored per game, so the scripts join it onto the participants through `partida_id`;
they stop with a message if the deposit does not contain the 462 participants, 77 games (6 Equal, 27
Unequal-L, 44 Unequal-H) and the endowment structure of the three treatments.

## Run

```
pip install -r requirements.txt     # Python 3.11+; see requirements.txt if fastdtw fails to build
./run_all.sh                        # everything; the extensive run took about two hours (124 min) on one core
./run_all.sh --skip-extensive       # about two minutes: Table 1, Table 2, Fig. 3
```

Tables go to `results/`, intermediate files to `work/` (neither is tracked) and the data-driven figures to
`figures/`, which is tracked (see below). The last step compares the
outcome with `reference/` and prints `REPRODUCTION OK` or lists the differences.

## Pipeline

| Step | Script | Produces |
|---|---|---|
| 1 | `build_conditional_tables.py` | The four conditional-probability lookup tables (H0 self, H1 peer, H2 fund, H3 budget), from all 462 participants |
| 2 | `simulate_matched_run.py` | The 77-game synthetic runs (seed 42) of each heuristic and of the ensemble |
| 3 | `compute_validation_table.py` | Table 1 (`results/table1.json`, `results/table1_rows.tex`), including the random baseline |
| 4 | `compute_first_round_table.py` | Table 2; checks it against the first-round probabilities stored in `functions.py` |
| 5 | `make_conditional_prob_heatmaps.py` | Fig. 3, heatmaps of the lookup tables |
| 6 | `simulate_extensive_run.py` | The 30,000-game ensemble run (10,000 per treatment, seed 42), reduced to three small summaries |
| 7 | `make_validation_overlays.py`, `make_EHE_characterisation.py` | Fig. 4 (validation) and Fig. 5 (characterisation) |

`scripts/functions.py` is the simulation and metric library of the original thesis pipeline, unmodified.
`xaire_paths.py` holds the input and output locations and the checks on the deposit. The two schematic
figures of the article (pipeline and vote diagrams) and the graphical abstract are hand-drawn and have no code.

## Figures

`figures/` holds the figures of the article, named by figure number (`figures/README.md` lists them). Figs 3 to
5 are written there by the pipeline; they are saved without a creation date, so the same code, data and
library versions give byte-identical PDFs. Figs 1, 2 and the graphical abstract are hand-drawn and are stored
as submitted.

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

Starting from the deposit alone, as downloaded from CORA, the lookup tables and the five matched-run
synthetic datasets are identical, to the last digit, to the ones of the original thesis pipeline. Table 1
matches the article in all eight rows. Table 2 matches except one cell: for 48 MU the share contributing 2 MU
is 60/108 = 0.5556, printed here as 0.56 and in the article as 0.55 (the article rounds so that the row sums
to 1.00). The extensive run (30,000 games, seed 42) reproduces the pattern counts exactly and the two
summary curves to the last digit. Tested with Python 3.11, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1,
scikit-learn 1.8.0 and matplotlib 3.10.9; other versions are untested. Reference values are in `reference/`.

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
