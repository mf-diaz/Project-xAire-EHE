# figures/

The figures of the MethodsX article, by figure number.

| File | Figure | Origin |
|---|---|---|
| `graphical_abstract.pdf` | Graphical abstract | Hand-drawn, as submitted |
| `Fig1_pipeline.pdf` | Fig. 1, pipeline schematic | Hand-drawn, as submitted |
| `Fig2_vote_schematic.pdf` | Fig. 2, plurality-vote schematic | Hand-drawn, as submitted |
| `Fig3_conditional_probabilities.pdf` | Fig. 3, conditional-probability heatmaps | `scripts/make_conditional_prob_heatmaps.py` |
| `Fig4_validation.pdf` | Fig. 4, empirical against ensemble trajectories | `scripts/make_validation_overlays.py` |
| `Fig5_characterisation.pdf` | Fig. 5, agreement pattern and convergence of the scores | `scripts/make_EHE_characterisation.py` |

Figs 3 to 5 are rewritten by `./run_all.sh` (Figs 4 and 5 only without `--skip-extensive`). `git status` after a
run shows whether the regenerated files differ from the ones stored here.
