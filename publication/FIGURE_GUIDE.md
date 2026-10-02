# Figure design and data map

This guide is regenerated from `data/figure_provenance.json`. Asset identifiers
are file names, not final manuscript figure numbers. The manuscript and supplement
determine the in-paper numbering and placement.

The current provenance contains **14 generated figures**. The guide builder
checks each recorded input hash and figure PDF hash before listing it. This
check establishes file correspondence; final-page visual review and scientific
claim validation are separate requirements.

Run `python scripts/build_figures.py` from this publication directory to regenerate
all figures whose required inputs are complete, followed by this guide. No model
training or GPU timing is launched by figure generation. Figure 1 uses editable
TikZ source in `figures/fig01_design.tex`; regeneration requires pdfLaTeX, TikZ
and standalone. The supplied PDFs are the primary LaTeX assets; SVG and PNG
copies support editing and previews.

| Asset | Evidence population / scope | Description | Input files |
|---|---|---|---|
| fig01_design | Implemented design, with original and revision paths | TikZ: causal loop, aligned P/B4, oracle and cache paths; distinct original/revision evidence | `data/final_spec_v3.json`, `figures/fig01_design.tex` |
| fig02_prediction | Original campaign | Selected field validation error for all 24 predictive checkpoints | `data/selected_checkpoints.csv` |
| fig03_control | Original campaign | All eight controllers; primary paired-parent 95% intervals; dashed 5% margin | `data/control_all_methods.csv` |
| fig04_latency | Original campaign | Pooled raw-request p99 and exact 5 ms miss proportions for all methods | `data/latency_pooled_descriptive.csv` |
| fig05_h1 | Original campaign | Hierarchical seed/session bootstrap intervals and 25% reduction target | `data/h1_results.json` |
| fig06_heat | Original campaign | All observed heat episode events already occur at first advanced step; no later new or upper events | `data/heat_constraint_diagnostic.csv` |
| figS01_training | Original campaign | Recorded validation histories; phase boundary at update 2000 | `data/training_history.csv` |
| figR01_oracles | Revision or explicitly separated original/revision comparison | Original parents only: privileged-state and observed-state oracle comparisons. Primary fixed-denominator paired-parent95% intervals; learned-controller seeds averaged within parent. Shared horizontal scales within each PDE; no new-heat oracle implied. | `data/revision/REVISION_COST_SUMMARY.csv`, `data/control_all_methods.csv` |
| figR05_main_control | Revision or explicitly separated original/revision comparison | Main control comparison: original Burgers parents including two oracles; separately generated positive-initial-field heat parents and retrained learned controllers. Fixed-denominator paired-parent95% intervals. Symlog axis is linear between-5% and5%, logarithmic outside; common horizontal scales within each row. No original-heat oracle is compared with new-heat models. | `data/revision/REVISION_COST_SUMMARY.csv`, `data/control_all_methods.csv` |
| figR02_new_heat_constraints | Revision or explicitly separated original/revision comparison | New nonnegative-initial-field heat population only. Descriptive parent-and-seed means; duration, maximum and integrated excess accompany binary frequency. Zero observations do not establish a risk guarantee or discriminatory endpoint. | `data/revision/REVISION_COST_SUMMARY.csv` |
| figR03_K_ratio | Revision or explicitly separated original/revision comparison | Complete20-cell K/cache sweep, mean seed-specific p99 ratio and pointwise95% matched seed/session intervals. Slight horizontal offsets distinguish cache conditions; reference lines1 and0.75. No simultaneous coverage. | `data/revision/E2_ratio_bootstrap.csv` |
| figR04_K_p99 | Revision or explicitly separated original/revision comparison | Descriptive pooled request p99,45,000 requests per point, shared scales. These pooled percentiles are not the seed-specific ratio statistic. | `data/revision/E2_pooled.csv` |
| figRS02_latency_distributions | Revision or explicitly separated original/revision comparison | All observed timing values retained in shared log-spaced bins. Original campaign P/B4/B2 is separated from revisionK10 P/B4 cache conditions. Density is per log10 latency, not per linear millisecond. | `data/revision/E2_latency_histograms.csv` |
| figRS03_scaling_curves | Revision or explicitly separated original/revision comparison | E3 Burgers recorded validation curves. Same512parents with nested temporal target density; one refitted observer per factor shared across methods/seeds. Lines stop at actual termination; points mark composite-validation-selected updates, not minimum field-error reselection. Dashed horizontal gate0.05. | `data/revision/REVISION_TRAINING_CURVES.csv`, `data/revision/REVISION_TRAINING_SELECTION.csv` |

P is orange, B4 is blue, B0 is dark gray, and other controls use neutral
colors. Seed markers and line styles supplement color. Scientific captions
must distinguish fixed-denominator cost intervals, joint-denominator sensitivity
intervals, hierarchical seed/session latency intervals, and descriptive pooled
quantiles. These are different statistics and must not be interchanged.

The original signed-heat event figure (`fig06_heat`) belongs to Supplement S8.
The new heat constraint figure reports a different initial-condition population.
Original heat oracles must not be plotted as if evaluated on that new population.
The main control figure uses a symmetric logarithmic axis outside the central
linear region and must retain that explanation in its caption.

Figure 1 depicts implemented information flow, including the diagnostic oracle
and cache paths. It is not a physical apparatus or a simulated field image.
Long protocol qualifications belong in Supplement S2 and the Discussion;
sampling units, axes, interval definitions and population distinctions remain
in the relevant captions.
