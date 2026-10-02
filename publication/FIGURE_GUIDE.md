# Figure design and data map

This guide is regenerated from `data/figure_provenance.json`. Asset identifiers
are file names, not final manuscript figure numbers. The manuscript and supplement
determine the in-paper numbering and placement.

The current provenance contains **17 generated figures**. The guide builder
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
| fig01_design | Implemented design and follow-up paths | TikZ: causal loop, aligned P/B4, reference-solver and cache paths; distinct primary/follow-up evidence | `data/final_spec_v3.json`, `figures/fig01_design.tex` |
| fig02_prediction | Primary evaluation | Selected field validation error for all 24 predictive checkpoints | `data/selected_checkpoints.csv` |
| fig03_control | Primary evaluation | All eight controllers; primary paired-scenario 95% intervals; dashed 5% margin | `data/control_all_methods.csv` |
| fig04_latency | Primary evaluation | Pooled raw-request p99 and exact 5 ms miss proportions for all methods | `data/latency_pooled_descriptive.csv` |
| fig05_h1 | Primary evaluation | Hierarchical seed/session bootstrap intervals and 25% reduction target | `data/h1_results.json` |
| fig06_heat | Primary evaluation | All observed heat episode events already occur at first advanced step; no later new or upper events | `data/heat_constraint_diagnostic.csv` |
| figS01_training | Primary evaluation | Recorded validation histories; phase boundary at update 2000 | `data/training_history.csv` |
| figR01_oracles | Follow-up diagnostic; populations distinguished explicitly | Primary scenarios only: privileged-state and observed-state reference-solver comparisons. Primary fixed-denominator paired-scenario95% intervals; learned-controller seeds averaged within scenario. Shared horizontal scales within each PDE; no new-heat reference-solver implied. | `data/revision/REVISION_COST_SUMMARY.csv`, `data/control_all_methods.csv` |
| figR05_main_control | Follow-up diagnostic; populations distinguished explicitly | Main control comparison: primary Burgers scenarios including two reference-solver controls; separately generated positive-initial-field heat scenarios and retrained learned controllers. Fixed-denominator paired-scenario95% intervals. Symlog axis is linear between-5% and5%, logarithmic outside; common horizontal scales within each row. No primary-heat reference-solver is compared with new-heat models. | `data/revision/REVISION_COST_SUMMARY.csv`, `data/control_all_methods.csv` |
| figR02_new_heat_constraints | Follow-up diagnostic; populations distinguished explicitly | New nonnegative-initial-field heat population only. Descriptive scenario-and-seed means; duration, maximum and integrated excess accompany binary frequency. Zero observations do not establish a risk guarantee or discriminatory endpoint. | `data/revision/REVISION_COST_SUMMARY.csv` |
| figR03_K_ratio | Follow-up diagnostic; populations distinguished explicitly | Complete20-cell K/cache sweep, mean seed-specific p99 ratio and pointwise95% matched seed/session intervals. Slight horizontal offsets distinguish cache conditions; reference lines1 and0.75. No simultaneous coverage. | `data/revision/E2_ratio_bootstrap.csv` |
| figR04_K_p99 | Follow-up diagnostic; populations distinguished explicitly | Descriptive pooled request p99,45,000 requests per point, shared scales. These pooled percentiles are not the seed-specific ratio statistic. | `data/revision/E2_pooled.csv` |
| figRS02_latency_distributions | Follow-up diagnostic; populations distinguished explicitly | All observed timing values retained in shared log-spaced bins. Primary campaign P/B4/B2 is separated from follow-upK10 P/B4 cache conditions. Density is per log10 latency, not per linear millisecond. | `data/revision/E2_latency_histograms.csv` |
| figRS03_scaling_curves | Follow-up diagnostic; populations distinguished explicitly | D3 Burgers recorded validation curves. Same512scenarios with nested temporal target density; one refitted observer per factor shared across methods/seeds. Lines stop at actual termination; points mark composite-validation-selected updates, not minimum field-error reselection. Dashed horizontal gate0.05. | `data/revision/REVISION_TRAINING_CURVES.csv`, `data/revision/REVISION_TRAINING_SELECTION.csv` |
| figD02_stage_means | Follow-up diagnostic; populations distinguished explicitly | Stage mean durations for all 16 profiles (two PDEs × P/B4 × K=10/200 × cache off/on); ten separately synchronized stages and 500 records per case. The sum is not a separately recorded total profiled-request duration and gives no end-to-end or p99 deletion bound. Stage shares are reported in the accompanying table. | `data/followup/X2/X2a_stage_shares.csv` |
| figD07_error_decomposition | Follow-up diagnostic; populations distinguished explicitly | Non-additive nRMSE dot plot by evaluation family, PDE, target factor, and method. Marker shape identifies E_obs, E_prop, or E_total; horizontal whiskers show the minimum and maximum of three model seeds, not a confidence interval. Ratios and errors do not imply additive variance decomposition. | `data/followup/X3/X3_OPERATOR_SUMMARIES.json` |
| figD06_candidate_design | Follow-up diagnostic; populations distinguished explicitly | X1 candidate-design ablation on matched 128-scenario subsets: fixed-denominator relative excess cost with pointwise paired-scenario 95% intervals and exact applied-action no-change rate. Designated hold-slot rates are reported separately because K=50 contains duplicate zero-offset candidates. Horizontal references mark 0% and the 5% margin. The matched K=10/H=8 baseline is open-circle-marked. Feedback rollout is a separate candidate-set point. The full Burgers n=384 anchor is open-star-marked, outside the matched design trend. | `data/followup/X1/X1_ANALYSIS.csv` |

P is orange, B4 is blue, B0 is dark gray, and other controls use neutral
colors. Seed markers and line styles supplement color. Scientific captions
must distinguish fixed-denominator cost intervals, joint-denominator sensitivity
intervals, hierarchical seed/session latency intervals, and descriptive pooled
quantiles. These are different statistics and must not be interchanged.

The signed-heat event figure (`fig06_heat`) belongs to Supplement S7.
The unused all-zero NH constraint asset is retained for source continuity; it is not included in the manuscript.
SH reference-solver controls must not be plotted as if evaluated on the NH population.
The main control figure uses a symmetric logarithmic axis outside the central
linear region and must retain that explanation in its caption.

Figure 1 depicts implemented information flow, including the diagnostic reference-solver
and cache paths. It is not a physical apparatus or a simulated field image.
Long protocol qualifications belong in Supplement S2 and the Discussion;
sampling units, axes, interval definitions and population distinctions remain
in the relevant captions.
