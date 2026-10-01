# Figure design and data map

The data figures are generated from the included records. Figure 1 is an editable TikZ diagram in `figures/fig01_design.tex`, compiled by `scripts/build_diagram.py`. The common entry point `scripts/build_figures.py` builds all seven figures. Figure 1 regeneration requires pdfLaTeX, TikZ and standalone; the ready-to-use PDF is already included. PDF is the primary LaTeX asset; SVG preserves vectors and text; PNG provides a 300 dpi preview. No learned model is executed to draw a figure.

| In-paper label | Asset | Purpose | Input |
|---|---|---|---|
| Figure 1 | fig01_design | Causal loop, P/B4 computation and evidence flow | Executed method description and frozen specification |
| Figure 2 | fig02_prediction | All 24 selected predictive checkpoints | selected_checkpoints.csv |
| Figure 3 | fig03_control | Eight methods across six PDE/condition groups | control_all_methods.csv |
| Figure 4 | fig06_heat | First-step event decomposition, counts retained | heat_constraint_diagnostic.csv |
| Figure 5 | fig04_latency | Pooled p99 and strict 5-ms exceedances | latency_pooled_descriptive.csv |
| Figure 6 | fig05_h1 | Registered seed-specific ratio and bootstrap interval | h1_results.json, derived from latency_analysis_v3.json |
| Figure S1 | figS01_training | Recorded validation curves for three seeds | training_history.csv, derived from checkpoint manifest |

P is orange, B4 is blue, B0 is dark gray, and other controls are neutral gray. Seed markers/line styles supplement color. Prediction panels share ranges; cost panels share ranges within a PDE row; deadline proportions use a common 0–100% scale. Each caption specifies the unit and denominator. Cost intervals are the recorded paired-parent, fixed-denominator intervals. H1 intervals are hierarchical seed/session intervals. They are not interchangeable uncertainty measures.

The schematic is a diagram of the implemented information flow. It is not a physical apparatus, a field solution, or a newly simulated trajectory. Heat bars preserve independent-parent counts and distinguish first-step events from parents without recorded events. No hypothetical heat field is substituted for stored evidence.

Long protocol caveats are centralized in Supplement S2 and the main Discussion. Short, figure-specific qualifications remain in captions where needed to interpret axes, sampling units or intervals.
