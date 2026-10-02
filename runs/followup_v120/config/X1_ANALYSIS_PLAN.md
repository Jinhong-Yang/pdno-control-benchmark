# X1 analysis and reporting plan

## Frozen estimand and execution order

X1 is a paired, post-primary-evaluation diagnostic of candidate design for RS-true. The populations are the 384 Burgers nominal scenarios and the 384 separately generated nonnegative-heat nominal scenarios identified in `X1_SPEC.json`. Every configured cell runs on exactly the same ordered scenario IDs within a PDE. RS-obs, stress populations, and any added cell are outside this scope.

The eight K×H cells and the single feedback-rollout cell are fixed in the JSON. Run the Burgers K10/H8 comparator first on all 384 scenarios. Continue to other cells only if its mean episode cost differs from archived Table S22 value 0.341698 by at most 0.0000005, the rounding half-unit at six decimal places. A mismatch stops X1 and is reported with its provenance and no changed numerical tolerance.

Before any locked outcome is inspected, estimate runtime using only train/validation parents. Full scope is 384 scenarios per PDE. If the full scope exceeds the available X1 compute allocation, declare the first 128 scenario IDs in ascending order for both PDEs and all cells before opening locked outcome arrays. Do not choose a reduced scope after any locked result is observed.

## Candidate and rollout implementation

K=10 reuses the archived projected 3×3 offset candidate set and B0 command. K=50 uses the archived timing-sweep construction rule: a square offset lattice over the same slew interval, `ceil(sqrt(K-2))` points per dimension, evenly spaced flattened lattice indices for the first K-2 candidates, followed by projected hold and B0 commands. Projection duplicates are kept and ties use NumPy `argmin` first-index behavior.

Each fixed-action candidate receives the existing objective: mean squared tracking error over the first H forecast endpoints on the 128-point scoring grid, plus the same one-decision effort, slew, and state-bound penalty. FP64 reference trajectories are converted to FP32 before objective arithmetic, matching the frozen scorer's dtype contract. The forecast starts from the 256-point true state, uses zero future disturbance, and preserves the Burgers reference integrator or heat Crank–Nicolson solver. The additional feedback-rollout candidate starts with B0's current action. Its future B0 feedback actions are recomputed from each reference-predicted full state and projected relative to that rollout's prior action; only its first action is passed to the plant. Its tracking score spans H=8 steps and action penalties use that first action, consistent with the common per-decision objective.

Actual closed-loop advancement, sensors, images, disturbances, target, 200 decisions, box/slew projection, and realized stage cost follow the frozen nominal evaluation implementation. The disturbance is never exposed to forecasts. The analysis saves every per-tick selected index, including feedback-rollout selection as index 10, plus resulting action histories.

## Endpoints and inference

For each cell and PDE, report mean episode cost, tracking RMSE, hold and B0 selection rates, and mean absolute applied-action change. The hold index is 4 for K10; for K50 it is the appended penultimate candidate (index 48). The B0 candidate is last (index K-1); the feedback rollout has index 10 in the 11-candidate set. Report baseline excess using both the baseline population mean denominator and a paired bootstrap denominator.

Bootstrap 10,000 paired scenario resamples with replacement using PCG64 / NumPy `default_rng(2026100270)`, the same seed and resampling convention as the frozen E1 interval audit. For fixed-denominator excess, divide each resampled mean paired cost difference by the full-sample B0 mean. For paired-denominator sensitivity, divide by the B0 mean in the same resample. Report two-sided percentile 95% intervals; cells and populations are pointwise and are not a familywise guarantee.

The branch rule is applied separately to fixed-action and feedback cells: Branch A if any of the eight K×H cells has point excess ≤ half the Burgers K10/H8 excess; otherwise Branch C if the feedback-rollout cell meets that threshold; otherwise Branch B. Heat is a parallel population report and does not redefine the manuscript branch. Retain all scenarios, ticks, and tied candidate outcomes. No post-result exclusions or alternative endpoint definitions are permitted.

## Required output and checksums

Save compressed arrays under `results/X1/<population>/<cell>/` with per-scenario cost, tracking RMSE, selected indices, applied actions, state trajectory, and action-change summary. Save a record alongside each array with exact source data, code, config, analysis-plan, baseline archive, and actual Git commit hashes; UTC start/end, scope, solver/device, and SHA-256. Analysis CSV/JSON must be regenerated from these raw arrays by a committed script. This seal must be committed before opening locked results. Frozen archives under `experiments/` and `runs/revision_v2/` are read-only.

## Validation permitted before locked execution

Use train or validation parents only for code tests, solver comparisons, and timing. Confirm K10 candidate actions match the archived implementation, K50 counts/hold/B0 indices follow the frozen rule, projection satisfies the same box/slew limits, the H=8 K10 fixed-action score equals the archived 8-step objective on a fixed validation snapshot, and the feedback rollout recomputes action at each future state. GPU use at this stage is limited to the root-coordinated train/validation microbenchmark. Do not run a locked/test smoke.
