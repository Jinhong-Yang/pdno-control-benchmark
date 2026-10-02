# v3 feature-use and runtime-boundary audit

## Common observation interface

The closed-loop runner constructs one causal observation dictionary for each method at each tick. It contains the same sensor and image values, masks, capture/receive timestamps, ages, known material context, goal, previous applied action and eight-step applied-action history. policy_action dispatches the same dictionary to B0/B1 or converts its declared fields for the learned model. In the generated query arrays, parent role and timestamps are audited before training.

This establishes a common source and availability contract. It does not imply that all algorithms use every available field in the same way:

| Method family | Fields consumed by its current algorithm |
|---|---|
| B0 modal LQR | newest available sensor row and available image samples, material parameters, heat goal, previous applied action |
| B1 Galerkin ROM | same modal measurement helper and inputs as B0, then evaluates the common candidate set |
| B2/B3 direct policies | eight-step sensor values, masks and ages; current image and mask; image age/validity; goal coefficients; material context; previous action and applied-action history |
| P, P-no-rank, B4, B5 | the same learned observation encoder and feature set as B2/B3; methods differ in action-conditioned rollout/physics/ranking mechanism |

The classical baselines are allowed the same raw causal observation object but summarize it differently: their modal observer takes the newest available value per sensor and image sample, without explicitly propagating stale measurements using age or the full sensor/action history. This is a declared algorithmic limitation and potential comparator disadvantage. Do not describe the controller comparison as equal feature utilization. The primary P/B4/B5/P-no-rank comparisons share the learned encoder and input feature path; all methods share the same sensor/image events and applied-action execution history.

## Action and timing boundaries

Candidate actions are generated from the previous applied action and common box/slew limits. B0/B1/learned candidate policies produce their selected or direct proposal; a common project_box_slew operation produces the applied action that advances the simulator and becomes the next history entry. Teacher queries use the same candidate constraints and the same initial state, with future disturbance set to zero and explicitly identified as privileged counterfactual supervision.

The validation closed-loop runtime pilot times the whole episode, including the CPU truth solver, and separately retains per-policy-call times. Neither quantity is host-ready E2E request latency. A future post-test host benchmark must separately measure host preparation, host-to-device transfer, encoder/operator/candidate scoring/projection, device-to-host return, warmup and synchronized completion at request batch one.

## Evidence boundary

The source paths reviewed are src/pdno/evaluation/closed_loop.py, src/pdno/controllers/linear.py, src/pdno/controllers/rom.py, src/pdno/models/encoders.py, src/pdno/models/policies.py, and src/pdno/models/operators.py. The full v3 causal/action tests and generated train/validation shard audits passed. Calibration and locked test remain unopened.
