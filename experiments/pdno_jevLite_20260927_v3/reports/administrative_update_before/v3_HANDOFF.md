# v3 final evidence and GPT-6 Astra handoff

**Status:** `HANDOFF_READY` after raw evidence freeze. This file was prepared on 2026-09-28T03:50:27+09:00. The final v3 analysis and manuscript have not yet been performed by Astra; delegation is the next action.

## Research scope and locked protocol

The study concerns low-latency control of manufacturing PDE processes (Burgers and heat), not automobiles. “Autonomous” refers to autonomous research execution. Test was opened once only after model, features, families, calibration, and endpoint were frozen. It is now complete and aggregated. Do not retune on test or rewrite any frozen raw artifact. Preserve B4 shared encoder/candidate batching and B3 direct surrogate-gradient method in all comparisons and reporting.

## Terminal evidence

- Locked test: 1,280 parent episodes; 25,600 control episodes; 120 shard records; registered aggregation produced 6 groups. Aggregate SHA256: `28f32812b8e1593da0a8a6f54bafc69460a0712a08e1bbc7991aa840308df3e9`.
- Host E2E: 120 method/PDE/seed/session rows, 20,000 batch-1 requests per row, 2,400,000 requests total, three sequential sessions; elapsed `9,254.4820389 s`. Summary SHA256: `98555f0a22937792548b6826ea8542f3135858334fa83ebfa844d07666ced4bb`.
- Registered latency analysis: 40 groups; SHA256 `f46114814f3f0b59dc27659336c73d9b8ade9d87c0cb62b30022aad879d584f4`.
- Raw evidence freeze: `evidence/raw_evidence_freeze_v3.json`; SHA256 `2fc83da6f7c577d09d1c75ce44e276726a6705383e074974f94d9f321a3eea41`. It covers 240 locked artifacts and all 120 host-latency raw rows. Do not modify these files. Post-freeze state updates and the derived CUDA assessment are documented separately and do not alter the frozen arrays.
- Authors: `AUTHOR_INPUT_REQUIRED.md` remains unresolved by design. Proceed with analysis and manuscript drafting; never invent author identity or submit anything.

## CUDA runtime evidence relevant to the user's question

Read `evidence/cuda_runtime_assessment_v3.json` (SHA256 `f20a168cb1f8bcb9556384cf9738feba8e674f14becaa22028366ae164bb5d72`). It was derived post-freeze from immutable latency raw files. For learned methods, the GPU event span at p99 is about 97–99% of host wall p99, with roughly 0.11 ms difference in the representative aggregate groups. This suggests learned inference is largely on-stream GPU work in this serialized batch-1 endpoint. Snapshot GPU utilization is not a profiler. B1 is CPU-only and slow on Burgers: host p99 15.544 ms; Heat B1 host p99 5.681 ms. Learned Heat P/B4/B5 p99 is around 10.1–10.3 ms; B2/B3 are around 3.9–4.2 ms. The 2 ms target is not met consistently; report hard 5 ms misses by model and PDE from the analysis file. H1 P/B4 p99 ratio: Burgers mean per-seed 0.9859, 95% hierarchical seed/session interval [0.9787,0.9929]; Heat mean 0.9930, interval [0.9849,1.0016]. Describe this cautiously; the three sessions are not independent environments.

Potential follow-up speed work (not performed, no claimed gain): compare `torch.compile` or CUDA Graphs for fixed-shape batch-1 learned inference while verifying identical projected actions and numerical outputs; investigate CPU↔GPU copies and synchronization separately; profile B1 CPU ROM candidate evaluation, then compare semantics-preserving vectorized or GPU-batched rollout; separately profile closed-loop PDE advancement, which calls CPU solver steps. Do not change or rerun the frozen endpoint as if it were the original. Any optimized runtime is a separate amended benchmark.

## Required Astra deliverables

Read this handoff, current `state/RUN_STATE.json`, the root and experiment `AGENTS.md` instructions, all frozen evidence, `config/`, `references.json`, reports/handoff from v2, and existing `AUTHOR_INPUT_REQUIRED.md`. Independently analyze the complete v3 scientific results (including negative gates, locked test, calibration, risk, utility, fairness/comparator limitations, and latency). Then create, in the v3 experiment tree, at minimum:

1. `reports/REPORT_KO.md` — Korean evidence-led final report and explicit scientific decision.
2. `reports/DECISION_PACKET.json` — machine-readable claims, outcome, limitations, next action.
3. `reports/claim_evidence.csv` — each claim with artifact, exact support, and status.
4. `reports/failures_and_limitations.md` — include every failure and correction from the JSONL/run ledger plus scientific limits.
5. `reports/REPRODUCE.md` — exact local RTX 5080 environment and registered commands, with hashes; never claim a GPU path that was not executed.
6. `manuscript_v3.md` (or a clearly named paper draft) — complete manuscript with related work grounded in the supplied references and explicit bounded claims. Keep author block marked `AUTHOR_INPUT_REQUIRED`; do not submit.

Retain negative or inconclusive results, baseline failures, risk/harm and calibration caveats. Do not convert lack of evidence into success. Do not describe Astra's prior v2 audit as the v3 analysis. Clearly report what Astra itself actually analyzed and authored during this task.

## Failure history and no protocol changes

Review `evidence/failures_and_modifications_v3.jsonl` and `state/RUN_LEDGER.md` before writing. The E2E preflight path and frozen 20-snapshot replay-axis mismatches were corrected in the runner without changing method set, request count, source data, endpoint, or timing logic. The empty early attempt directory was preserved. A transient training process receipt was reconstructed transparently from persisted completion evidence and labeled as such. The frozen benchmark returned exit 0 and was followed by registered analysis and raw freeze.
