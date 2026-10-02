# PDNO–JevLite v3 state

Updated 2026-09-27 14:48 KST.

Updated 2026-09-27 13:00 KST.

Updated 2026-09-27 11:38 KST. Research scope is low-latency manufacturing PDE control; “autonomous driving” means autonomous research execution. Automotive/MetaDrive artifacts are excluded.

## Current status

- v3 is a new, not-yet-frozen version. It has its own .venv, torch 2.12.0+cu130, CUDA 13.0, and verified local RTX 5080 / driver 591.86.
- G1: inherited solver checks and a new forced 200-tick Burgers/heat temporal-refinement, source-sign and balance audit passed. Heat observer endpoint basis repair now preserves both exact Dirichlet values and the analytic boundary derivative.
- Split: the v3 namespace appears inside parent IDs; metadata-only v1/v2 comparison and actual generated train/validation parent membership have zero overlap.
- Causal/action: 47/47 v3 unit tests passed after the signed heat-bound contract update; smoke and full train/validation trajectory audits passed. All 3,200 train/validation teacher queries pass signed-cost and role audits.
- Data generated: train/validation, calibration trajectories/queries, and six locked parent shards are generated and audited. The one-shot locked test is open and running; no aggregate outcome metrics have been computed or inspected.
- G2 quality pilot: completed in 403.73s; all P/P-no-rank/B4/B5 field nRMSE values exceeded the unchanged 0.05 limit (Burgers 0.235–0.236, heat 0.110–0.113). Teacher-regret medians were zero. Record the field-accuracy failure separately; do not select based on regret alone.
- Validation closed-loop runtime pilot: completed in 30.39s. The 25,600 locked evaluations estimate 13.305h raw or 15.301h with 15% margin, including CPU truth simulation. This is not host-ready E2E latency.
- Pretraining specification: frozen fail-inclusively with 88 inputs hashed after the v3 runner-contract audit passed. Mandatory training is complete: all 36 required checkpoints (P/P-no-rank/B4/B5/B2/B3 × Burgers/heat × seeds 11/23/37) and two shared observers have summaries. The training process exited 0; checkpoint manifest froze 36+2, retained the G2 failure, and has `test_opened=false`. A rehash of all 88 frozen inputs and 38 checkpoints found zero mismatches.
- H3 comparator selection completed on train/validation only, 64 independent parents per PDE, selecting B0. Scores: B0 1.0000, B1 1.2018, B2 1.2654, B3 1.2671, B4 1.2066, B5 1.2077. Independent audit passed all 14 checks; the locked test later opened once after input freeze.
- Calibration trajectories and 512 teacher queries/PDE were generated and audited. Burgers calibration queries took 274.68s (0.5365s/query) on CPU NumPy; heat took 1.91s. Nominal forecast calibration locked 26 groups; independent calibration-lock audit passed. Test remains sealed.
- Fresh locked data: 1,280 parents across six role/PDE shards generated in about 15 minutes of CPU-heavy work; locked role/action/numeric audit passed. Six shards and 61 source/config/evidence inputs were frozen while sealed (freeze SHA256 `0feb174dbc7d51634d6f3053da079ac9718eeaf6f8c8885e6a813dab8cd445b2`).
- The authorized one-shot locked test opened at 10:11:26 KST. Two shards are now complete (768/25,600 episodes, 2/120): nominal Burgers B0 took 845.54s (2.2019s/episode), B1 took 2,052.39s (5.3448s/episode). Their method-matched validation pilot rows were 2.0952s and 4.8202s respectively (actual/pilot 1.051x and 1.109x). Adjusting only these two Burgers method-PDE terms gives a provisional total of 13.4176h raw/15.4302h with 15% margin if all other 14 method-PDE rows match pilot; this replaces the misleading flat B0-only projection but remains provisional. The one-shot process is still active; no outcomes were inspected.
- At 11:08 KST the locked-test runner was still active with CPU 3,284s and RAM ~2.44GB; no third shard record yet. The method-weighted projection remains provisional.
- At 11:11 KST the locked-test runner remains active, CPU 3,465s, RAM ~2.47GB; still 768/25,600 episodes and 2/120 shards. No errors; test outputs remain unopened.
- At 11:16 KST the locked-test runner remains active, CPU 3,746s, RAM ~2.52GB; progress still 768/25,600. Method-weighted pilot projection remains provisional pending more actual shard timings.
- The third shard nominal Burgers/P seed11 completed in 1,277.90s/384 (3.3279s/episode, 1.0738x its matched pilot row). Progress is 1,152/25,600 episodes, 3/120 shards. Updating three Burgers method components gives provisional 13.5396h raw/15.5705h with margin (13 of 16 PDE-method rows still pilot-only).
- At 11:28 KST, runner remained active (CPU 4,411s/RAM ~2.43GB); progress is still 1,152/25,600 and no further shard is committed.
- At 11:35 KST, runner remains active (CPU 4,767s/RAM ~2.50GB); progress is still 1,152/25,600 and no errors or terminal state have appeared.
- Runtime evidence confirms optimizer-only fused AdamW gains (row median 8.26%, range 4.28–15.68%) do not address the dominant NumPy/Python Burgers integrations. CUDA can help only after a semantics-preserving batched solver port and audit; no such change was made to the frozen experiment.
- v2 remains terminal/incomplete and cannot be resumed. Its 1,536 partial test executions are not opened or used. Astra's v2 audit and bounded HOLD manuscript are complete; they do not constitute v3 analysis.

## Important unresolved work

- Run the v3 locked test exactly once; after a complete run, aggregate without tuning, then measure host-ready E2E and tail latency, freeze raw evidence, prepare Astra handoff.
- Run the v3 locked test exactly once; after a complete run, aggregate without tuning, then measure host-ready E2E and tail latency, freeze raw evidence, prepare Astra handoff.
- Current runtime model ID for earlier experimental execution is unavailable; do not attribute it to Luna. Reserve 9.596 hours of the original 10-hour Astra allocation for the final v3 evidence handoff.

See RUN_STATE.json, HANDOFF.md, RUN_LEDGER.md, and ../evidence/failures_and_modifications_v3.jsonl.

- 2026-09-27 11:38 KST — User asked whether CUDA optimization can accelerate the slow experiment. Active one-shot remains untouched: runner PID 33280 CPU 5,004.77s, ~2.66GB RAM, GPU sample 6% / 1,853 MiB / 21.94W; progress remains 1,152/25,600 and 3/120 shards at 4,177.03s. This suggests CPU/Python/synchronization may dominate, but no profiler attribution is available. CUDA or implementation changes are deferred until after aggregation and raw freeze; then assess a separate timing-only clone, beginning with batched/vectorized independent rollouts, invariant precomputation, and dispatch/sync reduction, followed by CUDA-specific kernels only where a profiler shows benefit. Require numeric/action equivalence; never mutate this locked run.




- 2026-09-27 11:41 KST — Fourth locked shard committed: nominal Burgers/P seed23, 384 episodes in ~1,230.24s (3.2038s/episode); total 1,536/25,600, 4/120 shards, 5,407.27s elapsed. Method-matched P pilot is 3.0991s/episode; seed23 factor ~1.0338x (seed11 ~1.0738x). Read-only inspection of frozen closed-loop code explains low GPU activity: each Burgers episode executes 200 ticks × 80 CPU burgers_imex_split_step calls in _advance (~16,000 Python solver steps), alongside CUDA policy calls. This justifies testing CPU vectorization/compiled stepping or independent-episode batching first after raw freeze, not changing the current one-shot. CUDA-specific tuning only if separate profiling shows measurable policy/transfer overhead; require numeric/action equivalence.


- 2026-09-27 11:46 KST — Read-only post-test preflight: aggregate_locked_test_v3.py requires complete terminal marker, 120 hash-verified records and all expected method/role/PDE/seed combinations; it verifies action bounds/slew, causal timestamps, state shapes and paired parent order. Registered host E2E performs 120 sequential variant/session rows × 20,000 requests = 2.4M batch-1 host requests, then analyze_latency_v3.py. Estimate runtime after the first completed session row; keep Luna/Astra reserve and do not drop methods. No raw evidence freeze helper currently exists; add a dedicated post-analysis hash-freeze script before declaring evidence frozen. Locked runner still active PID 33280, CPU 5,432.31s, 2.53GB RAM; progress 1,536/25,600 and 4/120 shards at 5,407.27s; stderr empty. No outcomes opened.



- 2026-09-27 11:48 KST — One-shot still active at PID 33280; CPU 5,531.28s, RAM 2.54GB, GPU sample 0% / 1,857 MiB / 20.79W. Progress remains 1,536/25,600 and 4/120 records, cumulative committed elapsed 5,407.27s; stderr empty. This is a verified live wait; current shard has not committed. Budget snapshot updated: 36.75h elapsed, 25.25h to Luna evidence cutoff. No restart or outcome read.



- 2026-09-27 11:52 KST — Confirmed same one-shot PID 33280 remains active; CPU 5,771.33s, RAM ~2.59GB, GPU 7%, 1,857 MiB, 24.99W. Progress remains 1,536/25,600 (4/120 shards, 5,407.27s committed elapsed), stderr empty. Current shard is still running; no restart or outcomes accessed. Prepared post-test raw-freeze utility remains gated. Budget snapshot: 36.80h elapsed, 25.20h until Luna deadline.


- 2026-09-27 11:55 KST — Runner PID 33280 remains live; CPU 5,933.19s, RAM ~2.62GB. Progress 1,536/25,600 and 4/120 shards; committed elapsed 5,407.27s; GPU sample 7% / 1,857 MiB / 15.29W, stderr empty. CPU accumulation continues while the next shard is uncommitted; no failure evidence and no restart. Luna budget snapshot: 25.15h remaining.


- 2026-09-27 11:57 KST — One-shot runner PID 33280 is still active; CPU 6,032.25s, RAM ~2.63GB, GPU sample 7% / 1,857 MiB / 25.07W. Safe progress unchanged at 1,536/25,600 episodes and 4/120 shards; committed elapsed 5,407.27s, stderr empty. No terminal failure, restart, or outcome access. Budget snapshot: 36.88h elapsed, 25.12h until Luna deadline.

- 2026-09-27 11:58 KST — Locked test PID 33280 still live; CPU 6,106.55s, RAM ~2.65GB. Progress remains 1,536/25,600, 4/120 shards, 5,407.27s committed elapsed; stderr empty. Current shard has run since last record commit at 11:41:51, with CPU activity continuing. No terminal marker, failure, restart, or outcome read. Budget snapshot 25.10h to Luna evidence deadline.



- 2026-09-27 12:00 KST — Re-polled active locked-test PID 33280; CPU 6,206.89s, RAM ~2.67GB, GPU sample 8% / 1,857 MiB / 22.65W. Progress remains 1,536/25,600 and 4/120 shards, 5,407.27s committed elapsed; stderr empty and marker nonterminal. CPU is increasing; the in-flight shard remains active. No restart or metric access. Luna evidence cutoff has 25.07h remaining.


- 2026-09-27 12:02 KST — Runner PID 33280 live with CPU 6,286.73s and RAM ~2.68GB; GPU sample 7% / 1,857 MiB / 16.10W. Progress remains 1,536 episodes / 4 shards and stderr is empty. Marker remains in progress, CPU time increased; no restart or outcome access. Luna evidence cutoff in 25.05h.


- 2026-09-27 12:03 KST — Fifth locked shard complete: nominal Burgers/P seed37, 384 episodes / 1,282.01s (3.3386s per episode). Progress 1,920/25,600, 5/120; cumulative elapsed 6,689.40s. P Burgers seed records: s11 1,277.90s, s23 1,230.13s, s37 1,282.01s. Recomputed measured-component pilot projection: B0 actual/pilot 2.2019/2.0952s per episode; B1 5.3448/4.8202; P three-seed mean 3.2900/3.0991. Total estimated full locked test 13.5194h raw / 15.5473h with 15% margin, holding all remaining method-PDE components at pilot. This remains a runtime forecast only; no outcome metrics inspected. Luna budget 25.03h.


- 2026-09-27 12:07 KST — Runner PID 33280 still active; CPU 6,583.80s, RAM ~2.53GB; no stderr. Progress remains 1,920/25,600 and 5/120 shards (6,689.40s committed elapsed). Current shard remains in flight with CPU time increasing. Forecast remains 13.5194h raw /15.5473h with margin, conditional on unmeasured rows matching pilot. Budget snapshot: 24.97h until Luna cutoff.


- 2026-09-27 12:08 KST — Verified same one-shot PID 33280 active; CPU 6,665.00s, RAM ~2.54GB; GPU 8% / 1,857 MiB / 19.56W. Progress is 1,920/25,600 episodes and 5/120 shards; cumulative committed elapsed 6,689.40s; stderr empty. In-flight shard remains uncommitted but CPU advances. Updated Luna budget to 24.95h; no restart or outcomes opened.


- 2026-09-27 12:11 KST — PID 33280 remains live with CPU 6,825.02s / RAM ~2.57GB. Progress remains 1,920/25,600 and 5/120; committed elapsed 6,689.40s; stderr empty. GPU sample 6% / 1,857 MiB / 21.86W. CPU continues accumulating; current shard remains uncommitted. No restart or outcome access. Budget: 24.90h to Luna deadline.


- 2026-09-27 12:14 KST — PID 33280 remains active at CPU 6,977.72s / RAM ~2.60GB; 1,920/25,600 episodes, 5/120 shards remain committed. Stderr empty and marker in progress. Latest RTX sample 2%, 1,900 MiB, 43.61W; utilization is variable and does not change next action. No raw outcomes opened or restart. Luna budget 24.87h.


- 2026-09-27 12:17 KST — Same one-shot runner PID 33280 is live at CPU 7,137.48s / RAM ~2.62GB. Progress 1,920 episodes / 5 shards and 6,689.40s committed elapsed; stderr empty. GPU utilization varied (2% latest; 21% earlier), process CPU time increases. The current shard remains in flight, with no failure/terminal marker. No restart or outcome read; Luna deadline in 24.78h.


- 2026-09-27 12:19 KST — PID 33280 remains live with CPU 7,278.67s and RAM ~2.65GB; progress is 1,920/25,600, 5/120, 6,689.40s committed elapsed. Stderr empty; CPU advances during in-flight shard. No failure/terminal marker, no restart, no outcome reads. Budget snapshot 24.74h to Luna cutoff.


- 2026-09-27 12:22 KST — Active locked-test PID 33280 CPU 7,427.59s / RAM ~2.67GB; progress 1,920/25,600, 5/120, committed elapsed 6,689.40s. Stderr empty and process live. Current shard has been uncommitted ~18.6m since 12:03:41; longer than several earlier shards, but process CPU accumulates and there is no terminal failure. Do not restart or access outcome arrays. Luna budget 24.70h.


- 2026-09-27 12:24 KST — Sixth shard completed: nominal Burgers/P-no-rank seed11, 384 episodes / 1,272.73s (3.3144s/episode), pilot 2.8419s/episode (1.1663x). Progress 2,304/25,600, 6/120, elapsed 7,962.24s. Replacing this three-seed-weighted method/PDE pilot component updates provisional full runtime to 13.7714h raw /15.8371h with 15% margin, assuming remaining unmeasured rows match pilot. This is runtime-only. Runner active, stderr empty; no outcomes read. Luna budget 24.61h.


- 2026-09-27 12:30 KST — One-shot PID 33280 still live; CPU 7,854.97s / RAM ~2.53GB; progress 2,304/25,600, 6/120, 7,962.24s committed elapsed. Stderr empty. Current shard remains uncommitted ~5m after last record, CPU increasing; no failure or terminal state. Forecast 13.7714h raw /15.8371h margin remains provisional. Luna deadline in 24.52h.


- 2026-09-27 12:34 KST — Locked runner PID 33280 remains active; CPU 8,072.94s / RAM ~2.57GB; 2,304/25,600 episodes, 6/120 shards, committed elapsed 7,962.24s. Stderr empty; GPU 7% / 1,884 MiB / 21.78W. Current shard uncommitted ~9m since last record, with CPU time increasing; no terminal failure or restart. No outcomes inspected. Luna budget 24.47h.


- 2026-09-27 12:36 KST — Runner PID 33280 remains live; CPU 8,227.86s / RAM ~2.60GB; 2,304/25,600 episodes, 6/120 shards, 7,962.24s committed elapsed; stderr empty. GPU samples fluctuate (10% latest, earlier brief 50%) without terminal issue. Current shard uncommitted ~12m; CPU increments. No restart or outcome inspection. Luna deadline in 24.43h.


- 2026-09-27 12:39 KST — Runner PID 33280 active at CPU 8,377.52s / RAM ~2.63GB. Progress 2,304 episodes / 6 shards, 7,962.24s committed; stderr empty. GPU sample 9% but variable. Current shard not committed; process CPU accumulates. No failure/terminal state, no restart or outcomes opened. Luna budget 24.38h.


- 2026-09-27 12:42 KST — Runner PID 33280 remains active; CPU 8,530.69s / RAM ~2.65GB; six shards/2,304 episodes remain committed; 7,962.24s cumulative completed-shard elapsed, stderr empty. GPU sample varies; process CPU keeps increasing, no terminal failure. Preserve one-shot; no outcome inspection or restart. Luna cutoff in 24.33h.


- 2026-09-27 12:46 KST — Runner PID 33280 live at CPU 8,736.41s / RAM ~2.69GB. Progress remains 6 shards / 2,304 episodes, committed elapsed 7,962.24s; stderr empty. Current shard in flight ~21.1m, comparable to last P-no-rank shard; CPU continues. No failure marker, no restart, no outcome access. Luna budget 24.27h.


- 2026-09-27 12:47 KST — Seventh shard complete: nominal Burgers/P-no-rank seed23, 384 episodes / 1,323.48s (3.4466s/episode); seed11 1,272.73s. Progress 2,688/25,600, 7/120, elapsed 9,285.83s. Two-seed P-no-rank mean 3.3805s/episode vs pilot 2.8419 (1.1895x); revised provisional full test estimate 13.8066h raw /15.8776h with 15% margin, assuming other unmeasured rows follow pilot. Runtime only, no outcome analysis. Luna budget 24.25h.


- 2026-09-27 12:51 KST — PID 33280 is still live; CPU 9,045.52s / RAM ~2.53GB; progress 2,688 episodes / 7 shards, 9,285.83s completed elapsed, stderr empty. GPU sample 10% / 1,891 MiB /26.46W. Current shard remains in flight; CPU time increases. Provisional projection 13.8066h raw /15.8776h margin-adjusted based on current actual Burgers components, other rows held at pilot. No restart or outcome access; Luna cutoff 24.18h.


- 2026-09-27 12:55 KST — Runner PID 33280 remains active with CPU 9,265.25s / RAM ~2.57GB; progress 2,688 episodes / 7 shards, 9,285.83s committed; stderr empty. GPU 7% / 1,879 MiB /22.75W. Current shard is still in flight since 12:47:29; CPU increments, no terminal state. No restart or outcomes opened. Luna budget 24.12h.


- 2026-09-27 12:58 KST — Locked runner PID 33280 still active; CPU 9,428.36s / RAM ~2.60GB; 2,688/25,600 episodes, 7/120 shards; committed elapsed 9,285.83s. Stderr empty. Current shard remains uncommitted ~11m; CPU increments; no terminal failure, restart, or outcome access. Luna budget 24.07h.


- 2026-09-27 13:00 KST — Runner PID 33280 still active; CPU 9,574.70s / RAM ~2.63GB. Progress unchanged at 2,688 episodes/7 shards, 9,285.83s completed elapsed; stderr empty. Current shard uncommitted since 12:47:29 with CPU increasing; no failure/terminal status. No restart/outcome access. Luna time 24.03h.


- 2026-09-27 13:03 KST — Live runner PID 33280 CPU 9,731.58s / RAM ~2.66GB; progress 2,688/25,600, 7/120 shards, 9,285.83s committed elapsed, stderr empty. Current shard in flight ~16m; CPU increments; no terminal failure or restart. No outcomes opened. Luna cutoff in 23.98h.


- 2026-09-27 13:06 KST — Runner PID 33280 active at CPU 9,878.19s / RAM ~2.69GB; progress 2,688 episodes/7 shards; committed elapsed 9,285.83s; stderr empty. Current shard in flight ~18.75m, CPU time accumulates. No terminal error; do not restart or read outcomes. Luna deadline 23.93h.


- 2026-09-27 13:07 KST — Eighth shard complete: Burgers/P-no-rank seed37, 384 episodes / 1,233.78s (3.2130s/episode). Progress 3,072/25,600, 8/120, elapsed 10,519.73s. P-no-rank Burgers three-seed mean 3.32464s/episode vs 2.84186 pilot (1.1699x). Updating projection from all three seeds yields 13.77685h raw /15.84338h with 15% margin, other unmeasured method-PDE rows at pilot. This remains runtime-only. A read-only metadata command had a malformed Join-Path invocation; corrected immediately, no experiment process or files affected. Runner active, stderr empty, no outcome access. Luna budget 23.91h.


- 2026-09-27 13:12 KST — Locked test PID 33280 live; CPU 10,219.44s / RAM ~2.54GB; progress 3,072/25,600 episodes and 8/120 shards, 10,519.73s committed elapsed; stderr empty. Current shard is still in flight since 13:07, CPU increments. No failure or terminal marker, no restart or outcome access. Forecast remains 13.7769h raw /15.8434h with margin, conditional. Luna cutoff in 23.83h.


- 2026-09-27 13:15 KST — Locked test PID 33280 remains active; CPU 10,388.73s / RAM ~2.57GB; progress 3,072/25,600 and 8/120 shards; committed elapsed 10,519.73s, stderr empty. Latest GPU snapshot 7%, 1,879 MiB, 23W. Current shard continues without record; no terminal state or failure. No restart or outcomes opened. Luna budget 23.78h.


- 2026-09-27 13:18 KST — Runner PID 33280 active, CPU 10,544.56s / RAM ~2.60GB; progress 3,072/25,600, 8/120, completed elapsed 10,519.73s; stderr empty. Current shard continues; no terminal failure or outcome access. One telemetry command failed to spawn due transient working-directory error; immediate retry from default location succeeded with same live PID/progress. Updated Luna budget to 23.73h.


- 2026-09-27 13:21 KST — PID 33280 remains active; CPU 10,712.36s / RAM ~2.63GB; progress 3,072 episodes/8 shards, committed elapsed 10,519.73s, stderr empty. Current shard has not committed since 13:07, but CPU time advances. Latest GPU sample 1% / 1,879 MiB /21.58W; no terminal condition. No restart/outcome access. Luna deadline in 23.68h.


- 2026-09-27 13:26 KST — Active runner PID 33280 CPU 11,002.14s / RAM ~2.68GB; progress unchanged at 3,072/25,600 episodes and 8/120 shards; committed elapsed 10,519.73s. Stderr empty. Current shard remains uncommitted ~19m since 13:07, CPU continues; GPU snapshot varies. No failure/terminal status, no restart, no outcomes opened. Luna cutoff in 23.60h.


- 2026-09-27 13:27 KST — Ninth shard complete: nominal Burgers/B4 seed11, 384 episodes / 1,226.26s (3.1934s/episode) vs validation pilot 2.8581s (1.1173x). Progress 3,456/25,600, 9/120, 11,746.12s. Updated pilot-adjusted runtime to 13.95569h raw /16.04904h with 15% margin, replacing this three-seed-weighted B4 Burgers component with seed11 actual and retaining pilot values for all other unmeasured rows. Runtime estimate only. Runner active; no errors/outcome access. Luna budget 23.58h.


- 2026-09-27 13:31 KST — Runner PID 33280 remains live at CPU 11,301.86s / RAM ~2.53GB; progress 3,456 episodes, 9/120 shards, 11,746.12s completed elapsed; stderr empty. GPU sample 6% /1,879 MiB/21.78W. Current shard is uncommitted while CPU continues. No terminal failure, restart, or outcome access. Luna budget 23.51h.


- 2026-09-27 13:35 KST — Locked runner PID 33280 remains active at CPU 11,515.38s / RAM ~2.56GB; progress 3,456 episodes/9 shards, completed elapsed 11,746.12s, stderr empty. GPU sample 7% /1,885 MiB/22.85W. Current shard uncommitted; process CPU grows. No terminal error/restart/outcome access. Luna deadline in 23.45h.


- 2026-09-27 13:37 KST — PID 33280 remains active at CPU 11,664.39s / RAM ~2.59GB; progress 3,456 episodes / 9 shards, completed elapsed 11,746.12s, stderr empty. Current shard has not committed yet, CPU continues. No terminal failure, restart, or outcome access. Conditional full-run estimate stays 13.9557h raw /16.0490h with margin; Luna budget 23.41h.


- 2026-09-27 13:40 KST — PID 33280 active, CPU 11,827.80s / RAM ~2.62GB; progress still 3,456 episodes /9 shards /11,746.12s committed elapsed; stderr empty. Current shard remains in flight since 13:27:29 with CPU activity. GPU snapshot variable; no terminal failure or restart, no outcome access. Luna budget 23.36h.


- 2026-09-27 13:44 KST — Runner PID 33280 live at CPU 12,037.94s / RAM ~2.66GB; progress 3,456 episodes / 9 shards / 11,746.12s committed; stderr empty. Current shard uncommitted ~17m, process CPU increasing. GPU sample variable (8% /1,885 MiB/22.52W); no terminal error. No restart or result-array inspection. Luna budget 23.29h.


- 2026-09-27 13:48 KST — Tenth shard complete: nominal Burgers/B4 seed23, 384 episodes /1,222.97s; seed11=1,226.26s. Progress 3,840/25,600, 10/120, elapsed 12,969.20s. B4 two-seed mean=3.18909s/episode vs pilot=2.85807 (1.11582x). Updating B4 Burgers correction from seed11-only to two-seed mean revises full runtime to 13.95340h raw /16.04641h with margin, assuming remaining rows match pilot. Runtime only; runner active, no error/outcome analysis. Luna budget 23.22h.


- 2026-09-27 13:53 KST — PID 33280 remains live at CPU 12,548.58s / RAM ~2.54GB; progress 3,840 episodes/10 shards, committed elapsed 12,969.20s; stderr empty. Current shard has not committed since 13:48, CPU continues to accumulate. GPU snapshot 6% /1,885 MiB/15.41W; no terminal marker or failure. No restart/outcome inspection. Luna budget 23.13h.


- 2026-09-27 13:57 KST — Runner PID 33280 active; CPU 12,763.83s / RAM ~2.58GB; progress remains 3,840 episodes /10 shards; committed elapsed 12,969.20s, stderr empty. Current shard uncommitted since 13:48:10 (~9m), CPU increases. GPU 9% /1,885 MiB/26.02W. No terminal failure or restart/outcome access. Luna budget 23.06h.


- 2026-09-27 14:01 KST — Active runner PID 33280; CPU 13,003.06s / RAM ~2.62GB; progress remains 3,840 episodes/10 shards/12,969.20s; stderr empty. Current shard uncommitted since 13:48:10 with CPU accumulation. No terminal marker/failure, no restart, no outcome access. Luna evidence window remaining 22.99h.


- 2026-09-27 14:05 KST — Runner PID 33280 remains active at CPU 13,228.91s / RAM ~2.66GB; progress 3,840 episodes/10 shards/12,969.20s; stderr empty. Current shard remains uncommitted since 13:48:10, CPU increments; GPU sample 0% but other samples are variable. No failure or terminal marker, no restart or outcome access. Luna cutoff 22.92h.


- 2026-09-27 14:08 KST — Eleventh shard complete: nominal Burgers/B4 seed37, 384 episodes /1,219.05s. Progress 4,224/25,600, 11/120, 14,188.37s. B4 Burgers three-seed times: 1,226.26, 1,222.97, 1,219.05s; mean 3.18427s/episode vs 2.85807 pilot (1.11413x). Three-seed projection yields 13.95082h raw /16.04345h with 15% margin, holding all unmeasured rows at pilot. Runtime only. Runner active, stderr empty; no outcome analysis. Luna budget 22.87h.


- 2026-09-27 14:11 KST — Runner PID 33280 active at CPU 13,562.72s / RAM ~2.52GB; progress remains 4,224 episodes /11 shards /14,188.37s; stderr empty. Current shard uncommitted since 14:08:17, CPU continues. GPU telemetry variable, no terminal issue. No restart or outcome access. Luna budget 22.82h.


- 2026-09-27 14:15 KST — PID 33280 remains active; CPU 13,773.22s / RAM ~2.55GB; progress 4,224 episodes /11 shards, 14,188.37s committed; stderr empty. GPU 7% /1,885 MiB/18.43W. Current shard uncommitted since 14:08:17; CPU activity continues. No failure/terminal state, no restart or outcome access. Luna budget 22.76h.


- 2026-09-27 14:19 KST — Runner PID 33280 active at CPU 14,052.05s / RAM ~2.60GB; progress 4,224 episodes/11 shards/14,188.37s committed; stderr empty. Current shard uncommitted since 14:08:17, CPU still advances. GPU sample variable; no failure/terminal marker. No restart or outcome inspection. Luna budget 22.68h.


- 2026-09-27 14:23 KST — PID 33280 active; CPU 14,273.33s / RAM ~2.64GB; progress remains 4,224 episodes/11 shards/14,188.37s completed; stderr empty. Current shard has not committed since 14:08:17, CPU grows. GPU sample 7% /1,885 MiB/23.01W; no terminal marker/failure. No restart/outcome access. Luna cutoff in 22.61h.


- 2026-09-27 14:26 KST — Runner PID 33280 active at CPU 14,420.08s / RAM ~2.67GB; progress remains 4,224 episodes/11 shards/14,188.37s completed; stderr empty. Current shard uncommitted since 14:08:17 with CPU accumulation; no terminal failure/restart/outcome access. GPU telemetry varies. Luna deadline in 22.57h.


- 2026-09-27 14:29 KST — Twelfth shard complete: nominal Burgers/B5 seed11, 384 episodes /1,224.33s (3.18836s/episode) vs pilot 3.00993 (1.05928x). Progress 4,608/25,600, 12/120, elapsed 15,412.82s. Replacing the three-seed-weighted B5 Burgers row pilot component gives provisional 14.04599h raw /16.15288h with 15% margin, assuming other unmeasured rows match pilot. Runtime-only estimate; runner active, stderr empty, outcomes unopened. Luna budget 22.53h.


- 2026-09-27 14:33 KST — Runner PID 33280 live; CPU 14,795.94s / RAM ~2.56GB; progress 4,608 episodes/12 shards/15,412.82s committed; stderr empty. Current shard uncommitted since 14:29:01, CPU increments. GPU sample 7% /1,878 MiB/21.93W. No terminal marker/failure or restart; outcomes unopened. Luna budget 22.46h.


- 2026-09-27 14:36 KST — PID 33280 remains active; CPU 15,013.44s / RAM ~2.57GB; progress 4,608 episodes/12 shards/15,412.82s committed, stderr empty. Current shard uncommitted since 14:29:01 and CPU keeps increasing. GPU samples fluctuate (0% latest); no terminal issue. No restart/outcome access. Luna window 22.40h.


- 2026-09-27 14:39 KST — Runner PID 33280 active; CPU 15,162.12s / RAM ~2.60GB; progress unchanged at 4,608 episodes/12 shards/15,412.82s; stderr empty. Current shard uncommitted since 14:29:01 with CPU accumulation. GPU telemetry variable; no terminal failure, restart, or outcome access. Luna budget 22.35h.


- 2026-09-27 14:43 KST — PID 33280 remains active; CPU 15,376.94s / RAM ~2.64GB; 4,608 episodes /12 shards /15,412.82s committed elapsed; stderr empty. Current shard uncommitted since 14:29:01 but CPU advances. GPU 6% /1,878 MiB/21.88W. No terminal status/failure, no restart/outcome access. Luna budget 22.28h.


- 2026-09-27 14:45 KST — PID 33280 active with CPU 15,526.09s / RAM ~2.66GB; progress 4,608 episodes /12 shards /15,412.82s committed; stderr empty. Current shard remains uncommitted ~16.8m after 14:29:01 but CPU accumulates. GPU 0% latest (sample variability); no terminal failure or restart/outcome access. Luna budget 22.23h.


- 2026-09-27 14:48 KST — Thirteenth shard complete: nominal Burgers/B5 seed23, 384 episodes /1,202.51s (3.1315s/episode); seed11=1,224.33s. Progress 4,992/25,600, 13/120, elapsed 16,615.45s. B5 two-seed mean 3.15995s/episode vs pilot 3.00993 (1.04984x); updating prior one-seed correction revises provisional total to 14.03083h raw /16.13546h with 15% margin, remaining rows assumed pilot. Runtime only. Runner active, stderr empty, outcomes unopened. Luna budget 22.18h.

- 2026-09-27 14:51:43 KST — Revalidated active locked runner PID 33280 (CPU 15,862.98s, RAM ~2.52GB). Progress remains 4,992/25,600 episodes and 13/120 shards; committed elapsed 16,615.45s. CPU advanced since 14:48; next shard is still in progress, no terminal failure or restart. No outcome arrays inspected. Last instantaneous GPU sample 7% /1,878 MiB /21.76W is not a profile. Updated remaining wall budget to 32.13h and Luna evidence window to 22.13h.

- 2026-09-27 14:52:53 KST — Verified runner PID 33280 still active (CPU 15,929.61s, RAM ~2.53GB). No new commit since 14:48:22: still 4,992 episodes/13 shards/16,615.45s. CPU increased by about 68s since 14:51:43; current shard continues, with no terminal evidence. No outcome arrays inspected. Updated budget snapshot: 39.89h elapsed, 32.11h total remaining, 22.11h before Luna evidence deadline.

- 2026-09-27 14:54:48 KST — Rechecked the same one-shot PID 33280 after a bounded observation; process remains active (CPU 16,038.38s, RAM ~2.55GB). Progress remains 4,992 episodes/13 shards/16,615.45s, latest record at 14:48:22. CPU time advanced about 65s since 14:53:39; no terminal failure or restart. No outcome arrays inspected. Budget snapshot: 39.92h elapsed, 32.08h total remaining, 22.08h before Luna evidence deadline.

- 2026-09-27 14:56:55 KST — Runner PID 33280 remains active (CPU 16,158.98s, RAM ~2.57GB); committed progress is 4,992 episodes/13 shards/16,615.45s. Latest record B5_s23 at 14:48:22; CPU advanced about 80s over ~85s observed. Current shard elapsed since last commit is ~8.5m, shorter than prior ~20m Burgers shards. No failure/restart evidence; outcomes unopened. Budget snapshot: 39.96h elapsed, 32.04h total remaining, 22.04h before Luna evidence deadline.

- 2026-09-27 14:58:36 KST — Verified runner PID 33280 active (CPU 16,254.48s, RAM ~2.58GB). Progress remains 4,992 episodes/13 shards/16,615.45s; latest record B5_s23 at 14:48:22. Current uncommitted shard duration ~10m14s, within prior ~20m Burgers shard times; CPU advanced, no terminal failure/restart. No outcome arrays inspected. Budget snapshot: 39.99h elapsed, 32.01h total remaining, 22.01h before Luna evidence deadline.

- 2026-09-27 15:00:21 KST — Runner PID 33280 active (CPU 16,354.09s, RAM ~2.60GB). Progress remains 4,992 episodes/13 shards/16,615.45s; latest record B5_s23 at 14:48:22. Current shard is ~12m uncommitted, CPU continues to rise; this remains within the observed ~20m Burgers shard duration. No terminal failure/restart or outcome access. Budget: 40.02h elapsed, 31.98h total remaining, 21.98h before Luna deadline.

- 2026-09-27 15:02:00 KST — Runner PID 33280 remains active (CPU 16,448.97s, RAM ~2.62GB). Progress remains 4,992 episodes/13 shards/16,615.45s. Latest record B5_s23 at 14:48:22; current shard uncommitted ~13m38s, CPU advances, within observed shard duration. No terminal failure/restart and no outcome access. Budget snapshot: 40.05h elapsed, 31.95h total remaining, 21.95h before Luna deadline.

- 2026-09-27 15:05:13 KST — Runner PID 33280 active (CPU 16,628.17s; RAM ~2.65GB), progress 4,992/25,600 and 13/120 shards; latest B5_s23 record 14:48:22. Current shard uncommitted ~16m51s and CPU advances; no failure/restart/outcome access. Read-only inspection of benchmark_host_e2e_v3.py confirms 40 (PDE, method, seed) variants ×3 sequential sessions=120 rows, 20,000 batch-1 requests per row (2.4M total), 50 warmups per row, per-request full wall-clock timing plus CUDA synchronization for stream timing. The script gates on full locked completion; budget forecast must be revised after the first completed E2E row. Budget snapshot: 40.10h elapsed, 31.90h remaining total, 21.90h to Luna deadline.

- 2026-09-27 15:08:59 KST — Fourteenth shard complete: nominal Burgers/B5 seed37, 384 episodes /1,224.973804s (3.1900s/episode), SHA256 2c9d3464f2d118a1ead342b386724ea8bfd58b4c5172df967d4c2a63db0878ca. Progress 5,376/25,600, 14/120, 17,840.543751s. B5 three-seed mean=1,217.2713s/shard (3.16998s/episode), 1.0482x pilot. Component-only run projection updates to 14.03618h raw /16.14161h with 15% margin, other unmeasured cells left at pilot. Timing only; no metric arrays opened. Runner PID 33280 remains active. Budget: 40.16h elapsed, 31.84h total remaining, 21.84h until Luna deadline.

- 2026-09-27 15:11:33 KST — Runner PID 33280 active (CPU 16,992.42s, RAM ~2.60GB). Progress unchanged at 5,376/25,600 episodes and 14/120 shards; last record nominal Burgers/B5 seed37 at 15:08:47. CPU advanced ~73s across ~55s observation interval; no failure/restart evidence, outcome arrays unopened. Provisional component-updated forecast remains 14.03618h raw /16.14161h with 15% margin. Budget: 40.20h elapsed, 31.80h total remaining, 21.80h to Luna deadline.

- 2026-09-27 15:14:14 KST — Runner PID 33280 active (CPU 17,148.09s, RAM ~2.61GB); progress remains 5,376 episodes/14 shards/17,840.543751s. Latest record B5_s37 at 15:08:47. CPU advanced across observations; marker remains in-progress; no terminal failure/restart evidence and no outcome access. Budget snapshot: 40.25h elapsed, 31.75h total remaining, 21.75h until Luna deadline.

- 2026-09-27 15:16:55 KST — Runner PID 33280 active (CPU 17,304.39s, RAM ~2.64GB). Progress remains 5,376 episodes/14 shards/17,840.543751s; latest record B5_s37 at 15:08:47. Next shard is uncommitted for ~8m08s, CPU advances; no terminal failure/restart. No outcome arrays inspected. Budget: 40.29h elapsed, 31.71h total remaining, 21.71h to Luna deadline.

- 2026-09-27 15:20:40 KST — Runner PID 33280 active (CPU 17,522.55s, RAM ~2.68GB); progress remains 5,376 episodes/14 shards/17,840.543751s. Latest record B5_s37 at 15:08:47; next shard uncommitted ~11m53s, CPU continued advancing across polls. No terminal failure/restart evidence; raw outcomes unopened. Budget snapshot: 40.35h elapsed, 31.65h total remaining, 21.65h before Luna deadline.

- 2026-09-27 15:23:27 KST — Runner PID 33280 active (CPU 17,683.64s, RAM ~2.72GB). Progress remains 5,376 episodes/14 shards/17,840.543751s; latest record B5_s37 at 15:08:47. Next shard uncommitted ~14m40s; CPU advanced over successive polls. No terminal failure/restart or outcome access. Budget snapshot: 40.40h elapsed, 31.60h total remaining, 21.60h until Luna evidence deadline.

- 2026-09-27 15:26:07 KST — Fifteenth shard complete: nominal Burgers/B2 seed11, 384 episodes /1,031.089909s, SHA256 95522e672f475f3cd2404351412fd5bdc551e1a6e807e9f5c1847274991c975a. Progress 5,760/25,600, 15/120, 18,871.732679s. B2 pilot 2.599170s/episode predicts 998.081s/shard; actual=2.685130s/episode, +33.0086s (1.03307x). Reusing the registered prior component correction factor 5 updates provisional total to 14.08202h raw /16.19433h with 15% margin; only nominal Burgers B2/B4/B5 components are measured, all other method/role/PDE cells remain pilot-based. Runtime only; no outcome arrays inspected. Runner remains active. Budget: 40.44h elapsed, 31.56h total remaining, 21.56h to Luna deadline.

- 2026-09-27 15:31:39 KST — Runner PID 33280 active (CPU 18,161.59s, RAM ~2.64GB); progress remains 5,760 episodes/15 shards/18,871.732679s. Latest record B2_s11 at 15:25:58; next shard uncommitted ~5m41s. CPU advanced across observations; no terminal failure/restart evidence, no outcomes opened. Budget: 40.53h elapsed, 31.47h total remaining, 21.47h to Luna deadline.

- 2026-09-27 15:37:35 KST — Runner PID 33280 active (CPU 18,507.39s, RAM ~2.68GB). Progress remains 5,760 episodes/15 shards/18,871.732679s; latest record B2_s11 at 15:25:58. Next shard uncommitted ~11m37s and CPU advances across polls. No terminal failure/restart evidence; no outcome arrays accessed. Budget: 40.63h elapsed, 31.37h total remaining, 21.37h to Luna deadline.

- 2026-09-27 15:43:26 KST — Sixteenth shard complete: nominal Burgers/B2 seed23, 384 episodes /1,030.148553s, SHA256 bcdaf2c955cecfec66812995f58156f6a3fae035d96e31135c9cf3043f583628. Progress 6,144/25,600, 16/120, 19,901.984443s. B2 two-seed mean=1,030.6192s/shard (2.683904s/episode), 1.03260x pilot (998.0813s/shard). Reusing the prior factor-5 component correction updates provisional total to 14.08137h raw /16.19358h with 15% margin. Only nominal Burgers B2/B4/B5 components measured; others remain pilot-based. Runtime only, no outcomes opened. Budget: 40.73h elapsed, 31.27h total remaining, 21.27h to Luna deadline.

- 2026-09-27 15:48:33 KST — Runner PID 33280 active (CPU 19,147.67s, RAM ~2.61GB). Progress remains 6,144 episodes/16 shards/19,901.984443s. Latest record B2_s23 at 15:43:08; next shard uncommitted ~5m25s, CPU continues increasing; no terminal failure/restart evidence and outcomes unopened. Runtime projection 14.08137h raw/16.19358h margin-adjusted remains provisional. Budget: 40.82h elapsed, 31.18h total remaining, 21.18h to Luna deadline.

- 2026-09-27 15:53:27 KST — Runner PID 33280 active (CPU 19,433.42s, RAM ~2.66GB). Progress remains 6,144 episodes/16 shards/19,901.984443s; latest record B2_s23 at 15:43:08. Next shard uncommitted ~10m19s; CPU advances despite no new commit. No terminal failure/restart evidence; outcome arrays not accessed. Budget: 40.90h elapsed, 31.10h total remaining, 21.10h before Luna deadline.

- 2026-09-27 15:59:19 KST — Runner PID 33280 active (CPU 19,774.52s, RAM ~2.74GB). Progress remains 6,144 episodes/16 shards/19,901.984443s; latest record B2_s23 at 15:43:08. Next shard uncommitted ~6m11s, CPU continues to increase; no terminal failure/restart evidence and outcome arrays unopened. Runtime forecast 14.08137h raw/16.19358h with margin remains provisional. Budget: 41.00h elapsed, 31.00h total remaining, 21.00h to Luna deadline.

- 2026-09-27 16:00:26 KST — Seventeenth shard complete: nominal Burgers/B2 seed37, 384 episodes /1,037.448527s, SHA256 5ca250611b1d6f365b7427fc8ba54c470493532179e4c11339cc9fd30beff2ad. Progress 6,528/25,600, 17/120, 20,939.529724s. B2 three-seed mean=1,032.8957s/shard (2.689832s/episode), 1.03488x pilot. Updating B2 component from two-seed average gives provisional total 14.08453h raw /16.19721h with 15% margin. Only nominal Burgers B2/B4/B5 measured, all other cells pilot-based. Timing only; outcome arrays unopened. Budget: 40.92h elapsed, 31.08h total remaining, 21.08h to Luna deadline.

- 2026-09-27 16:07:11 KST — Runner PID 33280 active (CPU 20,234.03s, RAM ~2.63GB). Progress remains 6,528 episodes/17 shards/20,939.529724s; latest record B2_s37 at 16:00:26. Next shard uncommitted ~6m45s; CPU continues to increase. No terminal failure/restart evidence; outcomes unopened. Provisional runtime estimate 14.08453h raw/16.19721h margin-adjusted. Budget: 41.13h elapsed, 30.87h total remaining, 20.87h to Luna deadline.

- 2026-09-27 16:15:14 KST — Runner PID 33280 active (CPU 20,702.88s, RAM ~2.72GB). Progress remains 6,528 episodes/17 shards/20,939.529724s; latest record B2_s37 at 16:00:26. Next shard uncommitted ~8m48s, CPU advances; marker still in-progress with no failure/restart evidence. No outcomes accessed. Budget: 41.26h elapsed, 30.74h total remaining, 20.74h before Luna deadline.

- 2026-09-27 16:17:31 KST — Eighteenth shard complete: nominal Burgers/B3 seed11, 384 episodes /1,025.162443s, SHA256 733ec370ca25793fc6d63d3f02d1f352219f495e38c1305b1ccb0fdc80040499. Progress 6,912/25,600, 18/120, 21,964.790752s. B3 pilot 2.624067s/episode predicts 1,007.642s/shard; actual 2.669694s/episode, +17.5206s (1.01739x). Reusing factor-5 component correction yields provisional total 14.10887h raw /16.22520h with 15% margin. Four measured PDE-method pairs are nominal Burgers B2/B3/B4/B5 only; all other method/role/PDE cells remain pilot-based. Timing only; no outcomes opened. Budget: 41.20h elapsed, 30.80h total remaining, 20.80h to Luna deadline.

- 2026-09-27 16:24:15 KST — Runner PID 33280 active (CPU 21,227.33s, RAM ~2.63GB). Progress remains 6,912 episodes/18 shards/21,964.790752s; latest record B3_s11 at 16:17:31. Next shard uncommitted ~6m44s and CPU advances; marker in progress, no terminal failure/restart. Outcome arrays unopened. Runtime estimate 14.10887h raw/16.22520h margin-adjusted remains provisional. Budget: 41.41h elapsed, 30.59h total remaining, 20.59h to Luna deadline.

- 2026-09-27 16:29:07 KST — Runner PID 33280 active (CPU 21,511.88s, RAM ~2.68GB). Progress remains 6,912 episodes/18 shards/21,964.790752s; latest record B3_s11 at 16:17:31. Next shard uncommitted ~6m, CPU advances, no terminal failure/restart evidence; outcomes unopened. Provisional estimate 14.10887h raw/16.22520h with margin. Budget: 41.49h elapsed, 30.51h total remaining, 20.51h to Luna deadline.

- 2026-09-27 16:35:02 KST — Nineteenth shard complete: nominal Burgers/B3 seed23, 384 episodes /1,025.080295s, SHA256 8a26301e30c0a4d21191d8f849fcab4f0cbdfb250ffefec5e82d1ea19771256a. Progress 7,296/25,600, 19/120, 22,989.976080s. B3 two-seed mean=1,025.121369s/shard (2.669587s/episode), 1.01735x pilot (1,007.641843s/shard). Updated provisional projection 14.10881h raw /16.22513h with 15% margin. Four actual pairs remain nominal Burgers B2/B3/B4/B5 only; all others pilot-based. Runtime only; no outcome arrays opened. Budget: 41.59h elapsed, 30.41h total remaining, 20.41h to Luna deadline.

- 2026-09-27 16:41:22 KST — Runner PID 33280 active (CPU 22,222.95s, RAM ~2.63GB). Progress remains 7,296 episodes/19 shards/22,989.976080s; latest record B3_s23 at 16:34:36. Next shard uncommitted ~6m46s; CPU advances; marker in progress; no terminal failure/restart evidence or outcome access. Forecast remains 14.10881h raw /16.22513h with margin, provisional. Budget: 41.69h elapsed, 30.31h total remaining, 20.31h to Luna deadline.

- 2026-09-27 16:48:26 KST — Runner PID 33280 active (CPU 22,623.94s, RAM ~2.69GB). Progress remains 7,296 episodes/19 shards/22,989.976080s; latest record B3_s23 at 16:34:36. Next shard uncommitted ~7m50s, CPU advances; marker remains in progress; no terminal failure/restart evidence and outcomes unopened. Provisional forecast 14.10881h raw/16.22513h margin-adjusted. Budget: 41.81h elapsed, 30.19h total remaining, 20.19h to Luna deadline.

- 2026-09-27 16:54:25 KST — Twentieth shard complete: nominal Burgers/B3 seed37, 384 episodes /1,134.677630s, SHA256 8d33b7c0f8ef71f671665cae4e5444e839844d8cf0a7f0620b675806d532c689. Progress 7,680/25,600, 20/120, 24,124.751997s. B3 three-seed mean=1,061.6401s/shard (2.764688s/episode), 1.05359x pilot. Updating B3 from two-seed mean revises provisional total to 14.15953h raw /16.28346h with 15% margin. Only nominal Burgers B2/B3/B4/B5 are measured; all other role/method/PDE cells remain pilot-based. Timing only; raw outcomes unopened. Budget: 41.91h elapsed, 30.09h total remaining, 20.09h to Luna deadline.

- 2026-09-27 16:58:35 KST — Twenty-first shard complete: locked_coefficient_ood/Burgers/B0, n=128, elapsed 303.973510s, SHA256 e66ef30505a56f796e1f928ea8ba618f4498524ae305716ef2da0232c5942fb1. Progress 7,808/25,600, 21/120, 24,429.071253s. Pilot B0 Burgers 2.095229s/episode predicts 268.1893s for n=128; actual +35.7842s (1.13343x). Applying role-count factor 5 to project this one B0 cell across all three role sizes updates provisional total to 14.20923h raw /16.34061h with 15% margin. This cross-role projection is an assumption, clearly separated from direct OOD timing. Nominal Burgers B2/B3/B4/B5 plus coefficient-OOD Burgers/B0 now have actual metadata; other cells remain pilot-based. Runtime only; outcomes unopened. Budget: 41.98h elapsed, 30.02h total remaining, 20.02h to Luna deadline.

- 2026-09-27 17:04:33 KST — Runner PID 33280 active (CPU 23,562.20s, RAM ~2.42GB). Progress remains 7,808 episodes/21 shards/24,429.071253s; latest record coefficient-OOD/Burgers/B0 at 16:58:35. Next shard uncommitted ~5m58s and CPU increases; no terminal failure/restart evidence and outcomes unopened. Runtime forecast remains provisional 14.20923h raw/16.34061h with margin. Budget: 42.08h elapsed, 29.92h total remaining, 19.92h before Luna deadline.

- 2026-09-27 17:09:38 KST — Runner PID 33280 active (CPU 23,860.00s, RAM ~2.45GB). Progress remains 7,808 episodes/21 shards/24,429.071253s; latest record coefficient-OOD/Burgers/B0 at 16:58:35. Next shard uncommitted ~5m03s with CPU time accumulating; no terminal failure/restart evidence, outcomes unopened. Forecast provisional 14.20923h raw/16.34061h margin-adjusted. Budget: 42.16h elapsed, 29.84h total remaining, 19.84h to Luna deadline.

- 2026-09-27 17:10:16 KST — Twenty-second shard complete: locked_coefficient_ood/Burgers/B1 n=128, elapsed 700.292307s, SHA256 5d310d6dcb7974d557234aceed3b84f46ea81df70911dc41f40e7afb1b0ca0a5. Progress 7,936/25,600, 22/120, 25,129.400133s. Pilot B1 Burgers 4.820245s/episode predicts 616.9913s for n=128; actual +83.3010s (1.13501x). Applying role-size factor 5 updates provisional total to 14.32493h raw /16.47366h with 15% margin; projection assumes this OOD runtime transfers across the other two roles. Direct timing covers six cells: nominal Burgers B2/B3/B4/B5 and coefficient-OOD Burgers B0/B1. Other cells remain pilot-based. Runtime only, no outcome arrays opened. At 17:11:19 runner PID 33280 still active. Budget snapshot: 42.19h elapsed, 29.81h total remaining, 19.81h to Luna deadline.

- 2026-09-27 17:15:45 KST — Runner PID 33280 active (CPU 24,210.23s, RAM ~2.44GB). Progress remains 7,936 episodes/22 shards/25,129.400133s; latest record coefficient-OOD/Burgers/B1 at 17:10:16. Next shard uncommitted ~5m29s; CPU advances; no terminal failure/restart evidence, outcomes unopened. Runtime projection remains provisional 14.32493h raw/16.47366h with 15% margin. Budget: 42.29h elapsed, 29.71h total remaining, 19.71h to Luna deadline.