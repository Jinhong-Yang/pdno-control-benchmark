# v3 failures, repairs and scientific limitations

The executed benchmark supports a bounded negative decision. Administrative completion, correct hashing and passing arithmetic do not change failed scientific gates. The frozen `evidence/failures_and_modifications_v3.jsonl` and original ledgers are preserved. The [complete event index](analysis_20260928/failure_event_index.csv) contains every one of the 138 frozen JSONL records, including its full JSON and one-based source line. A [ledger review snapshot](ledger_review_snapshot.md) preserves the full ledger as read for this analysis; the tables below also cover failures recorded only there.

## Experimental and operational failures and corrections

| Event / source | What happened and disposition | Consequence |
|---|---|---|
| New venv; JSONL 1 | Initial pytest collection failed because torch was absent; official CUDA wheel installed before subsequent testing | No successful test or GPU result attributed to the failed attempt |
| v3 source repair; JSONL 2 | Heat classical sensor coordinates corrected; namespace embedded in parent ID/RNG | Pretest correction; predecessor results not repaired retroactively |
| Boundary derivative; JSONL 4–5 | Removing endpoint mask initially violated strict boundary value tolerance (2.55e-7); linear roundoff correction retained derivative and original tolerance | Initial failed fixture retained; later 45-test receipt, then 47-test receipt, are distinct |
| Signed thermal cost; JSONL 7 | Separate lower/upper heat hinges added throughout teacher/ROM/learned/closed-loop objectives | Earlier two-parent smoke queries became selection-ineligible; preserved |
| Stale teacher fixtures; ledger 06:13 | Signed-bound tests exposed missing bounds and arbitrary fixture costs; fixtures updated to new reproducible contract | 47/47 recorded afterwards; not a scientific quality pass |
| Prior fused-AdamW candidate; JSONL 9 | Earlier optimizer-step pilot showed speed reduction but all accuracy variants failed; not selected | No transfer of that gain to v3 inference, E2E, data generation or time-to-accuracy |
| G2; JSONL 11 and selected manifest | All predictive pilot cases and all 24 selected predictive checkpoints exceed 0.05 nRMSE | G2_FAIL_RETAINED; no threshold relaxation or positive claim |
| Runner static audit; ledger 06:59 | First source-contract audit failed because locked-data audit status was not explicitly checked; guard added before freeze | Subsequent static PASS is limited to those checks |
| Progress flushing; training ledger | Shared progress JSON lagged per-model completions until seed boundary | Completed summaries/checkpoints used; a fresh partial checkpoint alone was not counted complete |
| Read-only inspection; JSONL 14,16 | Malformed Join-Path and unsupported wildcard rg path failed; explicit-path inspection recovered | No experiment mutation |
| Projection correction; JSONL 34 | Early factor-five transfer over-projected role-specific timing differences; replaced by direct 120-row schedule sum | Historical forecasts remain estimates, not measured total time |
| Metadata transcription; JSONL 36 | One duplicated hash character in RUN_STATE corrected from canonical record | No raw result changed |
| Maintenance syntax; JSONL 39,53 | Two state-writing Python expressions failed before mutation; syntax corrected | No runner/protocol change |
| Console encoding; JSONL 42 | JSON validation passed but Unicode printing failed under cp949; ASCII verification used | Not a data-validation failure |
| Timing metadata reconciliation; JSONL 96,109,112 | Already-counted B1 row and two record commit times were corrected in narration | Counts and outcome arrays unaffected |
| Aggregation path; JSONL 113 | Repository-root aggregate script path did not exist; correct experiment path then completed | First command never executed the aggregator |
| Missing training receipt; JSONL 115 | E2E preflight required a missing transient supervisor receipt; receipt reconstructed from persisted progress, summaries/checkpoints, manifest and prior recorded exit | Reconstruction explicitly labeled; not a fresh process observation |
| E2E input root; JSONL 116 | Stale data/train path replaced with frozen data/train_validation_v3 path | Executed source differs from pretest hash; final raw freeze records actual source |
| Replay axes; JSONL 117 | Loader expected flat snapshots; saved shape was parent×20 snapshots | Flattened existing leading axes; corrected snapshot count; no new source data or timing endpoint |
| Repair indentation; JSONL 118 | Python parse-time failure after a patch; indentation repaired | Failed attempt did not execute |
| PowerShell and empty destination; JSONL 119 | A command was printed rather than invoked; later loader failure left an empty directory and overwrite guard rejected retry | Empty attempt directory preserved; successful benchmark has 120 full rows |
| Slow rows; JSONL 127–129,132,137 | B1 Burgers CPU and a heat learned row were slow | Retained without trimming; monitoring interval arithmetic is not a latency quantile |
| Completed E2E; JSONL 138 | 120 rows and 2,400,000 requests completed | Completion is not target-latency success |
| Post-freeze derived CUDA labels | q=.99 and .999 collided at p99 in supplied diagnostic; original p99 values were p99.9 | Original retained, own independent correction and origin-task corrected file supplied; raw/registered analysis unchanged |
| Administrative state drift | RUN_STATE and RUN_LEDGER changed after raw freeze for handoff/status | Entry differences disclosed; cannot report literal agreement of every current administrative byte |

The JSONL uses records for routine progress as well as failures. All 138 are indexed rather than selecting only favorable events. Some progress prose contains stale counts or rough elapsed intervals; final inventories and raw rows control quantitative claims. For example, the early throughput comment at JSONL 130 is not a validated throughput endpoint. The original full ledger snapshot remains available when the table compresses routine updates.

## Scientific limits and unresolved contract discrepancies

| Finding | Evidence / status | What it rules out |
|---|---|---|
| G2 failure | 24 selected operator errors >0.05; pilot failure retained | Claim of adequate predictive gate performance |
| Strong B0 result | Every non-B0 method has higher test mean cost in each group | Proposed control-quality superiority under the implemented comparison |
| H1 target missed | Mean P/B4 p99 ratios 0.9859/0.9930, target ≤0.75; heat interval spans one | Meaningful 25% factorization latency win |
| H3 implemented cost failure | All P cost CI upper endpoints exceed 0.05 | Implemented noninferiority claim |
| H3 estimand differs | Frozen aggregator uses fixed pooled comparator denominator; original instruction calls for floored parent-wise ratios | Full original-protocol H3 compliance; report only executed estimand |
| Checkpoint contract conflict | YAML describes eight-parent closed-loop selection; actual v3 selection is training-summary validation score | Claim that both selection methods were executed |
| Weak H2 evidence | P/B5 identical Burgers actions and nearly identical heat results | Established physical-loss control benefit or mechanistic explanation |
| Optimization incomplete as evidence | Many selected checkpoints at maximum update; one registered candidate; frozen observer | Global convergence, universal method failure, validated root cause |
| Heat endpoint saturation | First-step lower-bound events equal entire-episode events across all methods; signed initial fields | Meaningful discrimination of controller safety by the binary endpoint |
| Relative-risk degeneracy | Identical masks produce bootstrap [0,0] | Zero population risk or finite-sample guarantee from that interval |
| Zero Burgers events | Wilson upper95 0.9905% nominal, 2.9137% each stress | Zero risk; seeds do not triple parent count |
| Calibration is passive | 26 nominal margins; no action use, acceptance or rejection | Conditional harm, selective-risk control, OOD/uniform or closed-loop safety |
| Forecast estimands differ | Validation pooled all-candidate error vs test mean selected-action tick-100 relative error | Retrospective G2 success from small heat snapshot error |
| Common information, different processing | B0/B1 omit age/history processing used by learned methods | Equal effective observer information/capacity claim |
| Short held-action teacher | K=10, H=8, duplicate candidates possible; longer feedback cost differs | Oracle optimality for the 200-step task or sufficient headroom certificate |
| No no-age experiment | P-no-age absent from executed inventory | H4 mechanism claim |
| Narrow latency boundary | Stored ready snapshots; event joining/image/goal construction excluded | Raw-camera-to-actuator service latency |
| Missing timing protocols | 50 warm-ups, no thermal stabilization receipt, no extended 100k primary tail, scheduled/contention or trace-driven loop | Hard real-time/WCET, original full timing compliance, latency-aware control benefit |
| Fallback semantics | No exceptions observed, but no 5-ms wall-clock expiry in the simulator | Zero deadline failures from zero fallback |
| Event timing is not occupancy | CPU/launch/sync gaps can lie between CUDA events; marginal-quantile difference is not paired overhead | Claim that 97–99% is GPU compute utilization or that compile/graphs will give a measured gain |
| Limited numerical checks | Smooth/forced fixtures, not a population continuum error bound; teacher and evaluator share numerical machinery | Independent continuum solution accuracy or universal reference floor |
| Independent residual metrics absent | No saved test independently discretized residual/balance endpoint | Physical-consistency improvement at test |
| Draft action schema not fully represented | Measured endpoint returns finite two-component array; not every schema field | Complete typed-message service implementation |
| Statistical uncertainty | Three fitted seeds; conditional parent CIs, one machine, sequential sessions, no declared within-session block bootstrap | Environment replication, all-comparisons guarantees, hardware-general latency law |
| External validity | Synthetic 1D PDEs and image strips, no plant/PLC/HIL | Deployment readiness or industrial safety |
| Predecessor v2 defects/incomplete test | Reviewed reports only; partial outcomes remain unopened | Pooling predecessor test outcomes as independent confirmation |
| Budget/model provenance | Historical GPU occupancy and experiment-executor alias unreconciled | Exact GPU-hours remaining or retroactive Luna/Astra execution attribution |
| Administrative publication inputs | AUTHOR_INPUT_REQUIRED | Author approval, final declarations or submission readiness |

## Failures in this final analysis task

One exploratory Python `-c` command included a literal escaped newline and failed at parse time; the read-only inspection was rewritten as a here-string. Exploratory reads used nonexistent expected document names (`v3/AGENTS.md`, `contracts/observation_contract.md` and a root `state/RUN_LEDGER.md`); inventory confirmed that the applicable instructions are at the repository root, the observation contract is JSON, and historical experimental ledgers are under experiment state directories. Early combined terminal outputs were truncated, so consequential files and fields were reread in bounded form. None of these attempts ran an experiment or changed frozen evidence. The independent arithmetic run and source audit completed successfully.

The first final-packet QA run passed 1,959 of 1,960 checks but falsely flagged valid nested LaTeX closing braces as an unresolved template. The original receipt is retained in [qa_history/initial_validation_false_positive.json](qa_history/initial_validation_false_positive.json). The report-only validator was narrowed to actual uppercase template markers and rerun after correction. No scientific quantity or frozen evidence was changed to obtain a pass. These errors are recorded here, not appended to the already frozen experimental failure JSONL.

## Permitted next work

Human review can use this complete negative-result manuscript. Any future scientific experiment must use a separate prospective namespace and resolve the initial-condition/constraint contract, training sufficiency, statistical estimand and timing endpoint before new calibration/test. Any runtime optimization requires its own semantic/action equivalence checks and measured amended benchmark. None is initiated by this report. Missing author fields do not invalidate these analytical deliverables, but remain required before any publication workflow.
