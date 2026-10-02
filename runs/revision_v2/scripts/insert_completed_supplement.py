"""Evidence-backed supplement edits while later experiments run; no abstract rewrite."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];P=R.parents[1]/'publications/ieee_access_pdno_revision_v2_20261002';path=P/'overleaf/supplement.tex';text=path.read_text();log=json.loads((P/'CHANGELOG_revision.json').read_text())
def replace(old,new,reason,review):
    global text
    assert old in text,old[:100]
    line=text[:text.index(old)].count('\n')+1;text=text.replace(old,new,1);log.append({'location':f'overleaf/supplement.tex:{line}','original':old,'revised':new,'reason':reason,'review':review})
replace('These are existing receipts; preparing this publication package involved no solver, model-inference, training, calibration-generation, test, or timing rerun.','These checks belong to the original campaign. The separately identified revision experiments add reference-solver control comparisons, component-gradient diagnostics, retraining, and timing analyses; their own completion records and numerical checks are reported separately.','Separate historical checks from newly authorized experimental work','R-3/S1')
anchor='\\clearpage\n\\section*{S2. Statistical implementation and protocol correspondence}'
p0=r'''\subsection*{Revision gradient-path diagnostic}
The seed-11 P and B5 weights at update 2,000 were restarted for 100 matched CPU updates. Their trainable weights initially agreed exactly. The physical residual remained connected to the response branch and trunk: all 100 physics-gradient norms were nonzero in each PDE. Only coordinate tensors were detached before enabling coordinate derivatives; this did not detach predictions from model parameters. The observer encoder and initial-field head were deliberately fixed for both methods. Table~\ref{tab:revision_gradients} reports the weighted component norms and final parameter differences.

The saved checkpoints lacked optimizer moments and random-generator states. Both optimizers were therefore reinitialized identically at the scheduled update-2,000 learning rate, with matched sampled training batches and collocation points. This diagnostic is not a bitwise continuation of historical CUDA training. The reported ratios describe the first 100 updates of the physics ramp, not every update of the training campaign. Nonzero but very small ratios support the G-B interpretation: the original weight of 0.01 provides a weak intervention on the training gradients. The revision weight sweep tests its sensitivity; equality of original actions alone does not establish that physical penalties are generally ineffective.
\input{tables/revision_gradients}

'''
replace(anchor,p0+anchor,'Add actual component-gradient evidence with restart and ramp limitations','P0-1/M3/S1')
old='Runtime repairs & Stale training-data path and parent-by-snapshot loader axes repaired before successful replay & Final benchmark source differs from pretest source; completed frozen timing arrays are retained.'
new='Runtime repairs & Latency replay path, parent-by-snapshot axes, and missing process record repaired on September 28, 2026, after control completion & Historical logs separate these repairs from control generation; missing before-source snapshots prevent an independent before/after timing-effect audit.'
replace(old,new,'Use the recorded repair chronology and disclose missing source diffs','P0-2/m9/S2')
anchor='\\textbf{Matched loss ablation.}'
provenance=r'''\textbf{Registration and repair chronology.} No external prospective registration or independently time-stamped pre-test source commit was located. We use ``primary'' for the original endpoints; internal local seals identify the recorded configuration but do not establish external preregistration. The recorded control start and elapsed duration imply completion at approximately 01:02:39 Korea Standard Time on September 28, 2026. The latency-replay repairs are logged between 01:10 and 01:12 that day. This completion time is reconstructed from local records, not independently certified. The stale path concerned train/validation data used for the thermal threshold; the axis repair loaded observation snapshots saved with test trajectories. Access was therefore not limited to development files. No complete before-source snapshot survives, so a categorical claim of zero influence on measured latency is unsupported. The control evaluation preceded these repairs.

\textbf{Denominator sensitivity.} The revision recomputes 10,000 paired-parent bootstrap draws using the same sampled parent indices for numerator and denominator. Tables below retain the archived primary fixed-denominator intervals and place the joint-denominator sensitivity intervals alongside them. The new random seed changes Monte Carlo draws; archived primary endpoints were not replaced by a rerun of the fixed-denominator bootstrap. Seed averaging still occurs within parent, and no simultaneous coverage claim is added.
\input{tables/revision_bootstrap}

'''
replace(anchor,provenance+anchor,'Disclose primary terminology, data access, and denominator-resampling evidence','P0-2/P0-3/E7/R-3/S2')
anchor='\\input{tables/supp_control.tex}'
add=r'''

\subsection*{Revision candidate-oracle comparisons on original parents}
Both reference-solver controllers retain the ten projected candidates, eight-step constant-action horizon, 256-point numerical solver, and 128-point candidate-score grid. O-cand-state uses the true current state only for this privileged diagnostic; O-cand-obs uses B0's seven-mode observation reconstruction. Future unknown disturbances are set to zero as in the original counterfactual targets. These two controllers completed 2,560 episodes over the same 1,280 original parents. This reuse supports a post-hoc comparison, not a fresh confirmatory evaluation. The joint-denominator intervals are also included in the accompanying CSV. No dense-candidate or longer-horizon oracle was executed, so the comparison does not identify which individual design choice causes an observed gap.
\input{tables/revision_oracle}
'''
replace(anchor,anchor+add,'Add complete oracle results and delimit the causal interpretation','E1/M2/S3')
anchor='\\end{document}'
addition=r'''
\clearpage
\section*{S8. Original heat endpoint and cost decomposition}
This section preserves the signed-initial-field heat benchmark separately from the revision's nonnegative-initial-field design. Every original heat method and training seed has the same any-time indicator: 351/384 nominal, 114/128 coefficient-shift, and 115/128 delay/dropout parents. Each violating parent already violates the lower bound at the first advanced step. There are no upper-bound events or newly violating parents later in an episode. This statement concerns the timing of the first event; it does not imply that the accumulated violation penalty is confined to that step.

\begin{figure}[ht]
\centering\includegraphics[width=.72\textwidth]{fig06_heat.pdf}
\caption{Original heat benchmark: parents classified by first-step violation versus no recorded violation. The same masks occur for every method and seed.}\label{fig:original_heat_events}
\end{figure}

The additive decomposition below uses stored advanced states and applied actions. Learned-method values average the three training seeds within the same parent population. Tracking, effort, slew, and squared violation penalties reconstruct the saved episode cost up to floating-point reduction differences. The final column is the first-step portion of the violation penalty. Thus saturation of the binary event endpoint must be distinguished from the cause of a cumulative cost difference.
\input{tables/revision_heat_cost}

\clearpage
\section*{S9. Modal comparator and candidate selections}
B0 reconstructs available measurements with truncation parameter seven and ridge coefficient 0.001. Its discrete linear-quadratic regulator uses identity state weight, input weight $0.1I$, and control interval 0.02; the Riccati equation is solved for each material parameter setting. The Burgers constant mode receives no feedback command. Heat adds a least-squares steady input for the modal target. Every command is subject to the same box and slew projection as the learned methods. A contemporaneous search ledger for selecting the truncation, weights, or ridge coefficient was not found. The 64 validation parents per PDE used to select the comparator among controller families are not evidence of a B0 hyperparameter search budget.

Candidate index nine (zero based) is the B0 command, and index four is the unchanged previous action. Projection can make these and other entries identical. Original output files did not retain the selected index at every tick, but stored the candidate sets and actions at ticks 0,10,$\ldots$,190. For those snapshots, the lower frequency counts actions that uniquely identify a candidate and the upper frequency counts every action compatible with it. These bounds preserve uncertainty about duplicate-candidate ties. They are not sampling confidence intervals. The revision oracles record their actual argmin indices at every tick and therefore have exact descriptive frequencies. Snapshot and full-trajectory frequencies should not be treated as an identical sampling design.
\input{tables/revision_choices}

'''
replace(anchor,addition+anchor,'Add original heat cost decomposition and documented B0/choice-frequency evidence','E6/E9/E1-b/S8/S9')
path.write_text(text,encoding='utf-8');(P/'CHANGELOG_revision.json').write_text(json.dumps(log,indent=2,ensure_ascii=False),encoding='utf-8')
lines=['# Revision changelog','']
for i,row in enumerate(log,1):
    lines += [f"## {i}. {row['location']} ({row['review']})",'', 'Reason: '+row['reason'],'','Original:','```text',row['original'],'```','','Revised:','```text',row['revised'],'```','']
(P/'CHANGELOG_revision.md').write_text('\n'.join(lines),encoding='utf-8');print('Completed supplement evidence inserted; PDF build deliberately deferred during latency campaign')
