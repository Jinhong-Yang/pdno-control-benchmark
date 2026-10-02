"""Revise editable Figure 1 source; defer rendering while timing is active."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];P=R.parents[1]/'publications/ieee_access_pdno_revision_v2_20261002';path=P/'overleaf/figures/fig01_design.tex';old=path.read_text()
new=r'''\documentclass[tikz,border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usetikzlibrary{arrows.meta,calc}
\definecolor{ink}{HTML}{233647}
\definecolor{rule}{HTML}{B9C5CF}
\definecolor{panelbg}{HTML}{F7F9FB}
\definecolor{blue}{HTML}{23699D}
\definecolor{bluebg}{HTML}{EDF4FA}
\definecolor{orange}{HTML}{C76B25}
\definecolor{orangebg}{HTML}{FCF2EA}
\definecolor{green}{HTML}{397462}
\definecolor{greenbg}{HTML}{EFF6F2}
\begin{document}
\hyphenpenalty=10000\exhyphenpenalty=10000
\begin{tikzpicture}[x=1cm,y=-1cm,text=ink,
  box/.style={draw=rule,fill=white,line width=.65pt,rounded corners=2pt,
    align=center,text width=3.48cm,minimum height=1.05cm,inner sep=4pt,
    font=\fontsize{8.7}{11}\selectfont},
  head/.style={anchor=west,font=\bfseries\fontsize{10}{12}\selectfont},
  flow/.style={-{Stealth[length=4pt,width=4pt]},line width=.8pt,draw=ink},
  note/.style={font=\fontsize{8.2}{10}\selectfont,align=center}]
\filldraw[fill=panelbg,draw=rule,line width=.55pt,rounded corners=3pt]
  (0,0) rectangle (18,2.68);
\node[head] at (.28,.35) {(a) Closed-loop control};
\node[box,text width=3.05cm] (plant) at (2,1.26)
  {\textbf{Burgers / heat plant}\\256 spatial points};
\node[box] (obs) at (6.65,1.26)
  {\textbf{Causal observation $o_t$}\\Sensors, images, ages};
\node[box] (control) at (11.2,1.26)
  {\textbf{Controller}\\Box / slew projection};
\node[box,text width=2.65cm] (action) at (15.65,1.26)
  {\textbf{Applied action $a_t$}\\Two actuators};
\draw[flow] (plant.east)--(obs.west);
\draw[flow] (obs.east)--(control.west);
\draw[flow] (control.east)--(action.west);
\draw[flow] (action.south)--(15.65,2.12)--(2,2.12)--(plant.south);
\node[note,fill=panelbg,inner xsep=6pt] at (8.8,2.12)
  {200 feedback steps \enspace$\cdot$\enspace $\Delta t=0.02$ simulation units};
\filldraw[fill=white,draw=rule,line width=.55pt,rounded corners=3pt]
  (0,2.96) rectangle (18,9.77);
\node[head] at (.28,3.32) {(b) Learned predictors and reference-solver candidate controllers};
\node[note,anchor=west,text=orange,font=\bfseries\fontsize{9}{11}\selectfont]
  at (.35,3.88) {P \enspace Action-factorized operator};
\node[box,draw=orange,fill=orangebg] (pe) at (2.25,4.68)
  {\textbf{Encode once}\\$z=E(o_t)$};
\node[box,draw=orange,fill=orangebg] (pr) at (6.75,4.68)
  {\textbf{One response branch}\\6 coefficient blocks};
\node[box,draw=orange,fill=orangebg,minimum height=1.22cm] (pf) at (11.25,4.68)
  {\textbf{Trunk + action basis}\\Recompute $T$ / cached $T$\\$K$ forecast fields};
\node[box,draw=orange,fill=orangebg] (ps) at (15.75,4.68)
  {\textbf{Score, select, project}\\One feasible command};
\draw[flow,draw=orange] (pe.east)--(pr.west);
\draw[flow,draw=orange] (pr.east)--(pf.west);
\draw[flow,draw=orange] (pf.east)--(ps.west);
\node[note,anchor=west,text=blue,font=\bfseries\fontsize{9}{11}\selectfont]
  at (.35,5.58) {B4 \enspace Shared-encoder, candidate-batched operator};
\node[box,draw=blue,fill=bluebg] (be) at (2.25,6.38)
  {\textbf{Encode once}\\$z=E(o_t)$};
\node[box,draw=blue,fill=bluebg] (br) at (6.75,6.38)
  {\textbf{Conditioned branch}\\$K$ actions, one batch};
\node[box,draw=blue,fill=bluebg,minimum height=1.22cm] (bf) at (11.25,6.38)
  {\textbf{Coordinate trunk}\\Recompute $T$ / cached $T$\\$K$ forecast fields};
\node[box,draw=blue,fill=bluebg] (bs) at (15.75,6.38)
  {\textbf{Score, select, project}\\One feasible command};
\draw[flow,draw=blue] (be.east)--(br.west);
\draw[flow,draw=blue] (br.east)--(bf.west);
\draw[flow,draw=blue] (bf.east)--(bs.west);
\node[note,anchor=west,text=green,font=\bfseries\fontsize{9}{11}\selectfont]
  at (.35,7.28) {O-cand \enspace Reference-solver diagnostic};
\node[box,draw=green,fill=greenbg] (oi) at (2.25,8.08)
  {\textbf{State initialization}\\Privileged truth / B0 estimate};
\node[box,draw=green,fill=greenbg] (or) at (6.75,8.08)
  {\textbf{Reference PDE solver}\\256 points; constant action};
\node[box,draw=green,fill=greenbg] (of) at (11.25,8.08)
  {\textbf{Ten candidate forecasts}\\8 steps; 128-point scoring};
\node[box,draw=green,fill=greenbg] (os) at (15.75,8.08)
  {\textbf{Score, select, project}\\Same objective as P / B4};
\draw[flow,draw=green] (oi.east)--(or.west);
\draw[flow,draw=green] (or.east)--(of.west);
\draw[flow,draw=green] (of.east)--(os.west);
\draw[draw=rule,line width=.5pt] (.35,8.87)--(17.65,8.87);
\node[note,text width=17cm] at (9,9.31)
  {\textbf{Control:} $K=10$, eight-step horizon. \quad
   \textbf{Timing:} $K=10,25,50,100,200$; cache off / on.\\
   Fixed-grid trunk and spatial-basis caches are specific to each frozen model; encoding remains once per request.};
\node[head] at (.28,10.22) {(c) Original campaign and separately identified revision evidence};
\node[box,text width=3.48cm,minimum height=1.55cm,fill=panelbg] at (2.25,11.46)
  {\textbf{Development / freeze}\\Parent-separated training\\Validation and calibration};
\node[box,text width=3.48cm,minimum height=1.55cm,fill=panelbg] at (6.75,11.46)
  {\textbf{Original control test}\\1,280 test parents\\25,600 executions};
\node[box,text width=3.48cm,minimum height=1.55cm,fill=panelbg] at (11.25,11.46)
  {\textbf{Original timing replay}\\Ready observations\\2.4 million requests};
\node[box,text width=3.48cm,minimum height=1.55cm,fill=greenbg,draw=green] at (15.75,11.46)
  {\textbf{Revision experiments}\\Oracles; cache / candidate count\\Budget; losses; new heat};
\end{tikzpicture}
\end{document}
'''
assert old!=new;path.write_text(new,encoding='utf-8')
log=json.loads((P/'CHANGELOG_revision.json').read_text());log.append({'location':'overleaf/figures/fig01_design.tex','original':old,'revised':new,'reason':'Add explicit oracle forecast path, per-model cache options, candidate-count regimes and separate revision evidence while retaining editable vector source. Compilation/rendering deferred until timing ends.','review':'Figure1/M2/M4'})
(P/'CHANGELOG_revision.json').write_text(json.dumps(log,indent=2,ensure_ascii=False),encoding='utf-8')
lines=['# Revision changelog','']
for i,row in enumerate(log,1):lines += [f"## {i}. {row['location']} ({row['review']})",'', 'Reason: '+row['reason'],'','Original:','```text',row['original'],'```','','Revised:','```text',row['revised'],'```','']
(P/'CHANGELOG_revision.md').write_text('\n'.join(lines),encoding='utf-8');print('Editable Figure1 source updated; existing PDF is stale until deferred build and visual inspection')
