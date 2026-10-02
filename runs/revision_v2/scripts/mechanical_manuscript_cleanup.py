from pathlib import Path
import re,difflib,json
P=Path('publications/ieee_access_pdno_revision_v2_20261002');O=P/'overleaf';changes=[]
def edit(p, pairs):
 old=p.read_text(encoding='utf-8');new=old
 for a,b,why,item in pairs:
  if a not in new:continue
  for i,line in enumerate(new.splitlines(),1):
   if a in line:changes.append({'location':f'{p.relative_to(P)}:{i}','original':line,'revised':line.replace(a,b),'reason':why,'review':item})
  new=new.replace(a,b)
 if new!=old:p.write_text(new,encoding='utf-8')
common=[('registered','primary','No external prospective registration or pre-test Git commit found','R-3/M6'),('Registered','Primary','Use primary analysis terminology','R-3/M6'),('fitted seeds','training seeds','Terminology','R-4/m1'),('locked control evaluation','final control evaluation','Terminology','R-3'),('locked control evaluator','final control evaluator','Terminology','R-3'),('Locked control aggregate','Final control-results aggregate','Terminology','R-3'),('serialized ready-observation replay','serialized timing replay','Unify replay terminology','R-4'),('ready-observation replay','serialized timing replay','Unify replay terminology','R-4'),('teacher queries','reference-solver training targets','Define supervised target source','R-4')]
for p in list(O.glob('*.tex'))+list((O/'tables').glob('*.tex'))+list((O/'scripts').glob('*.py')):edit(p,common)
edit(O/'main.tex',[(r'\history{Submission draft: Negative Result article. Prepared October 2026.}',r'\history{}','Remove internal editorial header; journal does offer Negative Result type','R-1/m8'),(r'\doi{Not assigned}',r'\doi{}','Leave publisher DOI field empty','R-1'),('The frozen namespace is included in both parent identity and random-number derivation.','The experiment identifier is included in both parent identity and random-number derivation.','Replace internal namespace terminology','R-4'),('The final manifest contains','The final manifest (checkpoint list) contains','Define manifest','R-4'),(' The planned no-age mechanism comparison, H4, was not executed.','','Retain unexecuted H4 only in supplement S2','R-6.5'),('before the one-shot test','before the original control evaluation','Later repairs belong to latency replay; do not imply the control test followed them','P0-2/R-3'),('0.98591','0.9859','Unify main ratio precision','R-5'),('0.99296','0.9930','Unify main ratio precision','R-5'),('0.9905\\%','0.99\\%','Main percentage precision','R-5'),('2.9137\\%','2.91\\%','Main percentage precision','R-5')])
p=O/'main.tex';s=p.read_text();start=s.index(r'\section*{AI-Assistance Disclosure}');end=s.index(r'\bibliographystyle',start)
a=s[start:end];b='OpenAI Codex and ChatGPT assisted with manuscript drafting and revision, supplementary narrative, implementation and analysis code, and figure-generation code. This assistance covered Sections I--VI and the supplementary material; numerical results were obtained from recorded computations.\n\n'
edit(p,[(a,b,'IEEE guidance places AI disclosure in Acknowledgment. Actual scope disclosed; author verification is not fabricated. Checked https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/ on 2026-10-02.','R-2/m8')])
edit(O/'supplement.tex',[('Submission draft','','Remove internal header','R-1'),('P-nr abbreviates P-no-rank. ','','Definition retained in method table only','R-5')])
edit(O/'author_information.tex',[('Agency(NIPA)','Agency (NIPA)','Requested spacing revision','R-5'),('government(MSIT)','government (MSIT)','Requested spacing revision','R-5'),('No.RS-2026-25621689','No. RS-2026-25621689','Requested grant-number spacing','R-5')])
p=O/'references.bib';s=p.read_text();a=re.search(r'@misc\{codex,.*?url=\{https://openai.com/codex/\}\}',s,re.S).group();edit(p,[(a,'','Name AI tools in disclosure rather than bibliography','R-2')])
# Remove authoring comments from editable manuscript/figure/table sources, not third-party class assets.
for p in O.rglob('*.tex'):
 s=p.read_text();new=re.sub(r'(?m)(?<!\\)%[^\n]*','',s)
 if s!=new:
  changes.append({'location':str(p.relative_to(P)),'original':'Unescaped TeX comments','revised':'Removed','reason':'Clean shared manuscript sources; third-party class/font files untouched','review':'R-1.5'});p.write_text(new)
(P/'AUTHOR_CONFIRMATION_REQUIRED.md').write_text('''# Author confirmation before submission\n\n- [[AUTHOR-INPUT: contributions]]\n- [[AUTHOR-INPUT: competing interests]]\n- [[AUTHOR-INPUT: AI product/model versions actually used, if available]]\n- Confirm actual AI assistance scope; do not certify unperformed author review.\n- Confirm author approval, originality, absence of simultaneous submission, and submission date.\n- Confirm current affiliations and ORCIDs in the submission system.\n\nThese items are outside the manuscript. Publication type will be aligned with the completed revision evidence. IEEE Access currently lists both Research Article and Negative Result; the work-order claim that the latter does not exist was corrected.\n''',encoding='utf-8')
(P/'CHANGELOG_revision.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
with (P/'CHANGELOG_revision.md').open('w',encoding='utf-8') as f:
 f.write('# Revision change log\n\nPhase 3 mechanical cleanup only; Phase 2 narrative rewrite waits for experiment results.\n\n')
 for i,c in enumerate(changes,1):f.write(f'## C{i:03d} — {c["location"]}\n\n- Review item: {c["review"]}\n- Before: {c["original"]}\n- After: {c["revised"]}\n- Reason: {c["reason"]}\n\n')
print('Mechanical edits:',len(changes))
