"""Final report QA: read frozen evidence, write one report receipt; never run models."""
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import numpy as np

R = Path(__file__).resolve().parents[1]
O = R / 'reports'
A = O / 'analysis_20260928'
START = time.perf_counter()
checks = []

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def check(name, value):
    checks.append({'check': name, 'pass': bool(value)})

def close(a, b):
    return bool(np.allclose(a, b, rtol=1e-11, atol=1e-12))

required = [R/'manuscript_v3.md', *[O/f for f in (
    'REPORT_KO.md', 'DECISION_PACKET.json', 'claim_evidence.csv',
    'failures_and_limitations.md', 'REPRODUCE.md')]]
for p in required:
    check('required/' + p.name, p.is_file() and p.stat().st_size > 1000)

packet = read(O/'DECISION_PACKET.json')
audit = read(A/'audit.json')
claims = rows(O/'claim_evidence.csv')
control = read(R/'evidence/locked_test_analysis_v3.json')
latency = read(R/'evidence/latency_analysis_v3.json')
manifest = read(R/'evidence/confirmatory_checkpoint_manifest_v3.json')
cal = read(R/'evidence/calibration_lock_v3.json')
pool = rows(A/'latency_pooled_descriptive.csv')
check('claim_count', len(claims) == packet['claim_rows'] == 183)
check('unique_claim_ids', len({r['claim_id'] for r in claims}) == len(claims))
check('audit_checks', audit['status'] == 'PASS' and audit['arithmetic_checks'] == 1694 and not audit['failed_checks'])
check('outcome_and_authors', packet['outcome'] == 'NO_GO_V3_PROPOSED_CLAIM' and packet['authors'] == 'AUTHOR_INPUT_REQUIRED')
check('risk_unavailable', packet['risk']['conditional_harm'] is None and not packet['risk']['calibration_changes_actions'])
check('zero_new_experiment', packet['validation']['new_experiment_runs'] == 0)

for r in claims:
    cid = r['claim_id']
    p = R/r['evidence_artifact']
    check('claim_hash/' + cid, sha(p) == r['source_sha256'])
    s = json.loads(r['support_exact_json'])
    check('claim_locator_present/' + cid, bool(r['evidence_locator']))
    if cid.startswith('G2_'):
        k = int(cid.split('_')[1])
        check('exact_support/' + cid, s == manifest['checkpoints'][k]['metadata']['validation'])
    elif cid.startswith('CONTROL_'):
        _, gi, method = cid.split('_', 2)
        g = control['groups'][int(gi)]
        m = next(x for x in g['controller_metrics'] if x['method'] == method)
        check('exact_support/' + cid, s == {'metrics': {k:v for k,v in m.items() if k != 'seed_results'}, 'comparison': g['paired_vs_h3_comparator'][method]})
    elif cid.startswith('MATCHED_'):
        _, gi, label = cid.split('_', 2)
        check('exact_support/' + cid, s == control['groups'][int(gi)][label])
    elif cid.startswith('LAT_SEED_'):
        check('exact_support/' + cid, s == latency['controller_groups'][int(cid.rsplit('_', 1)[1])])
    elif cid.startswith('LAT_POOL_'):
        check('exact_support/' + cid, s == pool[int(cid.rsplit('_', 1)[1])])
    elif cid.startswith('CAL_'):
        g = cal['groups'][int(cid.split('_')[1])]
        check('exact_support/' + cid, s == {k:v for k,v in g.items() if k != 'parent_scores'})
    elif cid.startswith('H1_'):
        g = next(x for x in latency['H1_p99_ratio_analysis'] if x['pde'] == cid[3:])
        check('exact_support/' + cid, s == g)
    elif cid == 'RUNTIME':
        check('exact_support/' + cid, s == read(O/'RUNTIME_METADATA.json'))
    elif cid == 'NUMERICS':
        check('exact_support/' + cid, s == read(p))

expected_counts = {'control_all_methods.csv':48, 'control_per_seed.csv':120,
                   'latency_per_seed.csv':40, 'latency_pooled_descriptive.csv':16,
                   'heat_constraint_diagnostic.csv':60, 'P_B5_matched_actions.csv':18,
                   'calibration_margins.csv':26, 'selected_checkpoints.csv':36,
                   'failure_event_index.csv':138}
for f, n in expected_counts.items():
    check('csv_count/' + f, len(rows(A/f)) == n)
methods = {'B0','B1','B2','B3','B4','B5','P-no-rank','P'}
for g in control['groups']:
    check('required_methods/' + g['role'] + '/' + g['pde'], {m['method'] for m in g['controller_metrics']} == methods)
    p = g['paired_vs_h3_comparator']['P']
    # Decision fields are independently validated by the arithmetic receipt;
    # here verify table direction without interpreting arbitrary JSON key names.
for r in rows(A/'control_all_methods.csv'):
    if r['method'] != 'B0':
        check('higher_observed_cost/' + '/'.join(r[k] for k in ('role','pde','method')), float(r['relative_cost_difference_vs_B0']) > 0)
for r in rows(A/'latency_per_seed.csv'):
    check('p99_target/' + '/'.join(r[k] for k in ('pde','method','seed')), (float(r['p99_ms']) < 2) == (r['method'] == 'B0'))

events = [json.loads(s) for s in (R/'evidence/failures_and_modifications_v3.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
index = rows(A/'failure_event_index.csv')
check('complete_failure_event_content', len(events) == len(index) == 138 and all(json.loads(r['full_record_json']) == event and int(r['line']) == i+1 for i,(event,r) in enumerate(zip(events,index))))

# Check the originating task's correction directly against all timing arrays.
timings = defaultdict(lambda: {'host': [], 'stream': []})
for p in sorted((R/'evidence/latency_raw_v3').glob('*.npz')):
    with np.load(p, allow_pickle=False) as z:
        k = (str(z['pde'].item()), str(z['method'].item()))
        timings[k]['host'].append(z['latency_ms'].copy())
        timings[k]['stream'].append(z['cuda_stream_span_ms'].copy())
corrected_path = R/'evidence/cuda_runtime_assessment_v3_corrected.json'
corrected = read(corrected_path)
check('corrected_cuda_sha', sha(corrected_path) == 'f3b5d28f8e02b18143a8f508e9c5001ed4a93a8ef273d5a1cd979154bf214a6d')
check('corrected_cuda_group_inventory', len(corrected['groups']) == 16 and {(r['pde'],r['method']) for r in corrected['groups']} == set(timings))
for r in corrected['groups']:
    key = (r['pde'],r['method'])
    name = '/'.join(key)
    host = np.concatenate(timings[key]['host'])
    stream = np.concatenate(timings[key]['stream'])
    stream = stream[np.isfinite(stream)]
    check('corrected_cuda_counts/' + name, len(host) == r['host_requests'] and len(stream) == r['cuda_stream_requests'])
    for field,q in [('p50_ms',.5),('p95_ms',.95),('p99_ms',.99),('p99_9_ms',.999)]:
        check('corrected_cuda_host/' + name + '/' + field, close(np.quantile(host,q),r['host_wall'][field]))
        if len(stream):
            check('corrected_cuda_stream/' + name + '/' + field, close(np.quantile(stream,q),r['cuda_stream_span'][field]))
    if len(stream):
        h,s = np.quantile(host,.99), np.quantile(stream,.99)
        check('corrected_cuda_ratio/' + name, close(s/h,r['p99_cuda_stream_to_host_wall_ratio']))
        check('corrected_cuda_delta/' + name, close(h-s,r['p99_host_minus_stream_ms']))
    else:
        check('corrected_cuda_null_cpu/' + name, r['cuda_stream_span'] is None and r['p99_cuda_stream_to_host_wall_ratio'] is None and r['p99_host_minus_stream_ms'] is None)

# Markdown link integrity and exact machine-rendered table preservation.
tables = set()
for p in [R/'manuscript_v3.md',O/'REPORT_KO.md',O/'REPRODUCE.md',O/'failures_and_limitations.md']:
    s = p.read_text(encoding='utf-8')
    check('no_template_marker/' + p.name, re.search(r'\{\{[A-Z_]+\}\}', s) is None and '\ufffd' not in s)
    check('balanced_code_fences/' + p.name, sum(line.startswith('```') for line in s.splitlines()) % 2 == 0)
    for link in re.findall(r'\]\(([^)]+)\)',s):
        if link.startswith(('http://','https://','#')):
            continue
        local = unquote(link.strip('<>').split('#')[0])
        check('local_link/' + p.name + '/' + local, (p.parent/local).exists())
    for block in re.findall(r'(?:^\|.*(?:\n|$))+',s,re.M):
        tables.add(hashlib.sha256(block.rstrip('\n').encode()).hexdigest())
for name,h in read(O/'table_provenance.json')['tables'].items():
    check('rendered_table/' + name, h in tables)
for p,h in read(O/'table_provenance.json')['inputs'].items():
    check('table_input/' + p, sha(R/p) == h)
for r in packet['frozen_evidence_hashes']:
    check('packet_evidence_hash/' + r['path'], sha(R/r['path']) == r['sha256'])
for r in rows(O/'historical_commands.csv'):
    script = re.search(r"'scripts/([^']+)'",r['command']).group(1)
    check('historical_command_source/' + script, sha(R/'scripts'/script) == r['script_sha256'] and r['current_task_executed'] == 'False')

# Current administrative state may legitimately differ after final handoff.
admin = {'state/RUN_STATE.json','state/RUN_LEDGER.md','state/STATE.md','state/HANDOFF.md'}
drift = []
frozen_rows = rows(A/'freeze_hash_verification.csv')
for r in frozen_rows:
    p = R/r['path']
    current = sha(p)
    same = current == r['expected']
    name = r['path'].replace('\\','/')
    if not same:
        drift.append({'path':name,'frozen_sha256':r['expected'],'entry_sha256':r['actual'],'current_sha256':current,'administrative':name in admin})
    check('freeze_preservation/' + name, same or name in admin)
check('freeze_inventory_count', len(frozen_rows) == 1072)

receipt = {'status':'PASS' if all(r['pass'] for r in checks) else 'FAIL',
           'completed_at_utc':datetime.now(timezone.utc).isoformat(),
           'elapsed_seconds':time.perf_counter()-START,
           'scope':'Final deliverable arithmetic linkage, hashes, rendered tables, local links, full error index, and originating corrected CUDA quantiles; no experiment execution.',
           'checks':len(checks),'failed_checks':[r for r in checks if not r['pass']],
           'claim_rows':len(claims),'corrected_cuda_groups_verified':len(corrected['groups']),
           'frozen_inventory_count':len(frozen_rows),'administrative_hash_differences':drift,
           'required_deliverable_sha256':{str(p.relative_to(R)):sha(p) for p in required},
           'all_checks':checks}
(O/'FINAL_PACKET_VALIDATION.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('all_checks','required_deliverable_sha256','administrative_hash_differences')},indent=2))
raise SystemExit(0 if receipt['status']=='PASS' else 1)
