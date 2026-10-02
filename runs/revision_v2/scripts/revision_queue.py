"""Run authorized revision experiments sequentially, preserving every completed output."""
from pathlib import Path
import sys,json,time,subprocess,datetime,hashlib,os
R=Path(__file__).resolve().parents[1];ROOT=R.parents[1];py=R/'.venv/Scripts/python.exe';started=time.time();ledger=R/'state/REVISION_QUEUE.json';records=[]
def save(status,**extra):
    data={'status':status,'pid':os.getpid(),'started_utc':datetime.datetime.fromtimestamp(started,datetime.timezone.utc).isoformat(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.time()-started,'completed':records,**extra};tmp=ledger.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(ledger)
save('WAITING_FOR_E1_GPU_COMPLETION')
while not (R/'results/E1_burgers_cuda_COMPLETE.json').exists():
    if time.time()-started>7200:save('STOPPED_E1_NOT_COMPLETE');raise SystemExit('E1 did not finish within bounded wait')
    if (R/'state/STOP_REVISION').exists():save('STOP_REQUESTED');raise SystemExit()
    time.sleep(10)
# The E1 completion marker is written as the final statement; allow its CUDA process to release context.
time.sleep(5)
stages=[('batch_support','tests/check_batch_rollout.py',[]),('E2_timing','scripts/run_e2_timing.py',[]),('E3_targets','scripts/generate_e3_targets.py',[]),('E3_train','scripts/run_gpu_training.py',['--stage','E3']),('E3_evaluate','scripts/evaluate_revision.py',['--stage','E3']),('E5_train','scripts/run_gpu_training.py',['--stage','E5']),('E5_evaluate','scripts/evaluate_revision.py',['--stage','E5']),('E4_train','scripts/run_gpu_training.py',['--stage','E4']),('E4_evaluate','scripts/evaluate_revision.py',['--stage','E4'])]
for label,script,args in stages:
    if time.time()-started>23*3600:save('BUDGET_STOP_BEFORE_NEXT_STAGE',next_stage=label);raise SystemExit('Conservative GPU process wall budget')
    if (R/'state/STOP_REVISION').exists():save('STOP_REQUESTED');raise SystemExit()
    path=R/script;codehash=hashlib.sha256(path.read_bytes()).hexdigest();log=R/f'logs/queue_{label}.log';assert not log.exists(),str(log)
    with log.open('w',encoding='utf-8') as stream:
        t=time.time();proc=subprocess.Popen([str(py),'-B','-X','utf8',str(path),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        save('RUNNING',stage=label,child_pid=proc.pid,script_sha256=codehash,log=str(log.relative_to(R)))
        try:code=proc.wait(timeout=max(1,23*3600-(time.time()-started)))
        except subprocess.TimeoutExpired:
            # Terminate only this queue's child process tree when its explicit budget expires.
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
            save('BUDGET_EXPIRED_CHILD_TERMINATED',stage=label);raise SystemExit(124)
    row={'stage':label,'exit_code':code,'process_wall_seconds':time.time()-t,'script_sha256':codehash,'log':str(log.relative_to(R))};records.append(row)
    if code:save('FAILED_REQUIRES_INSPECTION',failed_stage=label);raise SystemExit(code)
    save('BETWEEN_STAGES');print(json.dumps(row),flush=True)
save('EXPERIMENT_QUEUE_COMPLETE_ANALYSIS_AND_MANUSCRIPT_PENDING')
