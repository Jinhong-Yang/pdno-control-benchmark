from pathlib import Path
import sys,json,hashlib,time,argparse
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import numpy as np
from pdno.data import generate
from pdno.data.revision_heat import positive_heat_initial
from pdno.data.teacher_queries import build_teacher_queries
from pdno.physics.heat import heat_cn_step
parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--test',action='store_true');args=parser.parse_args()
spec=json.loads((R/'config/PHASE1_SPEC.json').read_text());generate._heat_initial=positive_heat_initial
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if args.test:
 rng=np.random.default_rng(2026100205);mins=[];maxs=[];after=[]
 for _ in range(1000):
  q=positive_heat_initial(rng,256);assert np.min(q)>=0 and np.max(q)<=.5000000001
  u=heat_cn_step(q,1/257,.02,rng.uniform(.005,.032),rng.uniform(.2,.5),np.zeros(256));assert np.isfinite(u).all();mins.append(float(q.min()));maxs.append(float(q.max()));after.append(float(u.min()))
 (R/'results/E5_initial_condition_tests.json').write_text(json.dumps({'status':'PASS','synthetic_test_draws':1000,'min_initial':min(mins),'max_initial':max(maxs),'minimum_after_one_zero_action_tick':min(after),'contract_sha256':sha(R/'heat_ic_spec.md'),'not_benchmark_results':True},indent=2));print('1000 initial-condition fixtures PASS')
if args.generate:
 manifest=R/'config/heat_parent_manifest.json'
 if not manifest.exists():
  counts={'train':256,'validation':64,'locked_nominal':384,'locked_coefficient_ood':128,'locked_delay_dropout':128};parents=[];ix=0
  for role,count in counts.items():
   for j in range(count):
    pid=f'pdno-revision-v2-positive-heat-20261002:{role}:{j:06d}';h=int.from_bytes(hashlib.sha256(pid.encode()).digest()[:7],'big');parents.append({'parent_id':pid,'split':role,'pde':'heat','initial_condition_seed':h,'physical_parameter_seed':h+1,'disturbance_seed':h+2,'target_status':'NOT_GENERATED'});ix+=1
  manifest.write_text(json.dumps({'manifest_version':1,'namespace':'pdno-revision-v2-positive-heat-20261002','metadata_only':True,'test_opened':False,'roles':{'heat':{'role_counts':counts,'parents':parents}}},indent=2))
 out=R/'data/positive_heat';assert not out.exists();t=time.perf_counter();records=generate.generate_roles(manifest,out,roles=('train','validation'),outer_ticks=128)
 qs=[]
 for role in ['train','validation']:
  qs.append(build_teacher_queries(out/role/'heat/trajectories.npz','heat',role,R/f'data/positive_heat_queries/{role}/heat/teacher_queries.npz',snapshots_per_parent=4 if role=='train' else 1,q_max_override=2.1322593092918396))
 (R/'results/E5_train_generation.json').write_text(json.dumps({'status':'TRAIN_VALIDATION_COMPLETE_TEST_NOT_GENERATED','seconds':time.perf_counter()-t,'manifest_sha256':sha(manifest),'initial_condition_source_sha256':sha(R/'src/pdno/data/revision_heat.py'),'roles':records,'queries':qs},indent=2));print('Positive heat train/validation generation COMPLETE',time.perf_counter()-t)
