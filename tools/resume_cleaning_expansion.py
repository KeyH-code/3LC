"""One explicitly authorized epoch-4 D080 recovery; frozen original executor stays unchanged."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import types

ROOT=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
SOURCE=ROOT/'reports/train-cleaning-expansion-ss04-20261101'
OUT=ROOT/'reports/train-cleaning-expansion-resume-r1-20261102'
AUTH=ROOT/'reports/cleaning-expansion-080-070-060-20261029/TRAINING-RESUME-AUTHORIZATION-20261102.json'
CHECKPOINT_HASH='6fe96d336204c1ce3189efdc49f73b8b78e9058242f0294ec6e7a4d43b445623'
ENGINE=None
AMENDMENT=AUTH.parent/'RECOVERY-TECHNICAL-RETRY-AMENDMENT-20261102.json'
REPORT_FILES=('comparison.json','metrics.csv','per_class.csv','class_deltas.csv','change_scale.csv','RESULT.md','independent_verification.json')


def core():
    m=types.ModuleType('frozen_expansion');p=ROOT/'tools/train_cleaning_expansion.py';m.__file__=str(p)
    definitions=p.read_text(encoding='utf-8').split('\nBOOT_EVENTS=[]\n')[0]
    exec(compile(definitions,str(p),'exec'),m.__dict__)
    m.BOOT_EVENTS=[]
    return m


def check_checkpoint_metadata(ckpt):
    c=core();c.check(isinstance(ckpt,dict),'Checkpoint must be a dictionary')
    c.check(ckpt.get('epoch')==3,'Expected exactly four completed epochs')
    optimizer=ckpt.get('optimizer')
    c.check(isinstance(optimizer,dict) and bool(optimizer.get('state')) and bool(optimizer.get('param_groups')),'Missing optimizer state')
    c.check(ckpt.get('ema') is not None and bool(ckpt.get('scaler')) and isinstance(ckpt.get('updates'),int) and ckpt['updates']>0,'Missing EMA/scaler/update state')
    c.check(isinstance(ckpt.get('best_fitness'),(int,float)) and math.isfinite(ckpt['best_fitness']),'Missing finite best fitness')
    args=ckpt.get('train_args',{})
    for k,v in c.training_options().items():
        # auto optimizer sets this effective input to zero; compare to the actual original initialization.
        if k=='device':c.check(str(args.get(k))==str(v),'Checkpoint option drift: '+k)
        elif k!='warmup_bias_lr':c.check(args.get(k)==v,'Checkpoint option drift: '+k)
    return dict(completed_epochs=4,next_epoch=5,target_epochs=28,optimizer_present=True,ema_present=True,
        scaler_present=True,ema_updates=ckpt['updates'],best_fitness=ckpt['best_fitness'])


def merge_epochs(prior,new):
    c=core();c.check([r['epoch'] for r in prior]==list(range(1,5)),'Original epoch coverage mismatch')
    c.check([r['epoch'] for r in new]==list(range(5,29)),'Resumed epoch coverage mismatch')
    return prior+new


def check_resume_args(actual,baseline):
    c=core();c.check(str(actual.get('resume'))==str(OUT/'checkpoint/last.pt'),'Resume source changed or false')
    paths={'model','data','project','name','save_dir','resume'}
    for k,v in baseline.items():
        if k not in paths:c.check(actual.get(k)==v,'Effective resume argument drift: '+k)


def relocate_resume_args(args):
    args.model=args.resume=str(OUT/'checkpoint/last.pt')
    args.data=str(OUT/'D080.yaml');args.project=str(OUT/'D080');args.name='resumed'
    args.save_dir=str(OUT/'D080/resumed')


def ready(c):
    c.check(not (OUT/'supervision_started.json').exists() and not (OUT/'run_started.json').exists(),'Cannot repeat recovery')
    c.check((OUT/'seal.json').is_file() and (OUT/'preflight/result.json').is_file(),'Recovery sealed preflight required')
    c.verify_seal()


def configured():
    c=core();c.OUT=OUT;c.SELF=SELF
    normal=c.allowed_file
    def allowed(path,writing):
        p=Path(path).resolve()
        if p.is_relative_to(SOURCE.resolve()):return not writing
        if p==ROOT/'tools/train_cleaning_expansion.py':return not writing
        return normal(path,writing)
    c.allowed_file=allowed
    original_train=c.train
    c.train=lambda band,stage:resume_train(c,stage) if band=='D080' else original_train(band,stage)
    old_stage=c.stage_path
    c.stage_path=lambda role,band=None,ref=None:OUT/'inspection' if role=='inspect' else old_stage(role,band,ref)
    normal_timeout=c.timeout
    c.timeout=lambda role,deadline,current:normal_timeout('preflight' if role=='inspect' else role,deadline,current)
    original_launch=c.launch
    def launch(role,band=None,ref=None,deadline=None):
        # The frozen launcher passes this variable through to its workers; bootstrap uses our separate marker.
        return original_launch(role,band,ref,deadline)
    c.launch=launch
    c.ready_to_run=lambda:ready(c)
    return c


def resume_train(c,stage):
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import init_seeds
    c.local_runtime();init_seeds(42,deterministic=True)
    checkpoint=OUT/'checkpoint/last.pt';c.check(c.sha(checkpoint)==CHECKPOINT_HASH,'Resume checkpoint drift')
    c.CHECKPOINTS[str(checkpoint.resolve())]=CHECKPOINT_HASH
    safety=c.trainer_factory(DetectionTrainer)
    baseline=c.load(OUT/'D080/original_initialization.json')
    prior_path=OUT/'D080/prior_epochs.jsonl'
    prior=[json.loads(x) for x in prior_path.read_text(encoding='utf-8').splitlines()]
    shutil.copyfile(prior_path,stage/'epochs.jsonl')
    new=[]
    class ResumeTrainer(safety):
        def check_resume(self,overrides):
            super().check_resume(overrides)
            c.check(self.resume,'Native resume was disabled')
            relocate_resume_args(self.args)
        def get_model(self,cfg=None,weights=None,verbose=True):
            c.check(weights is not None and self.resume,'Resume cannot fall back to fresh training')
            return DetectionTrainer.get_model(self,cfg=cfg,weights=weights,verbose=verbose)
        def _load_checkpoint_state(self,ckpt):
            check_checkpoint_metadata(ckpt)
            super()._load_checkpoint_state(ckpt)
            c.check(len(self.optimizer.state)>0 and self.ema.updates==ckpt['updates'],'Optimizer/EMA restoration failed')
            c.check(self.scaler.state_dict()==ckpt['scaler'],'Scaler restoration failed')
    options=c.training_options();options.update(resume=str(checkpoint),model=str(checkpoint),data=str(OUT/'D080.yaml'),
        project=str(OUT/'D080'),name='resumed',save_dir=str(OUT/'D080/resumed'),exist_ok=False)
    trainer=ResumeTrainer(overrides=options)
    def initialized(t):
        check_resume_args(vars(t.args),baseline['effective_args'])
        c.check(t.start_epoch==4 and t.epochs==28 and bool(t.amp),'Resume epoch/AMP contract mismatch')
        c.check(type(t.optimizer).__name__=='AdamW' and t.optimizer.param_groups[0]['betas']==(.9,.999),'Optimizer type/betas drift')
        c.ab().assert_loader(t.train_loader.dataset,c.data_rows('D080'),c.BOXES['D080'])
        c.ab().assert_loader(t.test_loader.dataset,c.data_rows('val966'),11487,ordered=False)
        c.save(stage/'initialization.json',dict(initial_state_sha256=baseline['initial_state_sha256'],
            initial_hash_provenance='Original uninterrupted segment; not a new recovery initialization',
            effective_args=vars(t.args),resume_checkpoint_sha256=CHECKPOINT_HASH,start_epoch=4,epochs=28,
            restored_optimizer=True,restored_ema_updates=t.ema.updates,restored_scaler=True))
    def epoch_start(t):t._oom_retries=3
    def epoch_end(t):
        if getattr(t,'expansion_final_eval',False):return
        row=dict(epoch=t.epoch+1,metrics={k:float(v) for k,v in t.metrics.items()},fitness=float(t.fitness),
            best_updated=t.best_fitness==t.fitness,at=c.stamp(),segment='resumed')
        new.append(row)
        with (stage/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print('RESUMED EPOCH D080',row['epoch'],'AP95',row['fitness'],flush=True)
    trainer.add_callback('on_train_start',initialized);trainer.add_callback('on_train_epoch_start',epoch_start)
    trainer.add_callback('on_fit_epoch_end',epoch_end);trainer.train()
    combined=merge_epochs(prior,new)
    c.check(c.sha(checkpoint)==CHECKPOINT_HASH,'Input checkpoint changed')
    return dict(band='D080',epochs=28,resumed_epochs=24,prior_epochs=4,resume=True,
        best_epoch=max(r['epoch'] for r in combined if r['best_updated']),
        best=str(trainer.best),last=str(trainer.last),best_sha256=c.sha(trainer.best),last_sha256=c.sha(trainer.last),
        initial_state_sha256=baseline['initial_state_sha256'],resume_checkpoint_sha256=CHECKPOINT_HASH,
        native_final_metrics={k:float(v) for k,v in trainer.metrics.items()},uninterrupted_equivalence=False)


def inspection(c):
    import torch
    from ultralytics.utils.patches import torch_load
    p=OUT/'checkpoint/last.pt'
    c.check(c.sha(p)==CHECKPOINT_HASH,'Recovery checkpoint hash mismatch')
    ckpt=torch_load(p,map_location='cpu')
    result=check_checkpoint_metadata(ckpt)
    baseline=c.load(SOURCE/'D080/train/initialization.json')['effective_args']
    for key,value in baseline.items():c.check(ckpt['train_args'].get(key)==value,'Saved effective option differs: '+key)
    tensors=[]
    for net in (ckpt.get('model'),ckpt.get('ema')):
        if net is not None:tensors+=list(net.state_dict().values())
    tensors += [v for state in ckpt['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v)]
    c.check(all(torch.isfinite(v).all() for v in tensors if torch.is_tensor(v) and v.is_floating_point()),'Nonfinite checkpoint state')
    c.check(sum(p.numel() for p in ckpt['ema'].parameters())==3011628,'Checkpoint architecture mismatch')
    c.check(ckpt['ema'].names==c.NAMES,'Checkpoint class mapping mismatch')
    return dict(result,checkpoint_sha256=c.sha(p),finite_tensor_count=len(tensors),no_dataset_inference=True,no_optimizer_steps=True)


def worker(c,role,band,ref):
    stage=c.stage_path(role,band,ref);start=time.perf_counter()
    try:
        import torch,ultralytics
        from ultralytics.utils import ONLINE,SETTINGS
        c.check(sys.version_info[:3]==(3,12,15) and torch.__version__=='2.14.1+cu130' and ultralytics.__version__=='8.4.165'
            and torch.version.cuda=='13.0' and not ONLINE and not SETTINGS['sync'],'Frozen offline runtime mismatch')
        if role=='inspect':result=inspection(c)
        elif role=='preflight':result=c.preflight(stage)
        elif role=='train':result=c.train(band,stage)
        else:result=c.evaluate(band,ref,stage)
        c.check(not BOOT_EVENTS and not list(Path.cwd().glob('guard_events*.jsonl')),'Guard event in worker')
        if role!='inspect':result['guarded_spawn_children']=c.verify_child_guards(Path.cwd())
        result.update(state='completed',seconds=time.perf_counter()-start,at=c.stamp(),no_3lc_import=not any(k.split('.')[0] in ['tlc','tlc_ultralytics','tlcconfig'] for k in sys.modules))
        c.save(stage/'result.json',result)
    except BaseException:
        c.save(stage/'failure.json',dict(at=c.stamp(),traceback=traceback.format_exc()));raise


def inspect_once(c):
    c.check(not OUT.exists(),'Existing recovery preserved; no retry')
    a=c.load(AUTH);c.check(a['approved_by_user'] and a['user_message']=='B，同意恢复执行器和预算','Missing recovery authorization')
    old=core();old.verify_seal()
    c.check(old.sha(SOURCE/'D080/yolov8n/weights/last.pt')==CHECKPOINT_HASH,'Source last drift')
    OUT.mkdir();(OUT/'runtime').mkdir();(OUT/'checkpoint').mkdir()
    shutil.copyfile(SOURCE/'D080/yolov8n/weights/last.pt',OUT/'checkpoint/last.pt')
    c.save(OUT/'recovery_authorization.json',a)
    c.save(OUT/'retry_amendment.json',c.load(AMENDMENT))
    c.launch('inspect',deadline=time.perf_counter()+300)


def relocate_row(row):
    result=dict(row)
    for key in ('image','label'):
        rel=Path(row[key]).resolve().relative_to((SOURCE/'data').resolve())
        result[key]=str(OUT/'data'/rel)
    return result


def prepare(c):
    c.check(c.load(OUT/'inspection/result.json')['state']=='completed','Checkpoint inspection required')
    c.check(not (OUT/'prepared.json').exists() and not (OUT/'data').exists(),'Existing preparation preserved')
    old=core();old.verify_seal()
    keys=('D0','D085','D080','D070','D060','val966')
    groups={k:old.data_rows(k) for k in keys}
    required=sum(Path(r[f]).stat().st_size for rows in groups.values() for r in rows for f in ('image','label'))
    c.check(shutil.disk_usage(OUT).free>required+4_000_000_000,'Insufficient recovery copy reserve')
    pinned=dict(c.load(SOURCE/'protected.json'))
    # Preserve the complete original attempt's non-dataset evidence including checkpoint bytes.
    old_files=[p for p in SOURCE.rglob('*') if p.is_file() and p.relative_to(SOURCE).parts[0] not in ('data','runtime')]
    for p in old_files:pinned[p.relative_to(ROOT).as_posix()]=c.sha(p)
    for k,rows in groups.items():
        copied=[]
        for row in rows:
            r=relocate_row(row)
            for field in ('image','label'):
                src=Path(row[field]);dest=Path(r[field]);dest.parent.mkdir(parents=True,exist_ok=True)
                c.check(c.sha(src)==row[field+'_sha256'],'Source runtime data drift')
                with src.open('rb') as f,dest.open('xb') as g:shutil.copyfileobj(f,g)
                c.check(c.sha(dest)==row[field+'_sha256'] and src.stat().st_ino!=dest.stat().st_ino,'Recovery independent copy mismatch')
            copied.append(r)
        c.save(OUT/'manifests'/(k+'.json'),copied);print('COPIED',k,len(rows),flush=True)
    shutil.copytree(SOURCE/'architecture',OUT/'architecture')
    import yaml
    for p in SOURCE.glob('*.yaml'):
        value=yaml.safe_load(p.read_text(encoding='utf-8'))
        for key in ('path','train','val'):
            value[key]=str(OUT/Path(value[key]).relative_to(SOURCE))
        with (OUT/p.name).open('x',encoding='utf-8') as f:yaml.safe_dump(value,f,sort_keys=False)
    c.save(OUT/'authorization.json',c.load(SOURCE/'authorization.json'))
    c.save(OUT/'baseline_initialization.json',c.load(SOURCE/'baseline_initialization.json'))
    c.save(OUT/'protected.json',pinned)
    model_dir=OUT/'D080/resumed';(model_dir/'weights').mkdir(parents=True)
    for name in ('best.pt','last.pt'):shutil.copyfile(SOURCE/'D080/yolov8n/weights'/name,model_dir/'weights'/name)
    shutil.copyfile(SOURCE/'D080/yolov8n/results.csv',model_dir/'results.csv')
    shutil.copyfile(SOURCE/'D080/train/epochs.jsonl',OUT/'D080/prior_epochs.jsonl')
    c.save(OUT/'D080/original_initialization.json',c.load(SOURCE/'D080/train/initialization.json'))
    c.save(OUT/'interrupted_attempt.json',dict(source=str(SOURCE),prior_completed_epochs=4,
        prior_trace=c.load(SOURCE/'D080/train/launch.json'),original_process_interrupted=True,
        prior_training_csv_seconds=760.223,exact_interruption_time_unknown=True))
    c.save(OUT/'prepared.json',dict(at=c.stamp(),copy_bytes=required,original_evidence_files=len(old_files),
        independent_copies=True,D080_resume=True,D070_D060_from_scratch=True))
    c.verify_pins();print('RECOVERY PREPARED; no training/inference',flush=True)


def seal(c):
    c.check(c.load(OUT/'preflight/result.json')['state']=='completed','Preflight required')
    paths=[SELF,ROOT/'tools/summarize_cleaning_expansion_resume.py',ROOT/'tests/test_resume_cleaning_expansion.py',
        ROOT/'tools/train_cleaning_expansion.py',ROOT/'tools/summarize_cleaning_expansion.py',
        ROOT/'tools/train_cleaning_ab.py',ROOT/'tools/eval_cleaning_ab_recovery.py',ROOT/'tools/oof_diagnostic.py',
        AUTH,AMENDMENT,OUT/'checkpoint/last.pt',OUT/'prepared.json',OUT/'protected.json',OUT/'authorization.json',
        OUT/'recovery_authorization.json',OUT/'retry_amendment.json',OUT/'baseline_initialization.json',
        OUT/'D080/original_initialization.json',OUT/'D080/prior_epochs.jsonl',OUT/'interrupted_attempt.json',
        OUT/'preflight/result.json',OUT/'inspection/result.json',OUT/'architecture/yolov8n.yaml']
    paths+=list((OUT/'manifests').glob('*.json'))+list(OUT.glob('*.yaml'))
    c.save(OUT/'seal.json',dict(at=c.stamp(),hashes={p.relative_to(ROOT).as_posix():c.sha(p) for p in paths},
        dependencies=c.ab().support().dependency_hashes(),D080_protocol_deviation='resume epoch 4 to 28'))


def final_verify(c):
    # Frozen verifier requires complete 1..28 traces and the original preflight stage shape, both retained.
    for key in ('D0','D085','D080','D070','D060','val966'):c.verify_rows(c.data_rows(key))
    c.verify_pins();c.verify_seal()
    for name,digest in c.load(OUT/'seal.json')['dependencies'].items():c.check(c.sha(name)==digest,'Dependency drift')
    for band in c.BANDS:
        t=c.load(OUT/band/'train/result.json');c.weight_path(band)
        c.check(t['epochs']==28 and t['initial_state_sha256']==c.INITIAL and c.sha(t['last'])==t['last_sha256'],'Training artifacts drift')
        trace=[json.loads(x) for x in (OUT/band/'train/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        c.check([x['epoch'] for x in trace]==list(range(1,29)),'Incomplete combined epoch coverage')
        for ref in c.REFERENCES:
            v=c.load(c.stage_path('evaluate',band,ref)/'result.json')
            c.check(v['weight_sha256']==t['best_sha256'] and v['effective_options']==c.evaluation_options()
                and v['images']==(966 if ref=='val966' else 7852),'Uniform evaluation drift')
    for role,band,ref in [('inspect',None,None),('preflight',None,None)]+[('train',b,None) for b in c.BANDS]+[('evaluate',b,r) for b,r in c.evaluation_matrix()]:
        stage=c.stage_path(role,band,ref);e=c.load(stage/'exit.json');v=c.load(stage/'result.json')
        c.check(e['code']==0 and not e['timeout'] and v['state']=='completed','Stage incomplete')
    for work in (OUT/'runtime').iterdir():
        c.check(not list(work.glob('guard_events*.jsonl')),'Guard events present')
        if not work.name.startswith('inspect-'):c.verify_child_guards(work)
    c.check(c.sha(OUT/'checkpoint/last.pt')==CHECKPOINT_HASH,'Input checkpoint drift')
    return dict(at=c.stamp(),verified=True,completed_models=3,continuation_runs=1,fresh_trainings=2,
        uniform_evaluations=12,original_attempt_unchanged=True,D080_resume=True,uninterrupted_equivalence=False)


def verify_reports(c,marker=None):
    marker=c.load(OUT/'success.json') if marker is None else marker
    c.check(marker.get('D080_resume') is True and marker.get('models')==3 and marker.get('evaluations')==12
        and marker.get('original_attempt_unchanged') is True,'Invalid success delivery scope')
    for name in REPORT_FILES:
        path=OUT/name
        c.check(path.is_file() and marker.get('files',{}).get(name)==c.sha(path),'Missing/modified report: '+name)
    protocol=c.load(OUT/'comparison.json').get('recovery_protocol',{})
    for key,expected in dict(D080_resume=True,interrupted_completed_epochs=4,additional_epochs=24,uninterrupted_equivalence=False).items():
        c.check(type(protocol.get(key)) is type(expected) and protocol[key]==expected,'Missing/incorrect recovery disclosure: '+key)
    text=(OUT/'RESULT.md').read_text(encoding='utf-8')
    c.check(all(word in text for word in ('D080','resume=true','RNG','uninterrupted'))
        and 'Three from-scratch 28-epoch YOLOv8n runs' not in text,'Recovery limitations missing from report')
    return dict(reports_verified=True,report_files=len(REPORT_FILES),D080_resume_disclosed=True)


def run_stages(c):
    c.check(not (OUT/'run_started.json').exists(),'Cannot repeat recovery stages')
    c.verify_seal();c.verify_pins()
    for key in ('D0','D085','D080','D070','D060','val966'):c.verify_rows(c.data_rows(key))
    deadline=time.perf_counter()+max(0,21600-(time.time()-c.load(OUT/'supervision_started.json')['started_unix']))
    c.save(OUT/'run_started.json',dict(at=c.stamp(),D080_resume=True,training_processes=3,evaluations=12))
    try:
        c.launch('train','D080',deadline=deadline)
        for band in ('D070','D060'):c.launch('train',band,deadline=deadline)
        for band,ref in c.evaluation_matrix():c.launch('evaluate',band,ref,deadline=deadline)
        c.save(OUT/'independent_verification.json',final_verify(c))
        summary=c.module('recovery_summary',ROOT/'tools/summarize_cleaning_expansion_resume.py');summary.OUT=OUT;summary.main()
        c.check(time.perf_counter()<deadline,'Recovery total budget exhausted')
        marker=dict(at=c.stamp(),D080_resume=True,models=3,evaluations=12,
            files={name:c.sha(OUT/name) for name in REPORT_FILES},original_attempt_unchanged=True)
        verify_reports(c,marker)
        c.check(time.perf_counter()<deadline,'Recovery budget exhausted during report verification')
        c.save(OUT/'success.json',marker)
        print('RECOVERY COMPLETE',flush=True)
    except BaseException:
        c.save(OUT/'run_failure.json',dict(at=c.stamp(),traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['inspect','prepare','preflight','seal','run','verify','_run','_worker'])
    p.add_argument('--role',choices=['inspect','preflight','train','evaluate']);p.add_argument('--band',choices=['D080','D070','D060']);p.add_argument('--reference',choices=['D0','D085','self','val966'])
    args=p.parse_args();c=configured()
    if args.action=='inspect':inspect_once(c)
    elif args.action=='_worker':worker(c,args.role,args.band,args.reference)
    elif args.action=='prepare':prepare(c)
    elif args.action=='preflight':c.launch('preflight',deadline=time.perf_counter()+300)
    elif args.action=='seal':seal(c)
    elif args.action=='run':c.run()
    elif args.action=='_run':run_stages(c)
    else:
        result=final_verify(c);result.update(verify_reports(c));print(json.dumps(result,indent=2))


BOOT_EVENTS=[]
if os.environ.get('CLEANING_EXPANSION_WORK'):
    work=Path(os.environ['CLEANING_EXPANSION_WORK']).resolve()
    # Remove the old executor marker while loading its definitions, then install our guard before third-party imports.
    marker=os.environ.pop('CLEANING_EXPANSION_WORK')
    bootstrap=configured()
    os.environ['CLEANING_EXPANSION_WORK']=marker
    bootstrap.check(work.is_relative_to(OUT/'runtime') and work==Path.cwd().resolve(),'Invalid recovery bootstrap')
    BOOT_EVENTS=bootstrap.guard(work)
    bootstrap.save(work/f'guard_ready_{os.getpid()}.json',dict(pid=os.getpid(),module=__name__,at=bootstrap.stamp()))
if __name__=='__main__':main()
