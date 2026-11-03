"""One approved R01 epoch63 recovery, reuse finished R00, two FP32 evaluations."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import traceback
import types

ROOT=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
BASE_TOOL=ROOT/'tools/train_d060_ap50_cycle.py'
PREVIOUS=ROOT/'reports/d060-ap50-cycle-resume-r1-20261102'
OUT=ROOT/'reports/d060-ap50-cycle-resume-r2-20261103'


def base():
    m=types.ModuleType('epoch63_frozen_cycle');m.__file__=str(BASE_TOOL)
    exec(compile(BASE_TOOL.read_text(encoding='utf-8').split('\nBOOT_EVENTS=[]\n')[0],str(BASE_TOOL),'exec'),m.__dict__)
    m.BOOT_EVENTS=[]
    return m


def stages():return [('train','R01'),('evaluate','R00'),('evaluate','R01')]


def authorization():
    return dict(approved_by_user=True,user_message='续训至80轮，完成原定两次验证',
        resume_from_epoch=63,remaining_epochs=17,target_epochs=80,reuse_completed_R00=True,
        evaluations=2,no_parameter_changes=True,no_retry=True,no_test=True,timeout_seconds=None,
        full_recovery_entrypoint=True,competition_approved=False)


def validate_checkpoint(c,ck,rows):
    c.check([r['epoch'] for r in rows]==list(range(1,64)),'Expected exactly 63 original epochs')
    # Reuse full-horizon strict-selection checker on a temporary horizon, without changing globals.
    best=None
    for r in rows:
        score=r['fitness'];c.check(math.isfinite(score) and 0<=score<=1,'Invalid prior fitness')
        improved=best is None or score>best
        c.check(r['best_updated']==improved,'Prior best selection mismatch')
        if improved:best=score
    c.check(isinstance(ck,dict) and ck.get('epoch')==62,'Expected epoch63 checkpoint')
    opt=ck.get('optimizer')
    c.check(isinstance(opt,dict) and bool(opt.get('state')) and bool(opt.get('param_groups')),'Missing optimizer')
    c.check(ck.get('ema') is not None and bool(ck.get('scaler')) and isinstance(ck.get('updates'),int) and ck['updates']>0,'Missing EMA/scaler')
    c.check(ck.get('best_fitness')==best,'Saved best/trace mismatch')
    metrics=ck.get('train_metrics',{})
    c.check(metrics.get('fitness')==rows[-1]['fitness'] and all(metrics.get(k)==v for k,v in rows[-1]['metrics'].items()),'Saved metrics/trace mismatch')
    for k,v in base().training_options('R01').items():
        value=ck.get('train_args',{}).get(k)
        c.check(str(value)==str(v) if k=='device' else value==v,'Checkpoint parameter drift: '+k)


def configured():
    m=base();m.OUT=OUT;m.SELF=SELF;c=m.configured()
    allowed=c.allowed_file
    c.allowed_file=lambda p,w:(not w and Path(p).resolve()==BASE_TOOL.resolve()) or allowed(p,w)
    normal_preflight=m.preflight;normal_seal=m.seal;normal_summary=m.summarize;normal_verify=c.verify_seal
    m.validate_authorization=lambda a:c.check(a==authorization(),'Epoch63 authorization mismatch')
    m.train=lambda c,run,stage:resume_train(m,c,stage) if run=='R01' else c.check(False,'R00 retraining prohibited')
    def preflight(c,stage):
        r=normal_preflight(c,stage);r['checkpoint_inspection']=inspect_checkpoint(m,c)
        return r
    m.preflight=preflight
    def seal(c):
        normal_seal(c)
        paths=[BASE_TOOL,ROOT/'tools/resume_d060_ap50_cycle.py',ROOT/'tests/test_resume_d060_ap50_r01.py',
            OUT/'prior_epochs.jsonl',OUT/'prior_results.csv',OUT/'original_initialization.json',
            OUT/'checkpoint/last.pt',OUT/'checkpoint/best.pt',OUT/'recovery_prepared.json',OUT/'prior_costs.json']
        paths+=list((OUT/'R00').rglob('*'))
        c.save(OUT/'recovery_seal.json',dict(hashes={p.relative_to(ROOT).as_posix():c.sha(p) for p in paths if p.is_file()}))
    m.seal=seal
    def verify():
        normal_verify()
        for p,d in c.load(OUT/'recovery_seal.json')['hashes'].items():c.check(c.sha(ROOT/p)==d,'Recovery seal drift: '+p)
    c.verify_seal=verify
    def summary(c):
        normal_summary(c);cost=c.load(OUT/'prior_costs.json');t=c.load(OUT/'R01/train/result.json')
        with (OUT/'RESULT.md').open('a',encoding='utf-8') as f:
            f.write('\nBoth runs were interrupted and resumed: R00 after epoch4 OOM, R01 after epoch63 service restart. '
                'Not equivalent to uninterrupted runs; native resume uses EMA-derived weights and does not restore all RNG/loader state. '
                'R00 artifacts were reused without retraining. CSV time is segment-local, not continuous cumulative time.\n')
            f.write(f"R00 workers including failed segment: {cost['R00_worker_seconds_including_failure']:.3f}s. "
                f"R01 pre-interruption CSV training time: {cost['R01_prior_csv_seconds']:.3f}s (not full worker cost); "
                f"new recovery worker: {t['seconds']:.3f}s. No unobserved time is fabricated.\n")
    m.summarize=summary;m.run_stages=lambda c:run_stages(m,c)
    return m,c


def inspect_checkpoint(m,c):
    import torch
    from ultralytics.utils.patches import torch_load
    rows=[json.loads(s) for s in (OUT/'prior_epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    last=torch_load(OUT/'checkpoint/last.pt',map_location='cpu');validate_checkpoint(c,last,rows)
    best=torch_load(OUT/'checkpoint/best.pt',map_location='cpu')
    selected=max(rows,key=lambda r:r['fitness'])
    c.check(best.get('epoch')==selected['epoch']-1 and best.get('best_fitness')==selected['fitness'] and
        best.get('train_metrics',{}).get('fitness')==selected['fitness'] and
        all(best['train_metrics'].get(k)==v for k,v in selected['metrics'].items()),'Incumbent best/trace mismatch')
    for ck in (last,best):
        net=ck['ema'];c.check(net.names==c.NAMES and sum(p.numel() for p in net.parameters())==3011628,'Architecture drift')
        tensors=list(net.state_dict().values())+[v for state in ck['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v)]
        c.check(all(torch.isfinite(v).all() for v in tensors if torch.is_tensor(v) and v.is_floating_point()),'Nonfinite checkpoint')
    return dict(completed_epochs=63,next_epoch=64,target_epochs=80,best_epoch=selected['epoch'],
        best_AP50=selected['fitness'],ema_updates=last['updates'],optimizer_present=True,scaler_present=True,
        last_sha256=c.sha(OUT/'checkpoint/last.pt'),best_sha256=c.sha(OUT/'checkpoint/best.pt'),
        no_dataset_inference=True,no_optimizer_steps=True)


def resume_train(m,c,stage):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import init_seeds
    c.local_runtime();init_seeds(42,deterministic=True)
    checkpoint=OUT/'checkpoint/last.pt';c.CHECKPOINTS[str(checkpoint.resolve())]=c.sha(checkpoint)
    rows=[json.loads(s) for s in (OUT/'prior_epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    shutil.copyfile(OUT/'prior_epochs.jsonl',stage/'epochs.jsonl')
    baseline=c.load(OUT/'original_initialization.json');best_row=max(rows,key=lambda r:r['fitness'])
    safety=m.trainer_factory(c,DetectionTrainer)
    class ResumeTrainer(safety):
        def check_resume(self,overrides):
            super().check_resume(overrides);c.check(self.resume,'Native resume disabled')
            self.args.model=self.args.resume=str(checkpoint)
            self.args.data=str(OUT/'D060.yaml');self.args.project=str(OUT/'R01');self.args.name='yolov8n'
            self.args.save_dir=str(OUT/'R01/yolov8n')
        def get_model(self,cfg=None,weights=None,verbose=True):
            c.check(weights is not None and self.resume,'Cannot resume from scratch')
            return DetectionTrainer.get_model(self,cfg=cfg,weights=weights,verbose=verbose)
        def _load_checkpoint_state(self,ck):
            validate_checkpoint(c,ck,rows[:63]);super()._load_checkpoint_state(ck)
            c.check(len(self.optimizer.state)==len(ck['optimizer']['state']) and self.ema.updates==ck['updates'],'Optimizer/EMA restoration mismatch')
            c.check(self.scaler.state_dict()==ck['scaler'] and self.best_fitness==ck['best_fitness'],'Scaler/fitness restoration mismatch')
    options=m.training_options('R01');options.update(model=str(checkpoint),resume=str(checkpoint),
        data=str(OUT/'D060.yaml'),project=str(OUT/'R01'),name='yolov8n',save_dir=str(OUT/'R01/yolov8n'),exist_ok=False)
    t=ResumeTrainer(overrides=options)
    def initialized(t):
        t.study_rows=rows
        for k,v in baseline['effective_args'].items():
            if k not in ('model','data','project','name','save_dir','resume'):c.check(vars(t.args).get(k)==v,'Resume option drift: '+k)
        c.check(t.start_epoch==63 and t.epochs==80 and bool(t.amp),'Resume epoch/AMP drift')
        c.check(type(t.optimizer).__name__=='AdamW' and all(g['betas']==(.9,.999) for g in t.optimizer.param_groups),'Optimizer drift')
        c.ab().assert_loader(t.train_loader.dataset,c.data_rows('D060'),75634)
        c.ab().assert_loader(t.test_loader.dataset,c.data_rows('val966'),11487,ordered=False)
        c.check(not t.csv.exists() and not t.best.exists(),'Existing recovery artifacts preserved')
        shutil.copyfile(OUT/'prior_results.csv',t.csv);t.wdir.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(OUT/'checkpoint/best.pt',t.best)
        t.stopper.best_fitness=t.best_fitness;t.stopper.best_epoch=best_row['epoch']
        c.save(stage/'initialization.json',dict(initial_state_sha256=baseline['initial_state_sha256'],
            initial_hash_provenance='Original scratch R01 segment, not newly initialized',effective_args=vars(t.args),
            start_epoch=63,target_epochs=80,restored_optimizer=True,restored_scaler=True,ema_updates=t.ema.updates,
            checkpoint_sha256=c.sha(checkpoint),uninterrupted_equivalence=False))
    def start(t):t._oom_retries=3
    def done(t):
        if getattr(t,'expansion_final_eval',False):return
        row=dict(epoch=t.epoch+1,metrics={k:float(v) for k,v in t.metrics.items()},fitness=float(t.fitness),
            best_updated=bool(t.ap50_improved),lr=[g['lr'] for g in t.optimizer.param_groups],at=c.stamp(),segment='resumed_after_epoch63')
        rows.append(row)
        with (stage/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print('RESUMED R01',row['epoch'],'/80 AP50',row['fitness'],flush=True)
    t.add_callback('on_train_start',initialized);t.add_callback('on_train_epoch_start',start);t.add_callback('on_fit_epoch_end',done)
    t.train();best=m.validate_trace('R01',rows)
    return dict(run='R01',epochs=80,prior_epochs=63,resumed_epochs=17,resume=True,best_epoch=best,
        best_native_AP50=rows[best-1]['fitness'],best=str(t.best),last=str(t.last),best_sha256=c.sha(t.best),last_sha256=c.sha(t.last),
        checkpoint_evidence=t.checkpoint_evidence,initial_state_sha256=baseline['initial_state_sha256'],
        uninterrupted_equivalence=False,extra_native_final_validation=False)


def prepare(m,c):
    c.check(not OUT.exists(),'Existing recovery preserved; no retry')
    old=base();old.OUT=PREVIOUS;oldc=old.configured();oldc.verify_seal();old.verify_dependencies(oldc)
    for rel,d in c.load(PREVIOUS/'recovery_seal.json')['hashes'].items():c.check(c.sha(ROOT/rel)==d,'Previous recovery seal drift: '+rel)
    c.check(not (PREVIOUS/'R00/eval_val966').exists() and not (PREVIOUS/'R01/eval_val966').exists(),'Original evaluations must not be repeated')
    r00=c.load(PREVIOUS/'R00/train/result.json')
    c.check(r00['state']=='completed' and r00['epochs']==28,'R00 not complete')
    oldc.weight_path('R00');c.check(c.sha(r00['last'])==r00['last_sha256'],'R00 last drift')
    rows=[json.loads(s) for s in (PREVIOUS/'R01/train/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    c.check([r['epoch'] for r in rows]==list(range(1,64)),'Prior R01 coverage drift')
    protected=dict(c.load(PREVIOUS/'protected.json'))
    for p in PREVIOUS.rglob('*'):
        if p.is_file() and p.relative_to(PREVIOUS).parts[0] not in ('data','runtime'):protected[p.relative_to(ROOT).as_posix()]=c.sha(p)
    groups={k:c.load(PREVIOUS/'manifests'/(k+'.json')) for k in ('D060','val966')}
    required=sum(Path(r[f]).stat().st_size for rs in groups.values() for r in rs for f in ('image','label'))
    c.check(shutil.disk_usage(ROOT).free>required+4_000_000_000,'Insufficient reserve')
    OUT.mkdir();(OUT/'runtime').mkdir();(OUT/'checkpoint').mkdir();(OUT/'architecture').mkdir()
    for name in ('best.pt','last.pt'):shutil.copyfile(PREVIOUS/'R01/yolov8n/weights'/name,OUT/'checkpoint'/name)
    for src,dst in [('R01/train/epochs.jsonl','prior_epochs.jsonl'),('R01/yolov8n/results.csv','prior_results.csv'),
        ('R01/train/initialization.json','original_initialization.json'),('architecture/yolov8n.yaml','architecture/yolov8n.yaml'),
        ('baseline_initialization.json','baseline_initialization.json'),('baseline_evaluation.json','baseline_evaluation.json'),
        ('baseline_training.json','baseline_training.json')]:shutil.copyfile(PREVIOUS/src,OUT/dst)
    # Reuse completed R00 with independent weight copies and explicit result provenance.
    for name in ('best.pt','last.pt'):
        dest=OUT/'R00/yolov8n/weights'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(r00['best' if name=='best.pt' else 'last']),dest)
    reused=dict(r00,reused_without_training=True,source_training_result=str(PREVIOUS/'R00/train/result.json'),
        best=str(OUT/'R00/yolov8n/weights/best.pt'),last=str(OUT/'R00/yolov8n/weights/last.pt'))
    c.save(OUT/'R00/train/result.json',reused)
    for name in ('epochs.jsonl','initialization.json','exit.json'):shutil.copyfile(PREVIOUS/'R00/train'/name,OUT/'R00/train'/name)
    for key,rs in groups.items():
        copied=[];split='train' if key=='D060' else 'val'
        c.check(len(rs)==(7852 if key=='D060' else 966),'Data coverage drift')
        for row in rs:
            new=dict(row)
            for field,folder in [('image','images'),('label','labels')]:
                src=Path(row[field]);c.check(src.resolve().is_relative_to((PREVIOUS/'data'/key).resolve()),'Unsafe source')
                dst=OUT/'data'/key/folder/split/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
                c.check(c.sha(src)==row[field+'_sha256'],'Data source drift')
                with src.open('rb') as f,dst.open('xb') as g:shutil.copyfileobj(f,g)
                c.check(c.sha(dst)==row[field+'_sha256'] and src.stat().st_ino!=dst.stat().st_ino,'Independent copy mismatch')
                protected[src.relative_to(ROOT).as_posix()]=row[field+'_sha256'];new[field]=str(dst)
            copied.append(new)
        c.save(OUT/'manifests'/(key+'.json'),copied);print('COPIED',key,len(copied),flush=True)
    import csv,yaml
    for name in ('D060','R00-eval-val966','R01-eval-val966'):
        val=OUT/'data/val966/images/val'
        with (OUT/(name+'.yaml')).open('x',encoding='utf-8') as f:yaml.safe_dump(dict(path=str(OUT),
            train=str(OUT/'data/D060/images/train' if name=='D060' else val),val=str(val),nc=4,names=c.NAMES),f,sort_keys=False)
    with (OUT/'prior_results.csv').open(encoding='utf-8') as f:csvrows=list(csv.DictReader(f))
    c.check([int(r['epoch']) for r in csvrows]==list(range(1,64)),'Prior CSV coverage mismatch')
    c.save(OUT/'prior_costs.json',dict(R00_worker_seconds_including_failure=r00['seconds']+r00['prior_failed_worker_seconds'],
        R01_prior_csv_seconds=float(csvrows[-1]['time'])))
    c.save(OUT/'protected.json',protected);c.save(OUT/'authorization.json',authorization())
    options=m.training_options('R01');options.update(resume=str(OUT/'checkpoint/last.pt'),model=str(OUT/'checkpoint/last.pt'))
    c.save(OUT/'plan.json',dict(stages=stages(),R00='reuse completed result, no retraining',R01=options,
        evaluation=c.evaluation_options(),uninterrupted_equivalence=False,csv_time='segment-local',timeout_seconds=None))
    c.verify_pins()
    c.save(OUT/'recovery_prepared.json',dict(at=c.stamp(),last_sha256=c.sha(OUT/'checkpoint/last.pt'),
        best_sha256=c.sha(OUT/'checkpoint/best.pt'),protected_files=len(protected),no_model_calls=True))
    print('EPOCH63 RECOVERY PREPARED',flush=True)


def run_stages(m,c):
    c.check(not (OUT/'run_started.json').exists(),'Cannot repeat recovery')
    c.verify_seal();c.verify_pins();m.verify_dependencies(c)
    for key in ('D060','val966'):c.verify_rows(c.data_rows(key))
    c.save(OUT/'run_started.json',dict(at=c.stamp(),stages=stages(),reused_R00=True))
    try:
        for role,run in stages():m.launch(c,role,run)
        c.save(OUT/'independent_verification.json',dict(m.final_verify(c),R00_reused=True,R01_resumed_from_epoch=63,
            newly_executed_training_segments=1,newly_executed_evaluations=2))
        m.summarize(c)
        files=('independent_verification.json','comparison.json','metrics.csv','per_class.csv','RESULT.md')
        c.save(OUT/'success.json',dict(at=c.stamp(),new_training_segments=1,evaluations=2,R00_reused=True,
            files={n:c.sha(OUT/n) for n in files}))
        print('COMPLETE: R01 through80 + two original FP32 evaluations',flush=True)
    except BaseException:
        c.save(OUT/'run_failure.json',dict(at=c.stamp(),traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','preflight','seal','run','verify','_run','_worker'))
    p.add_argument('--role',choices=('preflight','train','evaluate'));p.add_argument('--run',choices=('R00','R01'))
    a=p.parse_args();m,c=configured()
    if a.action=='prepare':prepare(m,c)
    elif a.action=='preflight':m.launch(c,'preflight')
    elif a.action=='seal':m.seal(c)
    elif a.action=='run':m.run(c)
    elif a.action=='verify':print(json.dumps(m.final_verify(c),indent=2))
    elif a.action=='_run':m.run_stages(c)
    else:m.worker(c,a.role,a.run)


if os.environ.get('D060_AP50_WORK'):
    work=Path(os.environ['D060_AP50_WORK']).resolve();m,c=configured()
    c.check(work.is_relative_to(OUT/'runtime') and work==Path.cwd().resolve(),'Invalid epoch63 bootstrap')
    EVENTS=c.guard(work)
    c.save(work/f'guard_ready_{os.getpid()}.json',dict(pid=os.getpid(),module=__name__,at=c.stamp()))
if __name__=='__main__':main()
