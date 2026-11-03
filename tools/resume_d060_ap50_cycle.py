"""One authorized R00 epoch-four recovery; frozen failed attempt stays intact."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
FAILED=ROOT/'reports/d060-ap50-cycle-study-20261102'
OUT=ROOT/'reports/d060-ap50-cycle-resume-r1-20261102'
BASE_TOOL=ROOT/'tools/train_d060_ap50_cycle.py'


def study():
    m=types.ModuleType('frozen_cycle');m.__file__=str(BASE_TOOL)
    exec(compile(BASE_TOOL.read_text(encoding='utf-8').split('\nBOOT_EVENTS=[]\n')[0],str(BASE_TOOL),'exec'),m.__dict__)
    m.BOOT_EVENTS=[]
    return m


def validate_prior(rows):
    if [r['epoch'] for r in rows]!=[1,2,3,4]:raise RuntimeError('Expected four original epochs')
    best=None
    for row in rows:
        score=row['fitness']
        if not math.isfinite(score) or not 0<=score<=1:raise RuntimeError('Invalid prior fitness')
        improved=best is None or score>best
        if row['best_updated']!=improved:raise RuntimeError('Prior selection trace mismatch')
        if improved:best=score


def validate_checkpoint(c,ck,rows):
    validate_prior(rows)
    c.check(isinstance(ck,dict) and ck.get('epoch')==3,'Expected epoch-four checkpoint')
    opt=ck.get('optimizer')
    c.check(isinstance(opt,dict) and bool(opt.get('state')) and bool(opt.get('param_groups')),'Optimizer state absent')
    c.check(ck.get('ema') is not None and bool(ck.get('scaler')) and isinstance(ck.get('updates'),int) and ck['updates']>0,'EMA/scaler state absent')
    c.check(ck.get('best_fitness')==max(r['fitness'] for r in rows),'Checkpoint best fitness/trace mismatch')
    metrics=ck.get('train_metrics',{})
    c.check(metrics.get('fitness')==rows[-1]['fitness'] and
        all(metrics.get(k)==v for k,v in rows[-1]['metrics'].items()),'Checkpoint final metrics/trace mismatch')
    for k,v in study().training_options('R00').items():
        actual=ck.get('train_args',{}).get(k)
        c.check((str(actual)==str(v)) if k=='device' else actual==v,'Checkpoint option drift: '+k)


def relocate(args):
    args.model=args.resume=str(OUT/'checkpoint/last.pt')
    args.data=str(OUT/'D060.yaml');args.project=str(OUT/'R00');args.name='yolov8n'
    args.save_dir=str(OUT/'R00/yolov8n')


def recovery_authorization():
    return dict(approved_by_user=True,user_message='结束这三个残留进程，释放资源，采用第四轮续训',
        R00=dict(completed_epochs=4,resume_epochs=24,target_epochs=28),R01=dict(from_scratch=True,epochs=80),
        evaluations=2,no_more_retries=True,no_test=True,no_parameter_changes=True,timeout_seconds=None,
        full_recovery_entrypoint=True,competition_approved=False)


def configured():
    m=study();m.OUT=OUT;m.SELF=SELF
    c=m.configured()
    allowed=c.allowed_file
    c.allowed_file=lambda p,w: (not w and Path(p).resolve()==BASE_TOOL.resolve()) or allowed(p,w)
    normal_train=m.train;normal_preflight=m.preflight;normal_seal=m.seal;normal_summary=m.summarize
    m.train=lambda c,run,stage:resume_train(m,c,stage) if run=='R00' else normal_train(c,run,stage)
    def preflight(c,stage):
        result=normal_preflight(c,stage)
        result['checkpoint_inspection']=inspect_checkpoint(c)
        return result
    m.preflight=preflight
    def validate(a):c.check(a==recovery_authorization(),'Recovery authorization mismatch')
    m.validate_authorization=validate
    def seal(c):
        normal_seal(c)
        # Independent extension, never amend the first seal in place.
        paths=[BASE_TOOL,ROOT/'tests/test_resume_d060_ap50_cycle.py',OUT/'checkpoint/last.pt',
            OUT/'checkpoint/best.pt',OUT/'prior_epochs.jsonl',OUT/'prior_results.csv',OUT/'original_initialization.json',
            OUT/'original_failure.json',OUT/'recovery_prepared.json']
        paths+=list(OUT.glob('*clarification.json'))
        c.save(OUT/'recovery_seal.json',dict(hashes={str(p.relative_to(ROOT).as_posix()):c.sha(p) for p in paths}))
    m.seal=seal
    original_verify=c.verify_seal
    def verify():
        original_verify()
        for p,digest in c.load(OUT/'recovery_seal.json')['hashes'].items():c.check(c.sha(ROOT/p)==digest,'Recovery source drift: '+p)
    c.verify_seal=verify
    def summary(c):
        normal_summary(c)
        t=c.load(OUT/'R00/train/result.json');original=c.load(OUT/'original_failure.json')
        with (OUT/'RESULT.md').open('a',encoding='utf-8') as f:
            f.write('\nR00 resumed after OOM at epoch5 from epoch4 last.pt; not equivalent to an uninterrupted run. '
                'Native checkpoint uses EMA-derived model weights and does not restore all RNG/loader state. '
                'R01 remains independent from scratch.\n')
            f.write('CSV time restarts for the resumed segment; it is not a continuous cumulative clock. Use the segment costs below.\n')
            f.write(f"R00 original failed worker: {original['seconds']:.3f}s; resumed worker: {t['seconds']:.3f}s; sum: {original['seconds']+t['seconds']:.3f}s (includes partial failed epoch).\n")
    m.summarize=summary
    return m,c


def inspect_checkpoint(c):
    import torch
    from ultralytics.utils.patches import torch_load
    rows=[json.loads(s) for s in (OUT/'prior_epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    ck=torch_load(OUT/'checkpoint/last.pt',map_location='cpu');validate_checkpoint(c,ck,rows)
    tensors=list(ck['ema'].state_dict().values())
    tensors += [v for state in ck['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v)]
    c.check(all(torch.isfinite(v).all() for v in tensors if torch.is_tensor(v) and v.is_floating_point()),'Nonfinite checkpoint state')
    c.check(ck['ema'].names==c.NAMES and sum(p.numel() for p in ck['ema'].parameters())==3011628,'Checkpoint architecture mismatch')
    c.check(c.sha(OUT/'checkpoint/last.pt')==c.sha(OUT/'checkpoint/best.pt'),'Epoch-four best/last should coincide')
    return dict(epoch=4,next_epoch=5,target_epochs=28,optimizer_present=True,scaler_present=True,
        ema_updates=ck['updates'],best_fitness=ck['best_fitness'],checkpoint_sha256=c.sha(OUT/'checkpoint/last.pt'),
        no_dataset_inference=True,no_optimizer_steps=True)


def resume_train(m,c,stage):
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import init_seeds
    c.local_runtime();init_seeds(42,deterministic=True)
    checkpoint=OUT/'checkpoint/last.pt';c.CHECKPOINTS[str(checkpoint.resolve())]=c.sha(checkpoint)
    rows=[json.loads(s) for s in (OUT/'prior_epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    validate_prior(rows);prior_count=len(rows)
    shutil.copyfile(OUT/'prior_epochs.jsonl',stage/'epochs.jsonl')
    baseline=c.load(OUT/'original_initialization.json')
    safety=m.trainer_factory(c,DetectionTrainer)
    class ResumeTrainer(safety):
        def check_resume(self,overrides):
            super().check_resume(overrides)
            c.check(self.resume,'Native resume disabled');relocate(self.args)
        def get_model(self,cfg=None,weights=None,verbose=True):
            c.check(weights is not None and self.resume,'Cannot fall back to scratch')
            return DetectionTrainer.get_model(self,cfg=cfg,weights=weights,verbose=verbose)
        def _load_checkpoint_state(self,ck):
            validate_checkpoint(c,ck,rows[:prior_count]);super()._load_checkpoint_state(ck)
            c.check(len(self.optimizer.state)==len(ck['optimizer']['state']) and self.ema.updates==ck['updates'],'Optimizer/EMA restore mismatch')
            c.check(self.scaler.state_dict()==ck['scaler'] and self.best_fitness==ck['best_fitness'],'Scaler/fitness restore mismatch')
    options=m.training_options('R00');options.update(model=str(checkpoint),resume=str(checkpoint),
        data=str(OUT/'D060.yaml'),project=str(OUT/'R00'),name='yolov8n',save_dir=str(OUT/'R00/yolov8n'),exist_ok=False)
    trainer=ResumeTrainer(overrides=options)
    def initialized(t):
        t.study_rows=rows
        for k,v in baseline['effective_args'].items():
            if k not in ('model','data','project','name','save_dir','resume'):
                c.check(vars(t.args).get(k)==v,'Resume effective drift: '+k)
        c.check(t.start_epoch==4 and t.epochs==28 and bool(t.amp),'Resume epochs/AMP mismatch')
        c.check(type(t.optimizer).__name__=='AdamW' and all(g['betas']==(.9,.999) for g in t.optimizer.param_groups),'Resume AdamW mismatch')
        c.ab().assert_loader(t.train_loader.dataset,c.data_rows('D060'),75634)
        c.ab().assert_loader(t.test_loader.dataset,c.data_rows('val966'),11487,ordered=False)
        # Seed native CSV and incumbent best so strict selection remains global.
        csv_path=t.save_dir/'results.csv'
        c.check(not csv_path.exists(),'Unexpected existing resume CSV')
        shutil.copyfile(OUT/'prior_results.csv',csv_path)
        t.wdir.mkdir(parents=True,exist_ok=True)
        c.check(not t.best.exists() and not t.last.exists(),'Existing recovery checkpoint preserved')
        shutil.copyfile(OUT/'checkpoint/best.pt',t.best)
        t.stopper.best_fitness=t.best_fitness;t.stopper.best_epoch=4
        c.save(stage/'initialization.json',dict(initial_state_sha256=baseline['initial_state_sha256'],
            initial_hash_provenance='Original from-scratch segment, not a new initialization',effective_args=vars(t.args),
            start_epoch=4,target_epochs=28,checkpoint_sha256=c.sha(checkpoint),restored_optimizer=True,
            restored_scaler=True,ema_updates=t.ema.updates,uninterrupted_equivalence=False))
    def start(t):t._oom_retries=3
    def done(t):
        if getattr(t,'expansion_final_eval',False):return
        row=dict(epoch=t.epoch+1,metrics={k:float(v) for k,v in t.metrics.items()},fitness=float(t.fitness),
            best_updated=bool(t.ap50_improved),lr=[g['lr'] for g in t.optimizer.param_groups],at=c.stamp(),segment='resumed')
        rows.append(row)
        with (stage/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print('RESUMED R00',row['epoch'],'/28 raw AP50',row['fitness'],flush=True)
    trainer.add_callback('on_train_start',initialized);trainer.add_callback('on_train_epoch_start',start)
    trainer.add_callback('on_fit_epoch_end',done);trainer.train()
    best=m.validate_trace('R00',rows)
    return dict(run='R00',epochs=28,prior_epochs=4,resumed_epochs=24,resume=True,best_epoch=best,
        best_native_AP50=rows[best-1]['fitness'],best=str(trainer.best),last=str(trainer.last),
        best_sha256=c.sha(trainer.best),last_sha256=c.sha(trainer.last),checkpoint_evidence=trainer.checkpoint_evidence,
        initial_state_sha256=baseline['initial_state_sha256'],uninterrupted_equivalence=False,
        extra_native_final_validation=False,prior_failed_worker_seconds=c.load(OUT/'original_failure.json')['seconds'])


def prepare(m,c):
    c.check(not OUT.exists(),'Existing recovery preserved; no retry')
    old=study().configured();old.verify_seal()
    study().verify_dependencies(old)
    rows=[json.loads(s) for s in (FAILED/'R00/train/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    validate_prior(rows)
    protected=dict(old.load(FAILED/'protected.json'))
    for p in FAILED.rglob('*'):
        if p.is_file() and p.relative_to(FAILED).parts[0] not in ('data','runtime'):
            protected[p.relative_to(ROOT).as_posix()]=c.sha(p)
    groups={k:old.data_rows(k) for k in ('D060','val966')}
    required=sum(Path(r[f]).stat().st_size for rs in groups.values() for r in rs for f in ('image','label'))
    c.check(shutil.disk_usage(ROOT).free>required+4_000_000_000,'Insufficient recovery copy reserve')
    OUT.mkdir();(OUT/'runtime').mkdir();(OUT/'checkpoint').mkdir();(OUT/'architecture').mkdir()
    for name in ('last.pt','best.pt'):shutil.copyfile(FAILED/'R00/yolov8n/weights'/name,OUT/'checkpoint'/name)
    for src,dst in [('R00/train/epochs.jsonl','prior_epochs.jsonl'),('R00/yolov8n/results.csv','prior_results.csv'),
        ('R00/train/initialization.json','original_initialization.json'),('R00/train/failure.json','original_failure.json'),
        ('architecture/yolov8n.yaml','architecture/yolov8n.yaml'),('baseline_initialization.json','baseline_initialization.json'),
        ('baseline_evaluation.json','baseline_evaluation.json'),('baseline_training.json','baseline_training.json')]:
        shutil.copyfile(FAILED/src,OUT/dst)
    for key,rs in groups.items():
        copied=[];split='train' if key=='D060' else 'val'
        for row in rs:
            new=dict(row)
            for field,folder in [('image','images'),('label','labels')]:
                src=Path(row[field]);c.check(src.resolve().is_relative_to((FAILED/'data'/key).resolve()),'Unsafe source')
                dest=OUT/'data'/key/folder/split/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
                c.check(c.sha(src)==row[field+'_sha256'],'Source drift')
                with src.open('rb') as f,dest.open('xb') as g:shutil.copyfileobj(f,g)
                c.check(c.sha(dest)==row[field+'_sha256'] and src.stat().st_ino!=dest.stat().st_ino,'Independent copy mismatch')
                protected[src.relative_to(ROOT).as_posix()]=row[field+'_sha256'];new[field]=str(dest)
            copied.append(new)
        c.save(OUT/'manifests'/(key+'.json'),copied);print('COPIED',key,len(copied),flush=True)
    import yaml
    for name in ('D060','R00-eval-val966','R01-eval-val966'):
        val=OUT/'data/val966/images/val'
        payload=dict(path=str(OUT),train=str(OUT/'data/D060/images/train' if name=='D060' else val),val=str(val),nc=4,names=c.NAMES)
        with (OUT/(name+'.yaml')).open('x',encoding='utf-8') as f:yaml.safe_dump(payload,f,sort_keys=False)
    c.save(OUT/'protected.json',protected);c.save(OUT/'authorization.json',recovery_authorization())
    training={r:m.training_options(r) for r in m.RUNS}
    training['R00'].update(resume=str(OUT/'checkpoint/last.pt'),model=str(OUT/'checkpoint/last.pt'))
    c.save(OUT/'plan.json',dict(runs=m.RUNS,training=training,
        evaluation=c.evaluation_options(),R00_resume_from_epoch=4,uninterrupted_equivalence=False,timeout_seconds=None))
    c.verify_pins()
    c.save(OUT/'recovery_prepared.json',dict(at=c.stamp(),checkpoint_sha256=c.sha(OUT/'checkpoint/last.pt'),
        train_images=7852,val_images=966,protected_files=len(protected),no_model_calls=True))
    print('RECOVERY PREPARED',flush=True)


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
    c.check(work.is_relative_to(OUT/'runtime') and work==Path.cwd().resolve(),'Invalid recovery worker bootstrap')
    # Every configured instance needs the same process-local guard-event list.
    EVENTS=c.guard(work)
    c.save(work/f'guard_ready_{os.getpid()}.json',dict(pid=os.getpid(),module=__name__,at=c.stamp()))
if __name__=='__main__':main()
