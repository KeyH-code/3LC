"""Approved D060 R00/28 and R01/80 from-scratch study, two FP32 val evaluations."""
import argparse
import csv
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

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
SOURCE = ROOT/'reports/train-cleaning-expansion-resume-r1-20261102'
OUT = ROOT/'reports/d060-ap50-cycle-study-20261102'
RUNS = {'R00':28, 'R01':80}
SUPPORT = ('tools/train_cleaning_expansion.py','tools/train_cleaning_ab.py',
           'tools/eval_cleaning_ab_recovery.py','tools/oof_diagnostic.py')


def core():
    path=ROOT/SUPPORT[0];m=types.ModuleType('cycle_frozen_support');m.__file__=str(path)
    exec(compile(path.read_text(encoding='utf-8').split('\nBOOT_EVENTS=[]\n')[0],str(path),'exec'),m.__dict__)
    m.BOOT_EVENTS=[]
    return m


def training_options(run):
    if run not in RUNS:raise RuntimeError('Unknown approved run')
    options=core().training_options()
    options.update(epochs=RUNS[run],optimizer='AdamW',lr0=.00125,momentum=.9,warmup_bias_lr=0.)
    return options


def authorization():
    return dict(approved_by_user=True,
        user_message='可以采用这个方案，你现在可以完成训练入口的代码编写，并在编写好入口代码之后，开始训练',
        training_runs=2,evaluations=2,runs=RUNS,resume=False,full_entrypoint=True,
        no_retry=True,no_test=True,no_network_or_3lc=True,total_timeout_seconds=None,
        best_selection='unrounded native val AP50; strict improvement; earlier tie',
        independent_offline=True,competition_approved=False)


def validate_authorization(actual):
    if actual!=authorization():raise RuntimeError('Authorization scope mismatch')


def selection_factory(base):
    """Reuse native validation/save serialization, replacing only selection semantics."""
    class AP50Selection(base):
        def validate(self):
            previous=self.best_fitness
            metrics,_=super().validate()
            if metrics is None:raise RuntimeError('Every epoch requires validation')
            score=float(self.validator.metrics.box.map50)
            if not math.isfinite(score) or not 0<=score<=1:raise RuntimeError('Invalid raw AP50')
            self.ap50_improved=previous is None or score>previous
            self.best_fitness=score if self.ap50_improved else previous
            metrics['metrics/mAP50(B)']=score
            return metrics,score

        def save_model(self):
            if self.ap50_improved:return super().save_model()
            # Native saves best on equality. Redirect only that write, not fitness,
            # preserving correct serialized best_fitness and the original best bytes.
            original=self.best
            discard=original.with_name('unselected-tie.pt')
            if discard.exists():raise RuntimeError('Unexpected discarded checkpoint exists')
            self.best=discard
            try:return super().save_model()
            finally:
                self.best=original
                if discard.exists():discard.unlink()
    return AP50Selection


def validate_trace(run,rows):
    if [r['epoch'] for r in rows]!=list(range(1,RUNS[run]+1)):raise RuntimeError('Epoch coverage mismatch')
    best=None;best_epoch=None
    for row in rows:
        score=row['fitness']
        if not math.isfinite(score) or not 0<=score<=1:raise RuntimeError('Invalid trace score')
        improved=best is None or score>best
        if row['best_updated']!=improved:raise RuntimeError('Strict AP50 selection trace mismatch')
        if improved:best=score;best_epoch=row['epoch']
    return best_epoch


def verify_dependencies(c,historical=False):
    path=SOURCE/'seal.json' if historical else OUT/'seal.json'
    for p,digest in c.load(path)['dependencies'].items():
        c.check(c.sha(p)==digest,'Dependency drift: '+p)


def validate_stripped(payload,expected):
    if not isinstance(payload,dict) or payload.get('model') is None:
        raise RuntimeError('Checkpoint strip failed')
    if payload.get('epoch')!=-1 or any(payload.get(k) is not None for k in ('optimizer','ema','scaler','updates','best_fitness')):
        raise RuntimeError('Checkpoint strip incomplete')
    actual=payload.get('train_metrics',{})
    if actual.get('fitness')!=expected['fitness'] or any(actual.get(k)!=v for k,v in expected['metrics'].items()):
        raise RuntimeError('Checkpoint metrics/trace mismatch')
    results=payload.get('train_results',{})
    epochs=results.get('epoch',[])
    if epochs!=list(range(1,expected['epoch']+1)):
        raise RuntimeError('Checkpoint epoch/trace mismatch')
    # Native read_results_csv retains five-decimal CSV display precision.
    for k,v in expected['metrics'].items():
        values=results.get(k,[])
        if len(values)!=expected['epoch'] or not math.isclose(values[-1],v,rel_tol=1e-4,abs_tol=1e-5):
            raise RuntimeError('Checkpoint CSV/trace mismatch')


def checkpoint_evidence(c,payload,expected):
    import torch
    validate_stripped(payload,expected);model=payload['model']
    c.check(model.names==c.NAMES and sum(p.numel() for p in model.parameters())==3011628,'Checkpoint model identity mismatch')
    c.check(all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()),'Nonfinite checkpoint weights')
    return dict(epoch=expected['epoch'],fitness=expected['fitness'],metrics=payload['train_metrics'],
        model_state_sha256=c.ab().state_hash(model),finite_weights=True,optimizer_stripped=True)


def configured():
    c=core();c.OUT=OUT;c.SELF=SELF;c.BANDS=tuple(RUNS);c.REFERENCES=('val966',)
    old_allowed=c.allowed_file
    c.allowed_file=lambda p,w: (not w and Path(p).resolve()==(ROOT/SUPPORT[0]).resolve()) or old_allowed(p,w)
    def weight_path(run):
        c.check(run in RUNS,'Unknown run')
        r=c.load(OUT/run/'train/result.json');p=Path(r['best']).resolve()
        c.check(r['state']=='completed' and r['epochs']==RUNS[run],'Incomplete training')
        c.check(p.is_relative_to((OUT/run).resolve()) and p.suffix=='.pt','Unsafe actual best')
        c.check(c.sha(p)==r['best_sha256'],'Best hash drift')
        return p
    c.weight_path=weight_path
    return c


def assert_options(c,run,t):
    expected=dict(c.load(OUT/'baseline_initialization.json')['effective_args'])
    expected.update(training_options(run))
    for key,value in expected.items():
        if key in ('model','data','project','name','save_dir'):continue
        actual=vars(t.args).get(key)
        if key=='device':actual=str(actual);value=str(value)
        c.check(actual==value,'Effective training drift: '+key)
    c.check(type(t.optimizer).__name__=='AdamW' and bool(t.amp),'Optimizer/AMP mismatch')
    for group in t.optimizer.param_groups:
        c.check(group['lr']==.00125 and group['betas']==(.9,.999),'AdamW rate/betas mismatch')
    c.check(t.start_epoch==0 and t.epochs==RUNS[run],'From-scratch/epoch mismatch')


def trainer_factory(c,base):
    safety=c.trainer_factory(base)
    class StudyTrainer(selection_factory(safety)):
        def final_eval(self):
            # No extra dataset inference: exactly one standalone FP32 evaluation/run.
            self.expansion_final_eval=True
            import ultralytics.engine.trainer as engine
            def strip_only():
                self.checkpoint_evidence={}
                rows=self.study_rows
                selected=next(r for r in rows if r['fitness']==self.best_fitness)
                for key,path,row in [('last',self.last,rows[-1]),('best',self.best,selected)]:
                    payload=engine.strip_optimizer(path)
                    self.checkpoint_evidence[key]=checkpoint_evidence(c,payload,row)
            return c.native_final_eval(self,strip_only,engine)
    return StudyTrainer


def train(c,run,stage):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import init_seeds
    c.local_runtime();init_seeds(42,deterministic=True)
    model=YOLO(str(OUT/'architecture/yolov8n.yaml'),task='detect')
    c.check(not model.ckpt and not model.ckpt_path,'No pretrained checkpoint allowed')
    rows=[];meta={}
    def initialized(t):
        t.study_rows=rows
        assert_options(c,run,t)
        c.ab().assert_loader(t.train_loader.dataset,c.data_rows('D060'),75634)
        c.ab().assert_loader(t.test_loader.dataset,c.data_rows('val966'),11487,ordered=False)
        meta.update(initial_state_sha256=c.initial_contract(t.model),effective_args=vars(t.args),
            optimizer=type(t.optimizer).__name__,parameter_groups=[dict(lr=g['lr'],betas=g['betas'],
                weight_decay=g['weight_decay'],params=len(g['params'])) for g in t.optimizer.param_groups],
            amp=bool(t.amp),parameters=3011628,train_images=7852,train_boxes=75634,val_images=966,val_boxes=11487)
        c.save(stage/'initialization.json',meta)
    def epoch_start(t):t._oom_retries=3
    def epoch_end(t):
        if getattr(t,'expansion_final_eval',False):return
        row=dict(epoch=t.epoch+1,metrics={k:float(v) for k,v in t.metrics.items()},fitness=float(t.fitness),
            best_updated=bool(t.ap50_improved),lr=[g['lr'] for g in t.optimizer.param_groups],at=c.stamp())
        rows.append(row)
        with (stage/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print('EPOCH',run,row['epoch'],'/',RUNS[run],'raw AP50',row['fitness'],'best updated',row['best_updated'],flush=True)
    model.add_callback('on_train_start',initialized)
    model.add_callback('on_train_epoch_start',epoch_start);model.add_callback('on_fit_epoch_end',epoch_end)
    model.train(trainer=trainer_factory(c,DetectionTrainer),data=str(OUT/'D060.yaml'),
        project=str(OUT/run),name='yolov8n',exist_ok=False,**training_options(run))
    best_epoch=validate_trace(run,rows);t=model.trainer
    return dict(run=run,epochs=RUNS[run],best_epoch=best_epoch,best_native_AP50=rows[best_epoch-1]['fitness'],
        best=str(t.best),last=str(t.last),best_sha256=c.sha(t.best),last_sha256=c.sha(t.last),
        checkpoint_evidence=t.checkpoint_evidence,
        initial_state_sha256=meta['initial_state_sha256'],resume=False,extra_native_final_validation=False)


def preflight(c,stage):
    import torch,ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import init_seeds
    c.local_runtime();native=c.ab().preflight(stage)
    init_seeds(42,deterministic=True)
    y=YOLO(str(OUT/'architecture/yolov8n.yaml'),task='detect')
    checks={}
    for run in RUNS:
        # Same initialization path as native Model.train; no forward/dataset inference.
        init_seeds(0,deterministic=True)
        t=trainer_factory(c,DetectionTrainer)(overrides=dict(model=str(OUT/'architecture/yolov8n.yaml'),
            data=str(OUT/'D060.yaml'),project=str(stage),name=run,exist_ok=False,**training_options(run)))
        t.model=t.get_model(cfg=y.model.yaml,weights=None,verbose=False);t.model.names=c.NAMES
        initial=c.initial_contract(t.model)
        t.optimizer=t.build_optimizer(t.model,name='AdamW',lr=.00125,momentum=.9,decay=.0005,iterations=123*RUNS[run])
        checks[run]=dict(initial_state_sha256=initial,optimizer=type(t.optimizer).__name__,
            groups=[dict(lr=g['lr'],betas=g['betas']) for g in t.optimizer.param_groups])
        c.check(all(g['lr']==.00125 and g['betas']==(.9,.999) for g in t.optimizer.param_groups),'Preflight AdamW mismatch')
    native['no_model_construction']=False  # Matcher-only substage did not construct; full preflight did.
    return dict(native,initialization_and_optimizer=checks,no_dataset_inference=True,no_optimizer_steps=True,
        runtime=dict(python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))


def prepare(c):
    c.check(not OUT.exists(),'Existing run preserved; no prepare/retry')
    verify_dependencies(c,historical=True)
    # Bind trusted source manifests/configs/support to the historical execution seal.
    historical=c.load(SOURCE/'seal.json')['hashes']
    for rel in (*SUPPORT[1:],'tools/train_cleaning_expansion.py',
                (SOURCE/'manifests/D060.json').relative_to(ROOT).as_posix(),
                (SOURCE/'manifests/val966.json').relative_to(ROOT).as_posix(),
                (SOURCE/'architecture/yolov8n.yaml').relative_to(ROOT).as_posix()):
        c.check(c.sha(ROOT/rel)==historical[rel],'Frozen source drift: '+rel)
    groups={key:c.load(SOURCE/'manifests'/(key+'.json')) for key in ('D060','val966')}
    for key,rows in groups.items():
        c.check(len(rows)==(7852 if key=='D060' else 966),'Source coverage mismatch')
        for r in rows:
            for field in ('image','label'):c.check(Path(r[field]).resolve().is_relative_to((SOURCE/'data'/key).resolve()),'Source escape')
        c.verify_rows(rows)
    c.check(not {r['image_sha256'] for r in groups['D060']} & {r['image_sha256'] for r in groups['val966']},'Split overlap')
    required=sum(Path(r[k]).stat().st_size for rs in groups.values() for r in rs for k in ('image','label'))
    c.check(shutil.disk_usage(ROOT).free>required+4_000_000_000,'Insufficient copy/checkpoint reserve')
    protected=dict(c.load(SOURCE/'protected.json'))
    # Preserve all prior non-runtime/non-dataset evidence, including selected weights.
    for p in SOURCE.rglob('*'):
        if p.is_file() and p.relative_to(SOURCE).parts[0] not in ('runtime','data'):
            protected[p.relative_to(ROOT).as_posix()]=c.sha(p)
    OUT.mkdir();(OUT/'runtime').mkdir();(OUT/'architecture').mkdir()
    shutil.copyfile(SOURCE/'architecture/yolov8n.yaml',OUT/'architecture/yolov8n.yaml')
    for key,rows in groups.items():
        copied=[];split='train' if key=='D060' else 'val'
        for row in sorted(rows,key=lambda r:Path(r['image']).name):
            new=dict(row)
            for field,folder in [('image','images'),('label','labels')]:
                src=Path(row[field]);dest=OUT/'data'/key/folder/split/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
                with src.open('rb') as f,dest.open('xb') as g:shutil.copyfileobj(f,g)
                c.check(c.sha(dest)==row[field+'_sha256'] and src.stat().st_ino!=dest.stat().st_ino,'Independent copy mismatch')
                protected[src.relative_to(ROOT).as_posix()]=row[field+'_sha256'];new[field]=str(dest)
            copied.append(new)
        c.save(OUT/'manifests'/(key+'.json'),copied)
    import yaml
    for name in ('D060', 'R00-eval-val966','R01-eval-val966'):
        val=OUT/'data/val966/images/val'
        paths=dict(path=str(OUT),train=str(OUT/'data/D060/images/train' if name=='D060' else val),val=str(val),nc=4,names=c.NAMES)
        with (OUT/(name+'.yaml')).open('x',encoding='utf-8') as f:yaml.safe_dump(paths,f,sort_keys=False)
    c.save(OUT/'protected.json',protected)
    c.save(OUT/'baseline_initialization.json',c.load(SOURCE/'D060/train/initialization.json'))
    c.save(OUT/'baseline_evaluation.json',c.load(SOURCE/'D060/eval_val966/result.json'))
    c.save(OUT/'baseline_training.json',c.load(SOURCE/'D060/train/result.json'))
    c.save(OUT/'authorization.json',authorization())
    c.save(OUT/'plan.json',dict(runs=RUNS,training={r:training_options(r) for r in RUNS},
        evaluation=c.evaluation_options(),copy_bytes=required,initial_state_sha256=c.INITIAL,
        timeout_seconds=None,extra_native_final_validation=False))
    c.verify_pins();print('PREPARED D0607852/val966 independent copies, no model calls',flush=True)


def worker(c,role,run):
    stage=c.stage_path(role,run,'val966');start=time.perf_counter()
    try:
        if role!='preflight':verify_dependencies(c)
        import torch,ultralytics
        from ultralytics.utils import ONLINE,SETTINGS
        c.check(sys.version_info[:3]==(3,12,15) and torch.__version__=='2.14.1+cu130' and
            ultralytics.__version__=='8.4.165' and torch.version.cuda=='13.0' and not ONLINE and not SETTINGS['sync'],
            'Frozen offline runtime mismatch')
        result=preflight(c,stage) if role=='preflight' else (train(c,run,stage) if role=='train' else c.evaluate(run,'val966',stage))
        c.check(not BOOT_EVENTS,'Worker guard event')
        result.update(state='completed',seconds=time.perf_counter()-start,at=c.stamp(),
            guarded_spawn_children=c.verify_child_guards(Path.cwd()),
            no_3lc_import=not any(k.split('.')[0] in ('tlc','tlc_ultralytics','tlcconfig') for k in sys.modules))
        c.save(stage/'result.json',result)
    except BaseException:
        c.save(stage/'failure.json',dict(at=c.stamp(),seconds=time.perf_counter()-start,traceback=traceback.format_exc()));raise


def launch(c,role,run=None):
    c.check(role in ('preflight','train','evaluate') and (run in RUNS if role!='preflight' else run is None),'Stage scope mismatch')
    if role=='preflight':verify_dependencies(c,historical=True)
    else:c.verify_seal();verify_dependencies(c)
    stage=c.stage_path(role,run,'val966');stage.mkdir(parents=True,exist_ok=False)
    work=OUT/'runtime'/f'{role}-{run or "none"}';work.mkdir()
    env=c.ab().support().environment(work);env['D060_AP50_WORK']=str(work)
    args=[str(c.PYTHON),'-I','-B','-u',str(SELF),'_worker','--role',role]
    if run:args+=['--run',run]
    c.save(stage/'launch.json',dict(at=c.stamp(),role=role,run=run,timeout_seconds=None))
    process=None;job=None;start=time.perf_counter()
    try:
        with (stage/'console.log').open('x',encoding='utf-8') as f:
            process=subprocess.Popen(args,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            job=c.ab().support().WindowsJob(process);process.wait()
    finally:
        if job:job.close()
        if process is not None and process.poll() is None:process.kill();process.wait()
        c.save(stage/'exit.json',dict(at=c.stamp(),code=process.returncode if process else None,
            timeout=False,seconds=time.perf_counter()-start))
    c.check(process.returncode==0 and (stage/'result.json').exists(),'Stage failed; preserve '+str(stage))
    print('COMPLETED',role,run or '',flush=True)
    return c.load(stage/'result.json')


def seal(c):
    c.check(c.load(OUT/'preflight/result.json')['state']=='completed','Preflight required')
    verify_dependencies(c,historical=True)
    paths=[SELF,ROOT/'tests/test_d060_ap50_cycle.py',*(ROOT/p for p in SUPPORT),
        OUT/'plan.json',OUT/'authorization.json',OUT/'protected.json',OUT/'baseline_initialization.json',
        OUT/'baseline_evaluation.json',OUT/'baseline_training.json',OUT/'preflight/result.json',OUT/'architecture/yolov8n.yaml']
    paths+=list((OUT/'preflight').glob('*clarification.json'))
    paths+=list((OUT/'manifests').glob('*.json'))+list(OUT.glob('*.yaml'))
    dependencies=c.ab().support().dependency_hashes()
    for rel in ('engine/validator.py','utils/downloads.py'):
        p=c.PYTHON.parent/'Lib/site-packages/ultralytics'/rel;dependencies[str(p)]=c.sha(p)
    c.save(OUT/'seal.json',dict(at=c.stamp(),hashes={p.relative_to(ROOT).as_posix():c.sha(p) for p in paths},dependencies=dependencies))


def ready(c):
    c.check(not (OUT/'supervision_started.json').exists() and not (OUT/'run_started.json').exists(),'Cannot repeat a started study')
    c.check((OUT/'seal.json').is_file() and (OUT/'preflight/result.json').is_file(),'Sealed preflight required')
    validate_authorization(c.load(OUT/'authorization.json'));c.verify_seal()
    verify_dependencies(c)


def final_verify(c):
    c.verify_pins();c.verify_seal()
    for p,digest in c.load(OUT/'seal.json')['dependencies'].items():c.check(c.sha(p)==digest,'Dependency drift: '+p)
    for key in ('D060','val966'):c.verify_rows(c.data_rows(key))
    for run in RUNS:
        t=c.load(OUT/run/'train/result.json');c.weight_path(run)
        rows=[json.loads(s) for s in (OUT/run/'train/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        c.check(validate_trace(run,rows)==t['best_epoch'],'Best epoch mismatch')
        c.check(t['initial_state_sha256']==c.INITIAL and c.sha(t['last'])==t['last_sha256'],'State/last mismatch')
        c.check(t['best_native_AP50']==rows[t['best_epoch']-1]['fitness'],'Best score mismatch')
        for key,row in [('best',rows[t['best_epoch']-1]),('last',rows[-1])]:
            evidence=t['checkpoint_evidence'][key]
            c.check(evidence['epoch']==row['epoch'] and evidence['fitness']==row['fitness'] and
                all(evidence['metrics'].get(k)==v for k,v in row['metrics'].items()) and
                evidence['finite_weights'] and evidence['optimizer_stripped'],'Checkpoint evidence/trace mismatch')
        v=c.load(OUT/run/'eval_val966/result.json')
        c.check(v['state']=='completed' and v['images']==966 and v['weight_sha256']==t['best_sha256'] and
            v['effective_options']==c.evaluation_options() and [r['cls'] for r in v['classes']]==list(range(4)), 'Evaluation drift')
    stages=[OUT/'preflight']+[c.stage_path(role,r,'val966') for r in RUNS for role in ('train','evaluate')]
    for stage in stages:
        e=c.load(stage/'exit.json');c.check(e['code']==0 and not e['timeout'],'Stage exit failure')
    for work in (OUT/'runtime').iterdir():c.verify_child_guards(work)
    return dict(at=c.stamp(),verified=True,training_runs=2,evaluations=2,originals_unchanged=True,
        protected_files=len(c.load(OUT/'protected.json')),initial_state_sha256=c.INITIAL,no_guard_events=True)


def summarize(c):
    baseline=c.load(OUT/'baseline_evaluation.json')
    results={r:c.load(OUT/r/'eval_val966/result.json') for r in RUNS}
    ap=lambda v:v['native']['metrics/mAP50(B)']
    comparisons=dict(R01_minus_R00=ap(results['R01'])-ap(results['R00']),
        **{r+'_minus_old_SS04':ap(v)-ap(baseline) for r,v in results.items()})
    c.save(OUT/'comparison.json',dict(baseline=baseline,results=results,AP50_deltas=comparisons))
    metrics=[];classes=[]
    for run,v in [('old_SS04',baseline),*results.items()]:
        fixed=v['fixed'];p=fixed['precision_macro'];r=fixed['recall_macro']
        metrics.append(dict(run=run,AP50=ap(v),AP50_95=v['native']['metrics/mAP50-95(B)'],
            fixed_precision=p,fixed_recall=r,F1_of_macro_PR=2*p*r/(p+r) if p+r else 0.))
        for row,fix in zip(v['classes'],fixed['classes']):
            classes.append(dict(run=run,**row,TP=fix['TP'],FP=fix['FP'],FN=fix['FN'],
                fixed_precision=fix['precision'],fixed_recall=fix['recall'],
                fixed_F1=2*fix['TP']/(2*fix['TP']+fix['FP']+fix['FN']) if 2*fix['TP']+fix['FP']+fix['FN'] else 0.))
    for name,rows in [('metrics.csv',metrics),('per_class.csv',classes)]:
        with (OUT/name).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# D060 AP50 cycle study','', '| run | FP32 val AP50 | AP50–95 | best epoch | training seconds |',
          '|---|---:|---:|---:|---:|']
    for row in metrics:
        run=row['run'];t=c.load(OUT/run/'train/result.json') if run in RUNS else c.load(OUT/'baseline_training.json')
        text.append(f"| {run} | {row['AP50']:.12f} | {row['AP50_95']:.12f} | {t['best_epoch']} | {t['seconds']:.2f} |")
    text+=['','AP50 deltas (percentage points):']+[f'- {k}: {v*100:+.12f}' for k,v in comparisons.items()]
    text+=['','Target val AP50 > .90: '+str(any(ap(v)>.9 for v in results.values())),
        '', 'Long-cycle net strategy effect, not isolated extra epochs: stretched LR decay, mosaic closes at epoch71 vs19, more checkpoint selection opportunities.',
        'Single paired seed only; native selection and standalone FP32 evaluation differ. Old SS04 used AP95 selection.',
        'Per-class AP and fixed-point P/R/F1/counts: per_class.csv; F1=2TP/(2TP+FP+FN), zero denominator maps to zero.',
        'No hidden-test use, new submission, automatic follow-up, or competition-compliance claim.']
    with (OUT/'RESULT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')


def run_stages(c):
    c.check(not (OUT/'run_started.json').exists(),'Cannot repeat study')
    c.verify_seal();c.verify_pins()
    verify_dependencies(c)
    for key in ('D060','val966'):c.verify_rows(c.data_rows(key))
    c.save(OUT/'run_started.json',dict(at=c.stamp(),runs=RUNS,training_runs=2,evaluations=2))
    try:
        for run in RUNS:launch(c,'train',run)
        for run in RUNS:launch(c,'evaluate',run)
        c.save(OUT/'independent_verification.json',final_verify(c));summarize(c)
        names=('independent_verification.json','comparison.json','metrics.csv','per_class.csv','RESULT.md')
        c.save(OUT/'success.json',dict(at=c.stamp(),training_runs=2,evaluations=2,files={n:c.sha(OUT/n) for n in names}))
        print('STUDY COMPLETE: 2 trainings + 2 FP32 evaluations',flush=True)
    except BaseException:
        c.save(OUT/'run_failure.json',dict(at=c.stamp(),traceback=traceback.format_exc()));raise


def run(c):
    ready(c);c.save(OUT/'supervision_started.json',dict(at=c.stamp(),timeout_seconds=None))
    process=None;job=None;start=time.perf_counter()
    try:
        with (OUT/'stages.log').open('x',encoding='utf-8') as f:
            process=subprocess.Popen([str(c.PYTHON),'-I','-B','-u',str(SELF),'_run'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            job=c.ab().support().WindowsJob(process);process.wait()
    finally:
        if job:job.close()
        if process is not None and process.poll() is None:process.kill();process.wait()
        c.save(OUT/'supervision_exit.json',dict(at=c.stamp(),code=process.returncode if process else None,
            seconds=time.perf_counter()-start,timeout=False))
    c.check(process.returncode==0 and (OUT/'success.json').is_file(),'Study stopped; preserve evidence, no retry')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','preflight','seal','run','verify','_run','_worker'))
    p.add_argument('--role',choices=('preflight','train','evaluate'));p.add_argument('--run',choices=tuple(RUNS))
    a=p.parse_args();c=configured()
    if a.action=='prepare':prepare(c)
    elif a.action=='preflight':launch(c,'preflight')
    elif a.action=='seal':seal(c)
    elif a.action=='run':run(c)
    elif a.action=='verify':print(json.dumps(final_verify(c),indent=2))
    elif a.action=='_run':run_stages(c)
    else:worker(c,a.role,a.run)


BOOT_EVENTS=[]
if os.environ.get('D060_AP50_WORK'):
    work=Path(os.environ['D060_AP50_WORK']).resolve();c=configured()
    c.check(work.is_relative_to(OUT/'runtime') and work==Path.cwd().resolve(),'Invalid worker bootstrap')
    BOOT_EVENTS=c.guard(work)
    c.save(work/f'guard_ready_{os.getpid()}.json',dict(pid=os.getpid(),module=__name__,at=c.stamp()))
if __name__=='__main__':main()
