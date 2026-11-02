"""One authorized SS04 expansion run: three trainings, twelve evaluations, no retries."""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
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
OUT=ROOT/'reports/train-cleaning-expansion-ss04-20261101'
BASE=ROOT/'reports/cleaning-expansion-080-070-060-20261029'
DATA=BASE/'datasets-20261101'
OLD=ROOT/'reports/train-cleaning-ab-ss04-20261029'
RECOVERY=ROOT/'reports/eval-cleaning-ab-recovery-20261029'
AUTH=BASE/'TRAINING-AUTHORIZATION-20261101.json'
PYTHON=Path(r'E:\anaconda\envs\dl-3lc\python.exe')
BANDS=('D080','D070','D060')
REFERENCES=('D0','D085','self','val966')
BOXES={'D0':72273,'D085':72665,'D080':73002,'D070':74080,'D060':75634,'val966':11487}
INITIAL='b29d1a75cc221bee6f6f2b3722dbc7c65a266fad8cdea8295e938c975d78f8c0'
NAMES={0:'truck',1:'car',2:'van',3:'bus'}
CHECKPOINTS={}


def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def stamp(): return datetime.now(timezone.utc).isoformat()
def check(value,message):
    if not value: raise RuntimeError(message)
def module(name,path):
    m=types.ModuleType(name);m.__file__=str(path)
    exec(compile(Path(path).read_bytes(),str(path),'exec'),m.__dict__)
    return m
def ab(): return module('expansion_ab',ROOT/'tools/train_cleaning_ab.py')
def recovery(): return module('expansion_recovery',ROOT/'tools/eval_cleaning_ab_recovery.py')
def training_options(): return ab().training_options()
def evaluation_options(): return recovery().evaluation_options()
def evaluation_matrix(): return [(band,ref) for band in BANDS for ref in REFERENCES]
def timeout(role,deadline,current):
    remaining=deadline-current
    if remaining<=0: raise TimeoutError('Expansion total deadline exhausted')
    return min({'train':5400,'evaluate':3600,'preflight':300}[role],remaining)


def allowed_file(path,writing):
    p=Path(path).resolve()
    if p.is_relative_to(OUT.resolve()): return True
    if writing: return False
    exact=[SELF,ROOT/'tools/train_cleaning_ab.py',ROOT/'tools/eval_cleaning_ab_recovery.py',ROOT/'tools/oof_diagnostic.py']
    return p.is_relative_to(PYTHON.parent.resolve()) or p in exact


def weight_path(band):
    check(band in BANDS,'Unknown model')
    r=load(OUT/band/'train/result.json')
    check(r['state']=='completed' and r['epochs']==28,'Incomplete training')
    path=Path(r['best']).resolve()
    check(path.is_relative_to((OUT/band).resolve()) and path.suffix=='.pt','Unsafe actual best path')
    check(sha(path)==r['best_sha256'],'Best hash drift')
    return path


def guard(work):
    import importlib.abc
    events=[]
    def deny(event,**details):
        value=dict(at=stamp(),event=event,**details);events.append(value)
        with (work/f'guard_events_{os.getpid()}.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')
        raise PermissionError('Expansion offline guard: '+event)
    def hook(event,args):
        if event in ['socket.connect','socket.connect_ex','socket.bind','socket.getaddrinfo','socket.sendto',
                     'socket.sendmsg','subprocess.Popen','os.system','os.posix_spawn']: deny(event)
        mutations={'os.remove':1,'os.rmdir':1,'os.mkdir':1,'os.rename':2,'os.link':2,'os.symlink':2,
                   'os.chmod':1,'os.utime':1,'os.truncate':1}
        if event in mutations:
            for arg in args[:mutations[event]]:
                if not isinstance(arg,(str,bytes,os.PathLike)) or not allowed_file(os.fsdecode(arg),True):
                    deny('mutation',operation=event)
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            raw=os.fsdecode(args[0])
            if raw in ['/etc/os-release','/etc/dgx-release','/proc/cpuinfo','/proc/device-tree/model']:raise FileNotFoundError(raw)
            if raw.lower() in ['nul',r'\\.\nul']:return
            mode,flags=args[1:3]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
            if not allowed_file(raw,writing):deny('file',path=str(Path(raw).resolve()),writing=bool(writing))
    class No3LC(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ['tlc','tlc_ultralytics','tlcconfig']:deny('3lc_import',module=fullname)
    sys.meta_path.insert(0,No3LC());sys.addaudithook(hook)
    return events


def validate_authorization(a):
    required=dict(approved_by_user=True,training_runs=3,evaluations=12,models=list(BANDS),references=list(REFERENCES),
        training_timeout_seconds=5400,evaluation_timeout_seconds=3600,total_timeout_seconds=21600,
        no_retry=True,no_label_or_parameter_changes=True,no_test_evaluation=True,no_network_or_3lc=True)
    check(all(a.get(k)==v for k,v in required.items()) and bool(a.get('user_message')),'Authorization scope/budget mismatch')


def data_rows(key): return load(OUT/'manifests'/(key+'.json'))
def verify_rows(rows): ab().verify_rows(rows)
def verify_pins():
    for name,digest in load(OUT/'protected.json').items():check(sha(ROOT/name)==digest,'Protected drift: '+name)
def verify_seal():
    for name,digest in load(OUT/'seal.json')['hashes'].items():check(sha(ROOT/name)==digest,'Executor seal drift: '+name)


def prepare():
    check(not OUT.exists(),'Existing run preserved; do not prepare/retry')
    authorization=load(AUTH);validate_authorization(authorization)
    check(sha(BASE/'experiment_spec.json')==authorization['experiment_spec_sha256'],'Experiment specification drift')
    check(sha(DATA/'application_seal.json')==authorization['application_seal_sha256'],'Application seal drift')
    app=module('expansion_data_verifier',ROOT/'tools/cleaning_expansion_apply.py');app.verify(ROOT,DATA)
    frozen=load(BASE/'experiment_spec.json')
    check(frozen['training']==training_options() and frozen['evaluation']==evaluation_options(),'SS04 frozen option drift')
    groups={}
    old_rows=load(OLD/'train_manifest.json')
    for key,arm in [('D0','A'),('D085','B')]:groups[key]=[r for r in old_rows if r['arm']==arm]
    for band in BANDS:
        rows=load(DATA/band/'manifest.json')['rows']
        groups[band]=[dict(r,image=str(DATA/r['image']),label=str(DATA/r['label'])) for r in rows if r['split']=='train']
    groups['val966']=load(OLD/'val_manifest.json')
    for key,rows in groups.items():
        check(len(rows)==(966 if key=='val966' else 7852),'Unexpected split size')
        verify_rows(rows)
    identity=lambda rows:[(r.get('id'),Path(r['image']).name,r['image_sha256']) for r in sorted(rows,key=lambda r:Path(r['image']).name)]
    for key in ('D085',)+BANDS:check(identity(groups[key])==identity(groups['D0']),'Image membership/order differs')
    check(not {r['image_sha256'] for r in groups['D0']} & {r['image_sha256'] for r in groups['val966']},'Train/val hash overlap')
    protected=dict(load(BASE/'protected.json')['files'])
    protected.update(load(BASE/'protected.json')['sources'])
    for key,rows in groups.items():
        for row in rows:
            for field in ('image','label'):protected[Path(row[field]).relative_to(ROOT).as_posix()]=row[field+'_sha256']
    historical=[AUTH,BASE/'experiment_spec.json',DATA/'application_seal.json',DATA/'SUCCESS.json',
                OLD/'A/train/initialization.json',OLD/'B/train/initialization.json',RECOVERY/'success.json',RECOVERY/'comparison.json']
    for arm in ('A','B'):
        historical += [OLD/arm/'train/result.json']+[RECOVERY/arm/('eval_'+ref)/'result.json' for ref in ('A','B','val966')]
    for p in historical:protected[p.relative_to(ROOT).as_posix()]=sha(p)
    copies_bytes=sum(Path(r[k]).stat().st_size for rows in groups.values() for r in rows for k in ('image','label'))
    reserve=4_000_000_000
    check(shutil.disk_usage(ROOT).free>copies_bytes+reserve,'Insufficient independent copies/runtime/checkpoint reserve')
    OUT.mkdir();(OUT/'runtime').mkdir();(OUT/'architecture').mkdir()
    architecture=OLD/'architecture/yolov8n.yaml';shutil.copyfile(architecture,OUT/'architecture/yolov8n.yaml')
    protected[architecture.relative_to(ROOT).as_posix()]=sha(architecture)
    for key,rows in groups.items():
        copied=[];split='val' if key=='val966' else 'train'
        for row in sorted(rows,key=lambda r:Path(r['image']).name):
            new=dict(row)
            for field,folder in [('image','images'),('label','labels')]:
                src=Path(row[field]);dest=OUT/'data'/key/folder/split/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
                with src.open('rb') as f,dest.open('xb') as g:shutil.copyfileobj(f,g)
                check(sha(dest)==row[field+'_sha256'] and dest.stat().st_ino!=src.stat().st_ino,'Independent copy mismatch')
                new[field]=str(dest)
            copied.append(new)
        save(OUT/'manifests'/(key+'.json'),copied)
    import yaml
    for band in BANDS:
        configs={band:dict(train=OUT/'data'/band/'images/train',val=OUT/'data/val966/images/val')}
        for ref in REFERENCES:
            key=band if ref=='self' else ref;images=OUT/'data'/key/'images'/('val' if ref=='val966' else 'train')
            configs[band+'-eval-'+ref]=dict(train=images,val=images)
        for name,paths in configs.items():
            with (OUT/(name+'.yaml')).open('x',encoding='utf-8') as f:
                yaml.safe_dump(dict(path=str(OUT),train=str(paths['train']),val=str(paths['val']),nc=4,names=NAMES),f,sort_keys=False)
    save(OUT/'protected.json',protected)
    save(OUT/'baseline_initialization.json',load(OLD/'A/train/initialization.json'))
    save(OUT/'authorization.json',authorization)
    save(OUT/'plan.json',dict(at=stamp(),trainings=3,evaluations=12,models=BANDS,references=REFERENCES,
        total_timeout_seconds=21600,copy_bytes=copies_bytes,runtime_checkpoint_reserve_bytes=reserve,
        disk_free_after_prepare=shutil.disk_usage(ROOT).free,training=training_options(),evaluation=evaluation_options(),
        initial_state_sha256=INITIAL,boxes=BOXES,no_retry=True))
    verify_pins()
    print('PREPARED independent D0/D085/three-self/val data; no model calls',flush=True)


def approved_asset(file,*args,**kwargs):
    p=Path(file).resolve();approved=dict(CHECKPOINTS)
    for band in BANDS:
        if (OUT/band/'train/result.json').is_file():
            approved[str(weight_path(band))]=load(OUT/band/'train/result.json')['best_sha256']
    check(str(p) in approved and p.is_relative_to(OUT.resolve()) and p.is_file() and sha(p)==approved[str(p)],
          'Unapproved/missing weight; online fallback forbidden')
    return str(p)


def local_runtime():
    import ultralytics.data.utils as data_utils
    import ultralytics.engine.trainer as engine
    import ultralytics.utils.downloads as downloads
    data_utils.check_font=lambda *a,**k:None
    engine.check_amp=lambda model:True
    downloads.attempt_download_asset=approved_asset


def native_final_eval(trainer,final,engine):
    paths={str(Path(p).resolve()) for p in (trainer.best,trainer.last)}
    for p in paths:
        check(Path(p).is_relative_to(OUT.resolve()) and Path(p).is_file(),'Missing current-run checkpoint')
        CHECKPOINTS[p]=sha(p)
    original=engine.strip_optimizer
    def verified_strip(file,*args,**kwargs):
        p=str(Path(file).resolve())
        check(p in paths and sha(p)==CHECKPOINTS[p],'Unexpected checkpoint before native optimizer strip')
        result=original(file,*args,**kwargs)
        CHECKPOINTS[p]=sha(p)
        return result
    engine.strip_optimizer=verified_strip
    try:return final()
    finally:engine.strip_optimizer=original


def trainer_factory(base):
    import torch
    class ExpansionTrainer(base):
        def get_model(self,cfg=None,weights=None,verbose=True):
            check(weights is None,'Pretrained/resume prohibited')
            return super().get_model(cfg=cfg,weights=None,verbose=verbose)
        def _handle_nan_recovery(self,epoch):
            values=self.loss.values() if isinstance(self.loss,dict) else [self.loss]
            check(all(torch.isfinite(v).all() for v in values),'Nonfinite loss; stop without retry')
            check(self.fitness is None or math.isfinite(float(self.fitness)),'Nonfinite fitness')
            return False
        def final_eval(self):
            import ultralytics.engine.trainer as engine
            self.expansion_final_eval=True
            return native_final_eval(self,super().final_eval,engine)
        def save_model(self):
            for net in (self.model,self.ema.ema):
                for value in net.state_dict().values():
                    if value.is_floating_point():
                        check(torch.isfinite(value).all() and torch.isfinite(value.half()).all(),'Nonfinite/FP16 overflow; no silent repair')
            return super().save_model()
    return ExpansionTrainer


def initial_contract(model):
    check(model.names==NAMES and sum(p.numel() for p in model.parameters())==3011628,'YOLOv8n architecture/classes drift')
    digest=ab().state_hash(model)
    check(digest==INITIAL,'Initial parameters differ from original paired A/B')
    return digest


def train(band,stage):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import init_seeds
    local_runtime();init_seeds(42,deterministic=True)
    model=YOLO(str(OUT/'architecture/yolov8n.yaml'),task='detect')
    check(not model.ckpt and not model.ckpt_path,'Random initialization required')
    meta={};epochs=[]
    def initialized(t):
        support=ab()
        support.assert_training_contract(vars(t.args),type(t.optimizer).__name__,
            [p['lr'] for p in t.optimizer.param_groups],t.optimizer.param_groups[0]['betas'],t.amp)
        support.assert_loader(t.train_loader.dataset,data_rows(band),BOXES[band])
        support.assert_loader(t.test_loader.dataset,data_rows('val966'),11487,ordered=False)
        baseline=load(OUT/'baseline_initialization.json')['effective_args']
        ignore={'model','data','project','name','save_dir'}
        for key,value in baseline.items():
            if key not in ignore:check(vars(t.args).get(key)==value,'Full A/B effective option drift: '+key)
        meta.update(initial_state_sha256=initial_contract(t.model),effective_args=vars(t.args),
            optimizer=type(t.optimizer).__name__,initial_lr=[p['lr'] for p in t.optimizer.param_groups],
            betas=t.optimizer.param_groups[0]['betas'],amp=bool(t.amp),parameters=3011628,
            train_images=7852,train_boxes=BOXES[band],val_images=966,val_boxes=11487)
        save(stage/'initialization.json',meta)
    def epoch_start(t):t._oom_retries=3
    def epoch_end(t):
        if getattr(t,'expansion_final_eval',False):return
        row=dict(epoch=t.epoch+1,metrics={k:float(v) for k,v in t.metrics.items()},fitness=float(t.fitness),
                 best_updated=t.best_fitness==t.fitness,at=stamp())
        epochs.append(row)
        with (stage/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print('EPOCH',band,row['epoch'],'AP95',row['fitness'],flush=True)
    model.add_callback('on_train_start',initialized);model.add_callback('on_train_epoch_start',epoch_start)
    model.add_callback('on_fit_epoch_end',epoch_end)
    model.train(trainer=trainer_factory(DetectionTrainer),data=str(OUT/(band+'.yaml')),
                project=str(OUT/band),name='model',exist_ok=False,**training_options())
    check(len(epochs)==28,'Incomplete 28 epochs')
    t=model.trainer
    return dict(band=band,epochs=28,best_epoch=max(r['epoch'] for r in epochs if r['best_updated']),
        best=str(t.best),last=str(t.last),best_sha256=sha(t.best),last_sha256=sha(t.last),
        initial_state_sha256=meta['initial_state_sha256'],native_final_metrics={k:float(v) for k,v in t.metrics.items()})


def evaluate(band,ref,stage):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    local_runtime();support=ab();key=band if ref=='self' else ref
    rows=data_rows(key);capture=dict(expected_rows=rows,expected_boxes=BOXES[key])
    factory=support.scored_validator_factory(DetectionValidator,capture)
    weight=weight_path(band);model=YOLO(str(weight),task='detect')
    metrics=model.val(validator=factory,data=str(OUT/(band+'-eval-'+ref+'.yaml')),
        project=str(stage),name='native',**evaluation_options())
    v=capture['validator'];check(v.seen==len(rows) and v.args.quantize is None,'Coverage/FP32 drift')
    effective={k:getattr(v.args,k) for k in evaluation_options()}
    check(effective==evaluation_options(),'Effective evaluation option drift')
    classes=[dict(cls=int(c),name=NAMES[int(c)],precision_best_f1=float(metrics.box.p[i]),
        recall_best_f1=float(metrics.box.r[i]),AP50=float(metrics.box.all_ap[i,0]),
        AP50_95=float(metrics.box.all_ap[i].mean())) for i,c in enumerate(metrics.box.ap_class_index)]
    return dict(band=band,reference=ref,images=v.seen,weight=str(weight),weight_sha256=sha(weight),
        native={k:float(v) for k,v in metrics.results_dict.items()},classes=classes,fixed=v.common_fixed,
        precision='FP32 quantize=None',effective_options=effective,
        evaluator='Same Ultralytics native AP and filter-first/rematch fixed metrics as A/B recovery')


def preflight(stage):
    import torch,ultralytics
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    local_runtime()
    result=ab().preflight(stage)
    init_seeds(42,deterministic=True)
    y=YOLO(str(OUT/'architecture/yolov8n.yaml'),task='detect')
    t=trainer_factory(DetectionTrainer)(overrides=dict(model=str(OUT/'architecture/yolov8n.yaml'),
        data=str(OUT/'D080.yaml'),project=str(stage),name='initialization-only',exist_ok=False,**training_options()))
    model=t.get_model(cfg=y.model.yaml,weights=None,verbose=False);model.names=NAMES
    initial=initial_contract(model)
    return dict(result,initial_state_sha256=initial,model_parameters=3011628,
        no_dataset_inference=True,no_optimizer_steps=True,
        runtime=dict(python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,
                     cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))


def stage_path(role,band=None,ref=None):
    return OUT/'preflight' if role=='preflight' else OUT/band/('train' if role=='train' else 'eval_'+ref)


def verify_child_guards(work):
    check(not list(Path(work).glob('guard_events*.jsonl')),'Guard rejection in worker/child')
    children=[load(p) for p in Path(work).glob('guard_ready_*.json') if load(p)['module']=='__mp_main__']
    check(len(children)>=2,'Expected guarded DataLoader spawn children')
    return len(children)


def worker(role,band,ref):
    stage=stage_path(role,band,ref);start=time.perf_counter()
    try:
        import torch,ultralytics
        from ultralytics.utils import ONLINE,SETTINGS
        check(sys.version_info[:3]==(3,12,15) and torch.__version__=='2.14.1+cu130' and
              ultralytics.__version__=='8.4.165' and torch.version.cuda=='13.0' and not ONLINE and not SETTINGS['sync'],
              'Frozen offline environment mismatch')
        result=preflight(stage) if role=='preflight' else (train(band,stage) if role=='train' else evaluate(band,ref,stage))
        check(not BOOT_EVENTS and not list(Path.cwd().glob('guard_events*.jsonl')),'Guard rejection caught in worker/children')
        result['guarded_spawn_children']=verify_child_guards(Path.cwd())
        result.update(state='completed',seconds=time.perf_counter()-start,at=stamp(),
            no_3lc_import=not any(k.split('.')[0] in ['tlc','tlc_ultralytics','tlcconfig'] for k in sys.modules))
        save(stage/'result.json',result)
    except BaseException:
        save(stage/'failure.json',dict(at=stamp(),seconds=time.perf_counter()-start,traceback=traceback.format_exc()));raise


def launch(role,band=None,ref=None,deadline=None):
    limit=timeout(role,deadline,time.perf_counter());stage=stage_path(role,band,ref);stage.mkdir(parents=True,exist_ok=False)
    work=OUT/'runtime'/f'{role}-{band or "none"}-{ref or "none"}';work.mkdir()
    env=ab().support().environment(work);env['CLEANING_EXPANSION_WORK']=str(work)
    args=[str(PYTHON),'-I','-B','-u',str(SELF),'_worker','--role',role]
    if band:args+=['--band',band]
    if ref:args+=['--reference',ref]
    save(stage/'launch.json',dict(at=stamp(),role=role,band=band,reference=ref,timeout_seconds=limit))
    job=None;timed_out=False;start=time.perf_counter()
    with (stage/'console.log').open('x',encoding='utf-8') as f:
        process=subprocess.Popen(args,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            job=ab().support().WindowsJob(process);process.wait(timeout=limit)
        except subprocess.TimeoutExpired:timed_out=True
        finally:
            if job:job.close()
            if process.poll() is None:process.kill();process.wait()
    save(stage/'exit.json',dict(at=stamp(),code=process.returncode,timeout=timed_out,seconds=time.perf_counter()-start))
    check(not timed_out and process.returncode==0 and (stage/'result.json').exists(),'Stage failed; preserve '+str(stage))
    print('COMPLETED',role,band or '',ref or '',flush=True)
    return load(stage/'result.json')


def seal():
    check(load(OUT/'preflight/result.json')['state']=='completed','Preflight required')
    paths=[SELF,ROOT/'tools/summarize_cleaning_expansion.py',ROOT/'tests/test_train_cleaning_expansion.py',
        ROOT/'tests/test_summarize_cleaning_expansion.py',ROOT/'tools/train_cleaning_ab.py',
        ROOT/'tools/eval_cleaning_ab_recovery.py',ROOT/'tools/oof_diagnostic.py',
        OUT/'plan.json',OUT/'authorization.json',OUT/'protected.json',OUT/'baseline_initialization.json',
        OUT/'preflight/result.json',OUT/'architecture/yolov8n.yaml']
    paths+=list((OUT/'manifests').glob('*.json'))+list(OUT.glob('*.yaml'))
    pins={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    # Environment path explicitly authorized for this run; only dependency files used by the executor.
    dependencies=ab().support().dependency_hashes()
    save(OUT/'seal.json',dict(at=stamp(),hashes=pins,dependencies=dependencies))


def ready_to_run():
    check(not (OUT/'supervision_started.json').exists() and not (OUT/'run_started.json').exists(),'Cannot repeat a started run')
    check((OUT/'seal.json').is_file() and (OUT/'preflight/result.json').is_file(),'Sealed preflight required')
    validate_authorization(load(OUT/'authorization.json'));verify_seal()


def final_verify():
    verify_pins();verify_seal()
    for name,digest in load(OUT/'seal.json')['dependencies'].items():check(sha(name)==digest,'Dependency drift: '+name)
    for key in ('D0','D085','val966')+BANDS:verify_rows(data_rows(key))
    for band in BANDS:
        t=load(OUT/band/'train/result.json');weight_path(band)
        check(t['initial_state_sha256']==INITIAL and t['epochs']==28,'Initialization/epochs drift')
        check(sha(t['last'])==t['last_sha256'],'Last weight drift')
        epochs=[json.loads(line) for line in (OUT/band/'train/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        check([r['epoch'] for r in epochs]==list(range(1,29)),'Epoch coverage mismatch')
        for ref in REFERENCES:
            stage=stage_path('evaluate',band,ref);v=load(stage/'result.json');e=load(stage/'exit.json')
            check(v['state']=='completed' and v['weight_sha256']==t['best_sha256'] and
                v['images']==(966 if ref=='val966' else 7852) and v['effective_options']==evaluation_options() and
                e['code']==0 and not e['timeout'],'Incomplete/unbound evaluation')
    check(not list((OUT/'runtime').rglob('guard_events*.jsonl')),'Worker/child guard rejection exists')
    stages=[OUT/'preflight']+[stage_path('train',b) for b in BANDS]+[stage_path('evaluate',b,r) for b,r in evaluation_matrix()]
    for stage in stages:
        e=load(stage/'exit.json');check(e['code']==0 and not e['timeout'],'Stage failed')
    for work in (OUT/'runtime').iterdir():
        if work.is_dir():verify_child_guards(work)
    return dict(at=stamp(),verified=True,trainings=3,evaluations=12,protected_files=len(load(OUT/'protected.json')),
                originals_unchanged=True,initial_state_sha256=INITIAL,no_guard_events=True)


def run_stages():
    check(not (OUT/'run_started.json').exists(),'Cannot repeat experiment')
    verify_seal();verify_pins()
    for key in ('D0','D085','val966')+BANDS:verify_rows(data_rows(key))
    deadline=time.perf_counter()+max(0,21600-(time.time()-load(OUT/'supervision_started.json')['started_unix']))
    save(OUT/'run_started.json',dict(at=stamp(),trainings=3,evaluations=12))
    try:
        for band in BANDS:launch('train',band,deadline=deadline)
        for band,ref in evaluation_matrix():launch('evaluate',band,ref,deadline=deadline)
        verification=final_verify();save(OUT/'independent_verification.json',verification)
        summary=module('expansion_summary',ROOT/'tools/summarize_cleaning_expansion.py');summary.OUT=OUT;summary.main()
        check(time.perf_counter()<deadline,'Total deadline exceeded during verification/report')
        required=['comparison.json','metrics.csv','per_class.csv','class_deltas.csv','change_scale.csv','RESULT.md','independent_verification.json']
        save(OUT/'success.json',dict(at=stamp(),trainings=3,evaluations=12,files={n:sha(OUT/n) for n in required},
             initial_state_sha256=INITIAL,originals_unchanged=True,no_retry=True))
        print('EXPANSION COMPLETE: 3 trainings + 12 evaluations + verified reports',flush=True)
    except BaseException:
        save(OUT/'run_failure.json',dict(at=stamp(),traceback=traceback.format_exc()));raise


def run():
    ready_to_run()
    save(OUT/'supervision_started.json',dict(at=stamp(),started_unix=time.time(),timeout_seconds=21600))
    process=None;job=None;timed_out=False;error=None;start=time.perf_counter()
    try:
        with (OUT/'stages.log').open('x',encoding='utf-8') as f:
            process=subprocess.Popen([str(PYTHON),'-I','-B','-u',str(SELF),'_run'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            job=ab().support().WindowsJob(process);process.wait(timeout=21600)
    except subprocess.TimeoutExpired:timed_out=True
    except BaseException:error=traceback.format_exc()
    finally:
        if job:job.close()
        if process is not None and process.poll() is None:process.kill();process.wait()
    save(OUT/'supervision_exit.json',dict(at=stamp(),code=process.returncode if process else None,
        timeout=timed_out,error=error,seconds=time.perf_counter()-start))
    check(not timed_out and not error and process.returncode==0 and (OUT/'success.json').is_file(),
          'Execution stopped; preserve all evidence, no retry')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','preflight','seal','run','verify','_run','_worker'])
    p.add_argument('--role',choices=['preflight','train','evaluate']);p.add_argument('--band',choices=BANDS)
    p.add_argument('--reference',choices=REFERENCES);args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='preflight':launch('preflight',deadline=time.perf_counter()+300)
    elif args.action=='seal':seal()
    elif args.action=='run':run()
    elif args.action=='_run':run_stages()
    elif args.action=='verify':print(json.dumps(final_verify(),indent=2))
    else:worker(args.role,args.band,args.reference)


BOOT_EVENTS=[]
if os.environ.get('CLEANING_EXPANSION_WORK'):
    work=Path(os.environ['CLEANING_EXPANSION_WORK']).resolve()
    check(work.is_relative_to(OUT/'runtime') and work==Path.cwd().resolve(),'Invalid guarded worker bootstrap')
    BOOT_EVENTS=guard(work)
    save(work/f'guard_ready_{os.getpid()}.json',dict(pid=os.getpid(),module=__name__,at=stamp()))
if __name__=='__main__':main()
