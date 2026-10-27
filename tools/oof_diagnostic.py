"""One approved grouped five-fold OOF diagnostic. No model imports in the supervisor.

Offline only, no original labels changed. Runtime adaptations are process-local.
"""
from __future__ import annotations
import argparse
import collections
import csv
import ctypes
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/oof-diagnostic-5fold-24e-retry1-20261026'
CONFIG=ROOT/'tools/oof_diagnostic.yaml'
TEMPLATE=ROOT/'tools/oof_review.template.html'
TEST=ROOT/'tests/test_oof_diagnostic.py'
PLAN=ROOT/'.pi/plan/5折分组oof离线诊断-ss03最多24轮-独立内层早停patience8-20261026-1355.md'
PYTHON=Path(r'E:\anaconda\envs\dl-3lc\python.exe')
ENV=PYTHON.parent
PKG=ENV/'Lib/site-packages/ultralytics'
ARCH=PKG/'cfg/models/v8/yolov8.yaml'
PREVIOUS=ROOT/'reports/oof-diagnostic-5fold-24e-20261025'
AMENDMENT=ROOT/'reports/oof-recovery-authorization-20261026.md'
SCRATCH=Path(r'E:\pi-desktop\data\data\scratch\112b9029-dd22-4fc5-9132-2dab3956e6cc')
KIT=ROOT/'multi-vehicle_kaggle_competition_starter_kit_files'
INVENTORY=ROOT/'reports/bus-static-review/scene_manifest.json'
NAMES={0:'truck',1:'car',2:'van',3:'bus'}
HEAD='c385eefc1b97d5334c52633232ff3aa5ec4fa908'
GUARD_EVENTS=[]
BOOT_CONFIG=None
DEADLINE_TIMER=None


def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode('utf-8')).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
def append(p,x):
    with Path(p).open('a',encoding='utf-8',newline='\n') as f:f.write(canonical(x)+'\n');f.flush()
def update(p,x):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
    os.replace(tmp,p)
def rows(p):
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():raise ValueError('Blank JSONL record')
            yield json.loads(line)
def within(p,roots):return any(p==r or r in p.parents for r in roots)

def module(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def cfg():
    import yaml
    c=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if c['folds']!=5 or c['split_seed']!=42 or c['monitor_fraction_of_remaining']!=.1:raise ValueError('Approved split config changed')
    t=c['training'];p=c['prediction'];b=c['budget']
    required=dict(epochs=24,patience=8,batch=16,imgsz=640,device=0,workers=2,seed=0,deterministic=True,pretrained=False,resume=False,optimizer='auto',lr0=.01,mosaic=1.,mixup=0.,copy_paste=0.,cutmix=0.,close_mosaic=10,amp=True,cache=False,rect=False,compile=False,fraction=1.,val=True,save=True,save_period=-1,plots=False)
    if t!=required:raise ValueError('Approved SS03 training config changed')
    if p!=dict(conf=.001,iou=.7,max_det=300,agnostic_nms=False,batch=1,imgsz=640,device=0,half=False,augment=False,rect=False,compile=False):raise ValueError('Approved prediction config changed')
    if b!=dict(training_seconds_per_fold=2700,prediction_seconds_per_fold=900,batch_seconds=21600,training_attempts=5):raise ValueError('Budget drift')
    if c['safety']!=dict(random_initialization_only=True,independent_monitor=True,preserve_originals=True,allow_network=False,allow_3lc=False,allow_retry=True,allow_relabeling=False,allow_test=False,external_amp_probe=False,nan_recovery=False,competition_approved=False):raise ValueError('Safety drift')
    return c

def effective_value(key,value):
    # Installed trainer canonicalizes device=0 to cuda:0; this is not a device change.
    return 'cuda:0' if key=='device' and value in (0,'0','cuda:0') else value

def assert_effective(actual,expected):
    for key,value in expected.items():
        if effective_value(key,actual[key])!=effective_value(key,value):raise RuntimeError('Effective parameter drift: '+key)

def loss_values(value):return list(value.values()) if isinstance(value,dict) else ([] if value is None else [value])
def arm_deadline(deadline):
    """Bound supervisor CPU/copy/hash stages too; closing parent handles kills worker job trees."""
    global DEADLINE_TIMER
    def expired():
        try:
            if OUT.exists():write(OUT/'budget_exceeded.json',dict(state='paused_budget_exhausted',deadline_unix=deadline,observed_at=now(),exit_code=124,original_integrity='not claimed by timeout handler'))
        finally:os._exit(124)
    DEADLINE_TIMER=threading.Timer(max(.001,deadline-time.time()),expired)
    DEADLINE_TIMER.daemon=True;DEADLINE_TIMER.start()

def source_hashes():return {p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),CONFIG,TEMPLATE,TEST,PLAN,AMENDMENT)}
def dependency_hashes():
    rel=['cfg/default.yaml','cfg/models/v8/yolov8.yaml','engine/trainer.py','engine/model.py','models/yolo/detect/train.py','models/yolo/detect/predict.py','models/yolo/detect/val.py','data/dataset.py','data/utils.py','utils/checks.py','utils/torch_utils.py','utils/metrics.py','nn/tasks.py']
    return {str(PKG/r):sha(PKG/r) for r in rel}

def materialize_architecture(directory):
    """An existing scaled local YAML avoids full-path regex fallback; no model imports."""
    import yaml
    p=Path(directory)/'architecture/yolov8n.yaml';p.parent.mkdir(parents=True)
    with ARCH.open('rb') as f,p.open('xb') as g:shutil.copyfileobj(f,g)
    if sha(p)!=sha(ARCH):raise ValueError('Local architecture copy mismatch')
    parsed=yaml.safe_load(p.read_text(encoding='utf-8'))
    if 'n' not in parsed.get('scales',{}) or not parsed.get('backbone') or not parsed.get('head'):raise ValueError('Invalid local YOLOv8n architecture')
    return str(p)

def protected_files():
    fixed=['AGENTS.md','RULES.md','tools/bus_static_audit.py','multi-vehicle_kaggle_competition_starter_kit_files/train.py','multi-vehicle_kaggle_competition_starter_kit_files/val.py','multi-vehicle_kaggle_competition_starter_kit_files/predict.py','multi-vehicle_kaggle_competition_starter_kit_files/register_tables.py','multi-vehicle_kaggle_competition_starter_kit_files/config.yaml','multi-vehicle_kaggle_competition_starter_kit_files/dataset.yaml','reports/data_manifest.json','reports/data_summary.json']
    pins={r:sha(ROOT/r) for r in fixed if (ROOT/r).is_file()}
    report_dirs=['baseline-starter-20261024','bus-static-review','bus-crop-pilot','bus-crop-expansion-preflight-20261025','bus-crop-expansion-20261025','bus-crop-expansion-retry1-20261025','bus-crop-confidence-review-20261025','bus-val-contact-MVI_40181-20261025']
    for name in report_dirs:
        for p in (ROOT/'reports'/name).rglob('*'):
            if p.is_file():pins[p.relative_to(ROOT).as_posix()]=sha(p)
    return pins

def verify_hashes(hashes,base=ROOT):
    for name,h in hashes.items():
        if sha(base/name)!=h:raise ValueError('Frozen file changed: '+name)

def originals_integrity():return module('oof_original_audit','tools/bus_static_audit.py').integrity()

def inventory():
    inv=read(INVENTORY);expected=read(ROOT/'reports/bus-static-review/verification.json')['artifact_sha256']['scene_manifest.json']
    if sha(INVENTORY)!=expected:raise ValueError('Inventory hash drift')
    samples=sorted([s for s in inv['images'].values() if s['split']=='train'],key=lambda s:(s['scene'],s['frame'],s['id']))
    if len(samples)!=7852 or len({s['id'] for s in samples})!=7852 or len({s['scene'] for s in samples})!=100:raise ValueError('Train scope drift')
    counts=[sum(int(b[0])==i for s in samples for b in s['labels']) for i in range(4)]
    if counts!=[1226,59385,5558,6104]:raise ValueError('Label counts drift')
    for s in samples:
        if not s['scene'].startswith('MVI_'):raise ValueError('Unknown group')
        for kind in ('image','label'):
            p=(ROOT/s[kind+'_path']).resolve();parent=KIT/'data/train'/('images' if kind=='image' else 'labels')
            if p.parent!=parent.resolve():raise ValueError('Nontrain source')
    return samples

def features(samples):
    result=[float(len(samples))]+[0.]*8
    for s in samples:
        present=set()
        for label in s['labels']:
            cl=int(label[0])
            if cl not in NAMES:raise ValueError('Four classes only')
            result[1+cl]+=1;present.add(cl)
        for cl in present:result[5+cl]+=1
    return result

def components(samples):
    scenes=sorted({s['scene'] for s in samples});parent={g:g for g in scenes}
    def find(g):
        while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
        return g
    seen={}
    for s in samples:
        g=s['scene'];h=s['image_sha256']
        if h in seen:
            a,b=find(g),find(seen[h]);parent[max(a,b)]=min(a,b)
        else:seen[h]=g
    groups=collections.defaultdict(list)
    for s in samples:groups[find(s['scene'])].append(s)
    return [dict(key=k,samples=groups[k],vector=features(groups[k]),scenes=sorted({s['scene'] for s in groups[k]})) for k in sorted(groups)]

def add(a,b):return [x+y for x,y in zip(a,b)]
def cost(load,target):return sum((4 if i==0 else 1)*((x-y)/max(y,1))**2 for i,(x,y) in enumerate(zip(load,target)))

def split_folds(samples,n=5,seed=42):
    if n<2:raise ValueError('At least two folds')
    comp=components(samples);tot=features(samples);target=[v/n for v in tot]
    if len(comp)<n:raise ValueError('Too few indivisible groups')
    rng=random.Random(seed);ties={c['key']:rng.random() for c in comp}
    ordered=sorted(comp,key=lambda c:(-max(v/max(t,1) for v,t in zip(c['vector'],target)),ties[c['key']],c['key']))
    buckets=[[] for _ in range(n)];loads=[[0.]*9 for _ in range(n)]
    fold_ties=list(range(n));rng.shuffle(fold_ties)
    for c in ordered:
        j=min(fold_ties,key=lambda j:(cost(add(loads[j],c['vector']),target)-cost(loads[j],target),loads[j][0]))
        buckets[j].append(c);loads[j]=add(loads[j],c['vector'])
    folds=[]
    for j,bucket in enumerate(buckets):
        remain=[c for i,b in enumerate(buckets) if i!=j for c in b];target_mon=[v*.1 for v in features([s for c in remain for s in c['samples']])]
        fit=remain[:];mon=[];load=[0.]*9
        while fit:
            fit_total=[sum(c['vector'][i] for c in fit) for i in range(9)]
            viable=[c for c in fit if all(fit_total[i]-c['vector'][i]>0 for i in range(1,5))]
            if not viable:break
            best=min(viable,key=lambda c:(cost(add(load,c['vector']),target_mon),ties[c['key']],c['key']))
            required=not mon or any(load[i]==0 for i in range(1,5)) or load[0]<target_mon[0]*.75
            if not required and cost(add(load,best['vector']),target_mon)>=cost(load,target_mon):break
            mon.append(best);fit.remove(best);load=add(load,best['vector'])
        ids=lambda cs:sorted(s['id'] for c in cs for s in c['samples'])
        folds.append(dict(fold=j+1,fit=ids(fit),monitor=ids(mon),outer=ids(bucket)))
    validate_splits(samples,folds,n)
    return folds
def validate_splits(samples,folds,n=5):
    by={s['id']:s for s in samples};all_ids=set(by);outer=[]
    for kind in ('image_path','label_path'):
        paths=[s[kind] for s in samples if kind in s]
        if len(paths)!=len(set(paths)):raise ValueError('Duplicate source path across splits')
    if len(by)!=len(samples) or len(folds)!=n or [f['fold'] for f in folds]!=list(range(1,n+1)):raise ValueError('Invalid fold identity')
    for fold in folds:
        sets=[];gs=[];hs=[]
        for role in ('fit','monitor','outer'):
            ids=fold[role]
            if not ids or len(ids)!=len(set(ids)) or not set(ids)<=all_ids:raise ValueError('Empty/duplicate/unknown split')
            ss=[by[i] for i in ids];sets.append(set(ids));gs.append({s['scene'] for s in ss});hs.append({s['image_sha256'] for s in ss})
            for kind in ('image_path','label_path'):
                paths=[s[kind] for s in ss if kind in s]
                if len(paths)!=len(set(paths)):raise ValueError('Duplicate source path')
            if role!='outer' and any(features(ss)[i]==0 for i in range(1,5)):raise ValueError('Missing fit/monitor class')
        if set.union(*sets)!=all_ids:raise ValueError('Missing samples')
        for a in range(3):
            for b in range(a):
                if sets[a]&sets[b] or gs[a]&gs[b] or hs[a]&hs[b]:raise ValueError('Group/path/hash leakage')
        outer.extend(fold['outer'])
    if len(outer)!=len(samples) or set(outer)!=all_ids or len(set(outer))!=len(outer):raise ValueError('OOF coverage mismatch')

def environment(work):
    env={k:os.environ[k] for k in ('SystemRoot','WINDIR','PATH','PATHEXT','COMSPEC','NUMBER_OF_PROCESSORS','CUDA_VISIBLE_DEVICES') if k in os.environ}
    for k in ('HOME','USERPROFILE','APPDATA','LOCALAPPDATA','TEMP','TMP','XDG_CACHE_HOME','XDG_CONFIG_HOME','MPLCONFIGDIR','TORCH_HOME','TORCH_EXTENSIONS_DIR','CUDA_CACHE_PATH','NUMBA_CACHE_DIR','HF_HOME'):
        p=work/k.lower();p.mkdir();env[k]=str(p)
    settings=work/'config/Ultralytics/settings.json';settings.parent.mkdir(parents=True)
    obj=dict(settings_version='0.0.8',datasets_dir=str(work/'datasets'),weights_dir=str(work/'weights'),runs_dir=str(work/'runs'),uuid='0'*64,sync=False,api_key='',openai_api_key='')
    obj.update({k:False for k in ('clearml','comet','dvc','mlflow','raytune','tensorboard','wandb','vscode_msg','openvino_msg')});write(settings,obj)
    env.update(YOLO_CONFIG_DIR=str(work/'config'),YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_VERBOSE='false',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',WANDB_MODE='disabled',WANDB_DISABLED='true',COMET_MODE='DISABLED',TLC_COLLECTION_DISABLE='true')
    return env


def install_guard(c):
    work=Path(c['work']).resolve();allowed_roots=[ENV.resolve(),work]+[Path(p).resolve() for p in c['read_roots']]
    writes=[work]+[Path(p).resolve() for p in c['write_roots']]
    write_exact={Path(p).resolve() for p in c.get('write_files',[])}
    exact={Path(p).resolve() for p in c['read_files']}|{Path(__file__).resolve(),Path(c['config_path']).resolve()}
    weights={Path(p).resolve() for p in c['allowed_weights']}
    journal=work/f'guard-{os.getpid()}.jsonl'
    def denied(info):
        GUARD_EVENTS.append(info);append(journal,dict(time=now(),**info));raise PermissionError('Offline OOF guard: '+canonical(info))
    def hook(event,args):
        if event.startswith('socket.') and event in ('socket.connect','socket.connect_ex','socket.bind','socket.getaddrinfo','socket.sendto','socket.sendmsg'):
            denied(dict(event=event))
        if event in ('subprocess.Popen','os.system','os.posix_spawn'):denied(dict(event=event))
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            raw=os.fsdecode(args[0])
            if sys.platform=='win32' and raw in ('/etc/os-release','/etc/dgx-release','/proc/device-tree/model','/proc/cpuinfo'):raise FileNotFoundError('No non-Windows metadata read')
            p=Path(raw).resolve();mode=args[1];flags=args[2]
            writing=(isinstance(mode,str) and any(ch in mode for ch in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
            if str(p).lower()==os.devnull.lower():return
            if p.suffix.lower()=='.pt' and not writing and p not in weights:denied(dict(event='weight_read',path=str(p)))
            allowed=(within(p,writes) or p in write_exact) if writing else within(p,allowed_roots) or p in exact
            if not allowed:denied(dict(event='file',path=str(p),writing=bool(writing)))
    class No3LC(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('tlc','tlc_ultralytics','tlcconfig'):denied(dict(event='3lc_import',name=fullname))
    sys.meta_path.insert(0,No3LC());sys.addaudithook(hook)
    return GUARD_EVENTS

def bootstrap():
    p=Path(os.environ['OOF_CONFIG']).resolve()
    if not p.is_relative_to(OUT) or not p.name.endswith('_worker_config.json'):raise RuntimeError('Invalid worker config location')
    c=read(p)
    if c['token']!=os.environ.get('OOF_TOKEN') or Path(c['config_path']).resolve()!=p or c['role'] not in ('probe','train','predict'):raise RuntimeError('Private worker identity')
    if Path.cwd().resolve()!=Path(c['work']).resolve():raise RuntimeError('Private cwd')
    install_guard(c)
    return c


def load_runtime(c):
    import torch
    import ultralytics
    import numpy as np
    import PIL
    from ultralytics import YOLO
    from ultralytics.utils import ONLINE, SETTINGS, SETTINGS_FILE
    if Path(sys.executable).resolve()!=PYTHON.resolve() or sys.version_info[:3]!=(3,12,15):raise RuntimeError('Approved Python mismatch')
    if torch.__version__!='2.14.1+cu130' or ultralytics.__version__!='8.4.165' or torch.version.cuda!='13.0':raise RuntimeError('Approved package versions mismatch')
    if not torch.cuda.is_available() or 'RTX 4060 Laptop GPU' not in torch.cuda.get_device_name(0):raise RuntimeError('Approved GPU unavailable')
    if ONLINE or SETTINGS['sync'] or Path(SETTINGS_FILE).resolve()!=(Path(c['work'])/'config/Ultralytics/settings.json').resolve():raise RuntimeError('Isolation settings mismatch')
    meta=dict(python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,cuda=torch.version.cuda,numpy=np.__version__,pillow=PIL.__version__,device=torch.cuda.get_device_name(0),online=ONLINE,sync=SETTINGS['sync'],settings_file=str(SETTINGS_FILE))
    return torch,YOLO,meta


def train_worker(c):
    torch,YOLO,meta=load_runtime(c)
    from ultralytics.models.yolo.detect import DetectionTrainer
    import ultralytics.engine.trainer as engine
    import ultralytics.data.utils as data_utils
    from ultralytics.utils.torch_utils import init_seeds
    runroot=Path(c['runroot']);trace=runroot/'stopper_trace.jsonl';expected=c['training'];data_yaml=Path(c['data_yaml'])
    # Avoid font enumeration/download and the external YOLO26n AMP self-check.
    data_utils.check_font=lambda *a,**kw:None
    engine.check_amp=lambda model:True
    init_seeds(0,deterministic=True)
    stage=Path(c.get('stage_root',runroot));architecture=Path(c['architecture'])
    if architecture.name!='yolov8n.yaml' or not architecture.is_file() or sha(architecture)!=sha(ARCH):raise RuntimeError('Real local architecture copy missing/changed')
    write(stage/'model_construction_started.json',dict(fold=c['fold'],started_at=now(),architecture=str(architecture),architecture_sha256=sha(architecture)))
    model=YOLO(str(architecture),task='detect')
    write(stage/'model_constructed.json',dict(fold=c['fold'],finished_at=now(),initialization='random local architecture',architecture_sha256=sha(architecture)))
    if model.ckpt or model.ckpt_path is not None:raise RuntimeError('Unexpected pretrained checkpoint')
    training_meta=dict(runtime=meta,initialization='random local architecture only',external_amp_check_executed=False,amp_adaptation='Process-local check_amp returns enabled; no external image/weight self-test',nan_recovery_enabled=False,oom_recovery_enabled=False,started_at=now())
    class BoundedTrainer(DetectionTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            if weights is not None:raise RuntimeError('No pretrained or resumed model')
            return super().get_model(cfg=cfg,weights=None,verbose=verbose)
        def _handle_nan_recovery(self,epoch):
            for val in loss_values(self.loss)+loss_values(self.tloss):
                if not torch.isfinite(val).all():raise RuntimeError('Nonfinite loss: no recovery permitted')
            if self.fitness is not None and not math.isfinite(float(self.fitness)):raise RuntimeError('Nonfinite fitness: no recovery permitted')
            return False
        def save_model(self):
            for net in (self.model,self.ema.ema):
                for value in net.state_dict().values():
                    if isinstance(value,torch.Tensor) and value.is_floating_point():
                        if not torch.isfinite(value).all() or not torch.isfinite(value.half()).all():raise RuntimeError('Nonfinite/FP16-overflow weight; no repair permitted')
            return super().save_model()
        def final_eval(self):
            self.oof_final_eval=True
            return super().final_eval()
    def initialized(t):
        assert_effective(vars(t.args),expected)
        if str(t.device)!='cuda:0':raise RuntimeError('Actual training device drift')
        if t.args.resume or t.args.pretrained or t.args.patience!=8 or not t.amp:raise RuntimeError('Training wiring mismatch')
        if t.model.names!=NAMES or t.model.yaml.get('scale')!='n':raise RuntimeError('Architecture/class mismatch')
        params=sum(p.numel() for p in t.model.parameters())
        if params!=3011628:raise RuntimeError('YOLOv8n parameter count mismatch')
        if type(t.optimizer).__name__!='AdamW' or any(abs(pg['lr']-.00125)>1e-10 or pg['betas'][0]!=.9 for pg in t.optimizer.param_groups):raise RuntimeError('Actual optimizer differs from approved expectation')
        fit=read(c['fit_manifest']);monitor=read(c['monitor_manifest'])
        expected_fit={str(Path(c['data_root'])/'fit/images'/Path(s['image_path']).name) for s in fit}
        expected_mon={str(Path(c['data_root'])/'monitor/images'/Path(s['image_path']).name) for s in monitor}
        if {str(Path(p)) for p in t.train_loader.dataset.im_files}!=expected_fit or {str(Path(p)) for p in t.test_loader.dataset.im_files}!=expected_mon:raise RuntimeError('Dataloader membership drift')
        for role,dataset in [('fit',t.train_loader.dataset),('monitor',t.test_loader.dataset)]:
            expected_boxes=sum(len(s['labels']) for s in (fit if role=='fit' else monitor))
            if sum(len(item['cls']) for item in dataset.labels)!=expected_boxes:raise RuntimeError('Loader silently altered/dropped labels')
        initial=hashlib.sha256()
        for key,value in t.model.state_dict().items():initial.update(key.encode());initial.update(value.detach().cpu().contiguous().numpy().tobytes())
        training_meta.update(effective_args=vars(t.args),optimizer=type(t.optimizer).__name__,optimizer_initial_lr=[pg['lr'] for pg in t.optimizer.param_groups],betas=t.optimizer.param_groups[0]['betas'],parameters=params,amp_enabled=t.amp,initial_state_sha256=initial.hexdigest(),fit_images=len(expected_fit),monitor_images=len(expected_mon))
        write(runroot/'training_runtime.json',training_meta)
    def epoch_done(t):
        if getattr(t,'oof_final_eval',False):return
        if t.fitness is None or not math.isfinite(float(t.fitness)):raise RuntimeError('Invalid monitored fitness')
        if t.fitness!=t.metrics['metrics/mAP50-95(B)']:raise RuntimeError('Unexpected early stopping fitness')
        rec=dict(epoch=t.epoch+1,fitness=float(t.fitness),stopper_best_epoch=t.stopper.best_epoch,stopper_best_fitness=t.stopper.best_fitness,possible_stop=t.stopper.possible_stop,stop=bool(t.stop),patience=t.stopper.patience,best_checkpoint_updated=t.best_fitness==t.fitness,metrics={k:float(v) for k,v in t.metrics.items()},loss={k:float(v) for k,v in t.tloss.items()},time=now())
        append(trace,rec)
        print(f'OOF fold {c["fold"]} epoch {rec["epoch"]}/24 monitor_AP95={rec["fitness"]:.6f} stopper_best={rec["stopper_best_epoch"]} stop={rec["stop"]}',flush=True)
    def epoch_start(t):
        # Library checks >=3 before any batch-size reduction. Normal for/else resets it to0.
        t._oom_retries=3
    def batch_end(t):
        for val in loss_values(t.loss)+loss_values(t.tloss):
            if not torch.isfinite(val).all():raise RuntimeError('Nonfinite batch loss; stop without recovery')
    model.add_callback('on_train_start',initialized);model.add_callback('on_fit_epoch_end',epoch_done)
    model.add_callback('on_train_epoch_start',epoch_start);model.add_callback('on_train_batch_end',batch_end)
    # The real scaled YAML exists. No regex fallback, .pt probe, or external AMP check.
    write(stage/'training_invoked.json',dict(fold=c['fold'],started_at=now(),pretrained=False,resume=False,epochs=24,patience=8))
    model.train(trainer=BoundedTrainer,data=str(data_yaml),project=str(runroot),name='training',exist_ok=False,**expected)
    t=model.trainer;events=list(rows(trace))
    if not events or [r['epoch'] for r in events]!=list(range(1,len(events)+1)) or len(events)>24:raise RuntimeError('Unexpected epoch sequence')
    if len(events)<24 and not events[-1]['stop']:raise RuntimeError('Unexplained short training')
    best_epoch=max(r['epoch'] for r in events if r['best_checkpoint_updated'])
    for weight in (t.best,t.last):
        if not Path(weight).is_file():raise RuntimeError('Missing checkpoint')
        checkpoint=torch.load(weight,map_location='cpu',weights_only=False)
        net=checkpoint.get('ema') or checkpoint.get('model')
        if net is None or any(not torch.isfinite(x).all() for x in net.state_dict().values() if isinstance(x,torch.Tensor)):raise RuntimeError('Invalid checkpoint weights')
    result=dict(state='completed',fold=c['fold'],epochs=len(events),best_checkpoint_epoch=best_epoch,stopper_best_epoch=events[-1]['stopper_best_epoch'],early_stopped=len(events)<24,termination='patience8' if len(events)<24 else 'max_epochs24',best_file=str(t.best),last_file=str(t.last),best_sha256=sha(t.best),last_sha256=sha(t.last),monitor_best_metrics={k:float(v) for k,v in t.metrics.items()},runtime=meta,finished_at=now(),external_amp_selfcheck=False,nan_recovery_attempts=t.nan_recovery_attempts)
    if t.nan_recovery_attempts!=0:raise RuntimeError('Unexpected recovery')
    return result


def candidate(record_id,i,row):
    if len(row)!=6 or not all(math.isfinite(float(v)) for v in row):raise ValueError('Nonfinite candidate')
    x0,y0,x1,y1,conf,cl=map(float,row)
    if cl!=int(cl) or int(cl) not in NAMES or not (0<=x0<x1<=640 and 0<=y0<y1<=640 and .001<=conf<=1):raise ValueError('Invalid prediction box/class/confidence')
    return dict(id=record_id+f':p{i:03d}',cls=int(cl),class_name=NAMES[int(cl)],confidence=conf,xyxy=[x0,y0,x1,y1])

def validate_prediction(record,sample,fold,weight_hash,prediction):
    keys=['id','scene','frame','image_path','label_path','image_sha256','label_sha256']
    if any(record.get(k)!=sample[k] for k in keys) or record['fold']!=fold or record['model_sha256']!=weight_hash:raise ValueError('OOF identity drift')
    assert_effective(record['effective_params'],prediction)
    if record['input_shape']!=[1,3,640,640] or record['original_shape']!=[640,640]:raise ValueError('Prediction shape drift')
    if len(record['candidates'])>300:raise ValueError('max_det exceeded')
    for i,c in enumerate(record['candidates']):
        if c!=candidate(sample['id'],i,c['xyxy']+[c['confidence'],c['cls']]):raise ValueError('Candidate changed')


def predict_worker(c):
    torch,YOLO,meta=load_runtime(c)
    from PIL import Image
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    weight=Path(c['weight']);h=sha(weight)
    if h!=c['weight_sha256']:raise RuntimeError('Frozen best weight drift')
    model=YOLO(str(weight),task='detect')
    if model.names!=NAMES:raise RuntimeError('Wrong class mapping')
    shapes=[]
    class CheckedPredictor(DetectionPredictor):
        def preprocess(self,ims):
            out=super().preprocess(ims)
            if list(out.shape)!=[1,3,640,640] or out.dtype!=torch.float32:raise RuntimeError('Prediction input/dtype drift')
            reference=torch.from_numpy(ims[0][...,::-1].copy()).permute(2,0,1).unsqueeze(0).float()/255
            if not torch.allclose(out.cpu(),reference,rtol=0,atol=1e-7):raise RuntimeError('Unexpected prediction pixel preprocessing')
            shapes.append(list(out.shape));return out
    result_file=Path(c['runroot'])/'outer_predictions.jsonl'
    # Exclusive creation even for an empty/failed first sample.
    with result_file.open('x',encoding='utf-8'):pass
    for index,s in enumerate(c['samples']):
        p=ROOT/s['image_path']
        if sha(p)!=s['image_sha256']:raise RuntimeError('Outer image changed')
        with Image.open(p) as im:
            if im.size!=(640,640):raise ValueError('Unexpected original size')
            source=im.convert('RGB')
        rr=model.predict(source=source,predictor=CheckedPredictor,**c['prediction'],classes=None,save=False,save_txt=False,save_crop=False,show=False,verbose=False,project=str(Path(c['work'])/'runs'),name='outer')
        if len(rr)!=1 or list(rr[0].orig_shape)!=[640,640]:raise ValueError('Unexpected original prediction shape')
        effective={k:getattr(model.predictor.args,k) for k in c['prediction']}
        record={k:s[k] for k in ['id','scene','frame','image_path','label_path','image_sha256','label_sha256']}
        record.update(fold=c['fold'],model_sha256=h,input_shape=shapes[-1],original_shape=[640,640],effective_params=effective,candidates=[candidate(s['id'],i,row) for i,row in enumerate(rr[0].boxes.data.detach().cpu().tolist())])
        validate_prediction(record,s,c['fold'],h,c['prediction']);append(result_file,record)
        if (index+1)%100==0 or index+1==len(c['samples']):print(f'OOF fold {c["fold"]} outer {index+1}/{len(c["samples"])}',flush=True)
    if len(shapes)!=len(c['samples']):raise RuntimeError('Prediction coverage mismatch')
    return dict(state='completed',fold=c['fold'],images=len(shapes),weight_sha256=h,predictions_sha256=sha(result_file),runtime=meta,finished_at=now())


def worker():
    c=BOOT_CONFIG
    if c is None:raise RuntimeError('Private worker environment required')
    stage=Path(c['stage_root']);write(stage/'worker_started.json',dict(role=c['role'],fold=c.get('fold'),pid=os.getpid(),started_at=now()))
    started=time.perf_counter();result={}
    try:
        if c['role']=='probe':
            _,_,meta=load_runtime(c)
            from ultralytics.nn.tasks import yaml_model_load
            parsed=yaml_model_load(c['architecture'])
            if parsed.get('scale')!='n' or sha(c['architecture'])!=sha(ARCH):raise RuntimeError('Native local architecture parser check failed')
            result=dict(state='completed',runtime=meta,model_constructed=False,architecture_yaml_scale=parsed['scale'],architecture_yaml_source=parsed['yaml_file'])
        elif c['role']=='train':result=train_worker(c)
        elif c['role']=='predict':result=predict_worker(c)
        else:raise RuntimeError('Unknown worker role')
        if GUARD_EVENTS or any(k.split('.')[0] in ('tlc','tlcconfig','tlc_ultralytics') for k in sys.modules):raise RuntimeError('Isolation event')
    except BaseException:
        result.update(state='failed',exception=traceback.format_exc());raise
    finally:
        result.update(elapsed_seconds=time.perf_counter()-started,guard_events=GUARD_EVENTS,pid=os.getpid(),finished_at=now())
        write(stage/'worker_result.json',result)


class WindowsJob:
    """Kill-on-close job encloses each worker plus its Windows DataLoader children."""
    def __init__(self,process):
        if os.name!='nt':raise RuntimeError('Approved Windows runtime only')
        from ctypes import wintypes
        class BASIC(ctypes.Structure):
            _fields_=[('ProcessTime',ctypes.c_int64),('JobTime',ctypes.c_int64),('LimitFlags',wintypes.DWORD),('MinWorking',ctypes.c_size_t),('MaxWorking',ctypes.c_size_t),('Active',wintypes.DWORD),('Affinity',ctypes.c_size_t),('Priority',wintypes.DWORD),('Scheduling',wintypes.DWORD)]
        class IO(ctypes.Structure):_fields_=[(n,ctypes.c_uint64) for n in ('ReadOps','WriteOps','OtherOps','ReadBytes','WriteBytes','OtherBytes')]
        class EXT(ctypes.Structure):_fields_=[('Basic',BASIC),('Io',IO),('ProcessMemory',ctypes.c_size_t),('JobMemory',ctypes.c_size_t),('PeakProcessMemory',ctypes.c_size_t),('PeakJobMemory',ctypes.c_size_t)]
        k=ctypes.WinDLL('kernel32',use_last_error=True);self.k=k
        k.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR];k.CreateJobObjectW.restype=ctypes.c_void_p
        k.SetInformationJobObject.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD];k.SetInformationJobObject.restype=wintypes.BOOL
        k.AssignProcessToJobObject.argtypes=[ctypes.c_void_p,ctypes.c_void_p];k.AssignProcessToJobObject.restype=wintypes.BOOL
        k.CloseHandle.argtypes=[ctypes.c_void_p];k.CloseHandle.restype=wintypes.BOOL
        self.handle=k.CreateJobObjectW(None,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        limits=EXT();limits.Basic.LimitFlags=0x2000
        if not k.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)) or not k.AssignProcessToJobObject(self.handle,ctypes.c_void_p(int(process._handle))):
            err=ctypes.get_last_error();self.close();process.kill();process.wait();raise ctypes.WinError(err)
    def close(self):
        if self.handle:self.k.CloseHandle(self.handle);self.handle=None


def launch(role,stage,read_roots=(),read_files=(),write_roots=(),write_files=(),allowed_weights=(),timeout=60,**extra):
    stage=Path(stage)
    if (stage/'started.json').exists():raise FileExistsError('Existing attempt is immutable; recovery uses a fresh attempt namespace')
    stage.mkdir(parents=True,exist_ok=True)
    work=SCRATCH/('oof-'+role+'-'+uuid.uuid4().hex);work.mkdir();env=environment(work);token=uuid.uuid4().hex
    config_path=stage/(role+'_worker_config.json')
    c=dict(role=role,work=str(work),stage_root=str(stage),token=token,config_path=str(config_path),read_roots=list(map(str,[stage,*read_roots])),read_files=list(map(str,read_files)),write_roots=list(map(str,[stage,*write_roots])),write_files=list(map(str,write_files)),allowed_weights=list(map(str,allowed_weights)),**extra)
    write(config_path,c);env.update(OOF_CONFIG=str(config_path),OOF_TOKEN=token)
    write(stage/'started.json',dict(role=role,started_at=now(),timeout_seconds=timeout,work=str(work)))
    started=time.perf_counter();code=None;timed_out=False;error=None;job=None;process=None
    try:
        with (stage/'console.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen([str(PYTHON),'-I','-B','-u',str(Path(__file__).resolve()),'_worker'],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            job=WindowsJob(process)
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True;job.close();process.wait(timeout=30);code=process.returncode
    except BaseException as exc:error=repr(exc)
    finally:
        if job is not None:job.close()
        if process is not None and process.poll() is None:process.kill();process.wait()
    status=dict(role=role,exit_code=code,timed_out=timed_out,error=error,process_elapsed_seconds=time.perf_counter()-started,finished_at=now())
    write(stage/'status.json',status)
    guard_files=list(work.glob('guard-*.jsonl'))
    if guard_files:raise RuntimeError('Isolation events recorded: '+str(guard_files))
    result=read(stage/'worker_result.json') if (stage/'worker_result.json').exists() else None
    if code!=0 or timed_out or error or result is None or result['state']!='completed' or result['guard_events']:raise RuntimeError('Worker stopped; preserve evidence: '+str(stage))
    return result


def prepare():
    if OUT.exists():raise FileExistsError('Existing OOF evidence; do not overwrite')
    if Path(sys.executable).resolve()!=PYTHON.resolve():raise RuntimeError('Approved Python only')
    audit=module('oof_prepare_audit','tools/bus_static_audit.py');git=audit.git_snapshot()
    if git['branch']!='main' or git['head']!=HEAD:raise RuntimeError('Revision drift')
    c=cfg();started=time.time();prior=PREVIOUS/'plan_config.json'
    deadline=read(prior)['deadline_unix'] if prior.is_file() else started+c['budget']['batch_seconds']
    arm_deadline(deadline)
    original=originals_integrity();protected=protected_files();samples=inventory();folds=split_folds(samples)
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Need at least8GiB free for bounded copies/checkpoints')
    deps=dependency_hashes();OUT.mkdir();write(OUT/'preparation_started.json',dict(started_at=now(),deadline_unix=deadline,model_training_attempts=0,recovery_authorized=True,previous_attempt=str(PREVIOUS)))
    write(OUT/'preflight_snapshot.json',dict(configuration=c,source_sha256=source_hashes(),dependency_sha256=deps,integrity_before=original,protected_sha256=protected,git_before=git,created_at=now()))
    # Byte-identical, existing local file; n is selected by the filename as approved.
    architecture=materialize_architecture(OUT)
    probe=launch('probe',OUT/'preflight',read_files=[Path(architecture)],architecture=architecture,timeout=120)
    by={s['id']:s for s in samples};details=[]
    for fold in folds:
        directory=OUT/f'fold_{fold["fold"]:02d}';directory.mkdir();data=directory/'data';data.mkdir()
        for role in ('fit','monitor','outer'):
            ss=[by[i] for i in fold[role]];write(directory/(role+'_manifest.json'),ss)
            details.append(dict(fold=fold['fold'],role=role,images=len(ss),groups=len({s['scene'] for s in ss}),boxes_by_class=features(ss)[1:5],images_by_class=features(ss)[5:9]))
            if role=='outer':continue
            (data/role/'images').mkdir(parents=True);(data/role/'labels').mkdir()
            for s in ss:
                for kind in ('image','label'):
                    source=ROOT/s[kind+'_path'];destination=data/role/('images' if kind=='image' else 'labels')/source.name
                    if sha(source)!=s[kind+'_sha256']:raise ValueError('Original source changed')
                    with source.open('rb') as f,destination.open('xb') as g:shutil.copyfileobj(f,g)
                    if sha(destination)!=s[kind+'_sha256']:raise ValueError('Copy mismatch')
        import yaml
        with (directory/'data.yaml').open('x',encoding='utf-8') as f:yaml.safe_dump(dict(path=str(data),train='fit/images',val='monitor/images',names=NAMES),f,sort_keys=False)
        print(f'Prepared fold{fold["fold"]}: fit={len(fold["fit"])} monitor={len(fold["monitor"])} outer={len(fold["outer"])}',flush=True)
    write(OUT/'source_manifest.json',samples);write(OUT/'fold_manifest.json',folds)
    split=dict(seed=42,algorithm='MVI+hash union components; descending normalized rarity; greedy delta squared load/target cost; image dimension weight4, other8 dimensions weight1; inner greedy towards10%remaining with fit/monitor class coverage',monitor_fraction_of_remaining=.1,details=details,outer_unique_images=7852,group_hash_disjoint=True,near_duplicates_across_groups='not exhaustively checked')
    write(OUT/'split_verification.json',split)
    config=dict(created_at=now(),configuration=c,approved_plan_sha256=sha(PLAN),amendment_sha256=sha(AMENDMENT),source_sha256=source_hashes(),dependency_sha256=deps,inventory_sha256=sha(INVENTORY),architecture=architecture,architecture_sha256=sha(architecture),architecture_source=str(ARCH),runtime=probe['runtime'],integrity_before=original,protected_sha256=protected,git_before=git,deadline_unix=deadline,previous_attempt=str(PREVIOUS),authorization='Approved5-fold plan plus current user amendment: repair recoverable implementation failures and continue without renewed approval; preserve attempts; no relabeling or competition approval',plotting_note='plots=False disables visual/font side effects only; train augmentation/optimizer unchanged')
    write(OUT/'plan_config.json',config)
    if originals_integrity()!=original:raise RuntimeError('Originals changed during preparation')
    verify_hashes(protected)
    time_left(config,1)
    hashes={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()}
    write(OUT/'preparation_verified.json',dict(created_at=now(),artifact_sha256=hashes,images=7852,folds=5,training_attempts=0))
    print(json.dumps(dict(prepared=True,folds=details,deadline_unix=config['deadline_unix']),ensure_ascii=False,indent=2),flush=True)


def check_prepared(full=True):
    c=read(OUT/'plan_config.json');samples=read(OUT/'source_manifest.json');folds=read(OUT/'fold_manifest.json')
    if cfg()!=c['configuration'] or source_hashes()!=c['source_sha256'] or dependency_hashes()!=c['dependency_sha256']:raise ValueError('Frozen implementation/config/dependency drift')
    if samples!=inventory() or folds!=split_folds(samples) or sha(INVENTORY)!=c['inventory_sha256']:raise ValueError('Source/fold drift')
    validate_splits(samples,folds)
    verify_hashes(read(OUT/'preparation_verified.json')['artifact_sha256'],OUT)
    verify_hashes(c['protected_sha256'])
    if full and originals_integrity()!=c['integrity_before']:raise ValueError('Original dataset changed')
    return c,samples,folds


def time_left(c,limit):
    remaining=int(c['deadline_unix']-time.time())
    if remaining<=0:raise TimeoutError('Six-hour batch deadline reached')
    return min(limit,remaining)

def validate_fold_output(fold,samples,prediction,training):
    directory=OUT/f'fold_{fold["fold"]:02d}';tr=read(directory/'train_stage/worker_result.json');pr=read(directory/'predict_stage/worker_result.json')
    if tr['state']!='completed' or pr['state']!='completed' or tr['fold']!=fold['fold'] or not 1<=tr['epochs']<=24 or tr['nan_recovery_attempts']!=0:raise ValueError('Incomplete training/prediction')
    if sha(tr['best_file'])!=tr['best_sha256'] or sha(tr['last_file'])!=tr['last_sha256'] or pr['weight_sha256']!=tr['best_sha256']:raise ValueError('Model identity drift')
    rr=list(rows(directory/'outer_predictions.jsonl'));by={s['id']:s for s in samples}
    if [r['id'] for r in rr]!=fold['outer'] or len(rr)!=pr['images']:raise ValueError('Outer prediction gaps/duplicates/order')
    for r in rr:validate_prediction(r,by[r['id']],fold['fold'],tr['best_sha256'],prediction)
    rt=read(directory/'training_runtime.json')
    assert_effective(rt['effective_args'],training)
    if rt['optimizer']!='AdamW' or not rt['amp_enabled']:raise ValueError('Runtime training drift')
    trace=list(rows(directory/'stopper_trace.jsonl'))
    if [t['epoch'] for t in trace]!=list(range(1,tr['epochs']+1)) or any(t['patience']!=8 or not math.isfinite(t['fitness']) for t in trace):raise ValueError('Stopper trace drift')
    if tr['best_checkpoint_epoch']!=max(t['epoch'] for t in trace if t['best_checkpoint_updated']):raise ValueError('Best epoch drift')
    if sha(directory/'outer_predictions.jsonl')!=pr['predictions_sha256']:raise ValueError('Prediction stream drift')
    return tr,rr


def iou(a,b):
    area=lambda q:max(0,q[2]-q[0])*max(0,q[3]-q[1])
    inter=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]);union=area(a)+area(b)-inter
    return inter/union if union else 0.

def ground_truth(s):
    return [(int(cl),[(float(cx)-float(w)/2)*640,(float(cy)-float(h)/2)*640,(float(cx)+float(w)/2)*640,(float(cy)+float(h)/2)*640]) for cl,cx,cy,w,h in s['labels']]

def evaluate(records,samples):
    """CPU-only saved-record AP, exact same-class greedy-IoU matching; no model execution.
    This is an explicitly documented evaluator, not a weighted mean of fold AP.
    """
    import numpy as np
    by={s['id']:s for s in samples};thresholds=[.5+i*.05 for i in range(10)];truth=[0]*4;tp=[];scores=[];classes=[]
    for r in records:
        gt=ground_truth(by[r['id']]);truth=[truth[k]+sum(cl==k for cl,_ in gt) for k in range(4)]
        cc=r['candidates'];correct=[[False]*10 for _ in cc]
        for ti,t in enumerate(thresholds):
            matches=sorted([(iou(box,c['xyxy']),gi,pi) for gi,(cl,box) in enumerate(gt) for pi,c in enumerate(cc) if cl==c['cls'] and iou(box,c['xyxy'])>=t],reverse=True)
            used_g=set();used_p=set()
            for overlap,gi,pi in matches:
                if gi not in used_g and pi not in used_p:correct[pi][ti]=True;used_g.add(gi);used_p.add(pi)
        tp.extend(correct);scores.extend(c['confidence'] for c in cc);classes.extend(c['cls'] for c in cc)
    output=[]
    for cl in range(4):
        ids=sorted([i for i,c in enumerate(classes) if c==cl],key=lambda i:-scores[i]);aps=[]
        for t in range(10):
            if not ids or not truth[cl]:aps.append(0.);continue
            true=np.cumsum([tp[i][t] for i in ids]);false=np.cumsum([not tp[i][t] for i in ids]);rec=true/truth[cl];prec=true/(true+false)
            mr=np.concatenate(([0.],rec,[1.]));mp=np.concatenate(([1.],prec,[0.]));mp=np.flip(np.maximum.accumulate(np.flip(mp)))
            aps.append(float(np.trapezoid(np.interp(np.linspace(0,1,101),mr,mp),np.linspace(0,1,101))))
        output.append(dict(cls=cl,name=NAMES[cl],ground_truth=truth[cl],detections=len(ids),AP50=aps[0],AP50_95=sum(aps)/10))
    return dict(evaluator='Saved-record same-class global-per-image descending-IoU one-to-one greedy matching,101point interpolated precision trapezoid AP; conf>=.001 max_det300; not Ultralytics native matching/P-R operating point',classes=output,mAP50=sum(x['AP50'] for x in output)/4,mAP50_95=sum(x['AP50_95'] for x in output)/4,limits='Relative to original potentially noisy annotations; not label-quality truth, not official/test score, not weighted mean fold AP')


def render(records,samples,summary):
    data=dict(names=NAMES,samples=[],summary=summary)
    by={s['id']:s for s in samples}
    for r in records:
        s=by[r['id']];data['samples'].append(dict(r,labels=s['labels'],image_url=os.path.relpath(ROOT/s['image_path'],OUT).replace('\\','/')))
    text=TEMPLATE.read_text(encoding='utf-8')
    if text.count('__OOF_DATA__')!=1:raise ValueError('Review template marker')
    with (OUT/'review.html').open('x',encoding='utf-8') as f:f.write(text.replace('__OOF_DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')))


def run():
    if (OUT/'run_started.json').exists():raise FileExistsError('Existing invocation is immutable; recovery uses a fresh output namespace')
    arm_deadline(read(OUT/'plan_config.json')['deadline_unix'])
    c,samples,folds=check_prepared();by={s['id']:s for s in samples}
    write(OUT/'run_started.json',dict(started_at=now(),max_training_attempts=5,deadline_unix=c['deadline_unix']))
    state=dict(state='running',started_at=now(),completed_folds=[],training_attempts=0,outer_images=0)
    update(OUT/'progress.json',state);records=[];summary=[]
    try:
        for fold in folds:
            if source_hashes()!=c['source_sha256'] or dependency_hashes()!=c['dependency_sha256']:raise ValueError('Implementation drift during run')
            if sha(c['architecture'])!=c['architecture_sha256']:raise ValueError('Frozen local architecture changed')
            directory=OUT/f'fold_{fold["fold"]:02d}';data=directory/'data';runroot=directory
            # Validate frozen copies, configuration and disjoint sets before spending an attempt.
            hashes=read(OUT/'preparation_verified.json')['artifact_sha256']
            verify_hashes({k:v for k,v in hashes.items() if k.startswith(directory.name+'/')},OUT)
            state.update(current_fold=fold['fold'],phase='training',training_attempts=state['training_attempts']+1);update(OUT/'progress.json',state)
            tr=launch('train',directory/'train_stage',read_roots=[data,directory/'training'],read_files=[directory/'data.yaml',directory/'fit_manifest.json',directory/'monitor_manifest.json',directory/'stopper_trace.jsonl',Path(c['architecture'])],write_roots=[data,directory/'training'],write_files=[directory/'training_runtime.json',directory/'stopper_trace.jsonl'],allowed_weights=[directory/'training/weights/best.pt',directory/'training/weights/last.pt'],timeout=time_left(c,2700),fold=fold['fold'],runroot=str(runroot),data_root=str(data),data_yaml=str(directory/'data.yaml'),fit_manifest=str(directory/'fit_manifest.json'),monitor_manifest=str(directory/'monitor_manifest.json'),architecture=c['architecture'],training=c['configuration']['training'])
            # Worker stage can write fold outputs (trace/runtime) but never read outer manifest or original data.
            state.update(phase='outer_prediction');update(OUT/'progress.json',state)
            outer=[by[i] for i in fold['outer']]
            launch('predict',directory/'predict_stage',read_files=[ROOT/s['image_path'] for s in outer]+[Path(tr['best_file']),directory/'outer_predictions.jsonl'],write_files=[directory/'outer_predictions.jsonl'],allowed_weights=[tr['best_file']],timeout=time_left(c,900),fold=fold['fold'],runroot=str(runroot),weight=tr['best_file'],weight_sha256=tr['best_sha256'],samples=outer,prediction=c['configuration']['prediction'])
            tr,rr=validate_fold_output(fold,samples,c['configuration']['prediction'],c['configuration']['training']);records.extend(rr)
            metrics=evaluate(rr,outer);write(directory/'outer_metrics.json',metrics)
            row=dict(fold=fold['fold'],fit=len(fold['fit']),monitor=len(fold['monitor']),outer=len(fold['outer']),epochs=tr['epochs'],best_epoch=tr['best_checkpoint_epoch'],stopper_best_epoch=tr['stopper_best_epoch'],early_stopped=tr['early_stopped'],training_seconds=tr['elapsed_seconds'],best_file=tr['best_file'],model_sha256=tr['best_sha256'],monitor_metrics_native=tr['monitor_best_metrics'],outer_metrics_saved_records=metrics)
            summary.append(row);state['completed_folds'].append(fold['fold']);state['outer_images']=len(records);update(OUT/'progress.json',state)
            print(f'Completed fold{fold["fold"]} epochs={tr["epochs"]} best={tr["best_checkpoint_epoch"]} outer_total={len(records)}/7852',flush=True)
        if len(records)!=7852 or len({r['id'] for r in records})!=7852:raise ValueError('Merged coverage failure')
        with (OUT/'oof_predictions.jsonl').open('x',encoding='utf-8') as f:
            for r in records:f.write(canonical(r)+'\n')
        write(OUT/'OOF_INDEX.json',dict(images=7852,records=[dict(id=r['id'],fold=r['fold'],model_sha256=r['model_sha256'],empty=not r['candidates']) for r in records]))
        write(OUT/'fold_summary.json',summary);write(OUT/'oof_metrics.json',evaluate(records,samples));render(records,samples,summary)
        finish_verification(c)
        time_left(c,1)
        state.update(state='completed_awaiting_visual_review',finished_at=now(),phase='verified');update(OUT/'progress.json',state)
        write(OUT/'run_status.json',state)
        text='# 5折OOF离线诊断完成\n\n'
        text+='每折只用fit更新参数、monitor决定早停/best；outer在权重冻结后才预测。没有使用val966或test。没有自动判错/改标签、模型集成或第六次训练。\n\n'
        text+='|折|fit|monitor|outer|轮数|best轮|早停|训练秒|\n|---|---:|---:|---:|---:|---:|---|---:|\n'
        for r in summary:text+=f'|{r["fold"]}|{r["fit"]}|{r["monitor"]}|{r["outer"]}|{r["epochs"]}|{r["best_epoch"]}|{r["early_stopped"]}|{r["training_seconds"]:.2f}|\n'
        text+='\n全体7852张每图唯一一次OOF预测，含空预测，低置信结果完整保留。指标相对于可能错漏的原标签，不是数据质量真值；场景外分布变化、近重复残留、内层标签噪声和24轮能力限制仍存在。oof_metrics.json为明示匹配规则的统一saved-record评估，不冒充原生Ultralytics精确匹配或加权平均AP。\n\n打开review.html查看原图四类原标签与预测；预测阈值仅用于显示。不收集逐框JSON，不自动给错误概率排序。\n\nSS03训练增强保留；plots=false仅避免字体/画图副作用。AMP启用但跳过外部权重自检，没有声称自检通过。Python层I/O/network guard是纵深防御，非原生OS沙箱。\n\n训练/预测产物仅作本次获准离线诊断，未获主办方独立训练入口认可，不用于伪标签/提交或竞赛集成。未改官方入口/默认配置/AGENTS/原标签，不安装依赖、联网或读取3LC凭据。\n\n只读复验：\n```powershell\n& \'E:\\anaconda\\envs\\dl-3lc\\python.exe\' -I -B .\\tools\\oof_diagnostic.py verify\n& \'E:\\anaconda\\envs\\dl-3lc\\python.exe\' -I -B -m unittest discover -s .\\tests -p \'test_oof*.py\'\n```\n'
        with (OUT/'RESULT.md').open('x',encoding='utf-8') as f:f.write(text)
        files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()}
        write(OUT/'verification.json',dict(verified_at=now(),artifact_sha256=files,original_files=17668,unique_oof_images=7852,training_attempts=5,source_sha256=source_hashes(),limits='No human label adjudication; near-duplicate isolation beyond exact hashes not established'))
        print('OOF complete; read-only verification and visual review can proceed.',flush=True)
    except BaseException:
        state.update(state='paused_recovery_authorized',finished_at=now(),exception=traceback.format_exc());update(OUT/'progress.json',state)
        write(OUT/'run_status.json',state)
        try:finish_verification(c);write(OUT/'failure_integrity.json',dict(originals_and_protected_unchanged=True,checked_at=now()))
        except BaseException as exc:write(OUT/'failure_integrity.json',dict(originals_and_protected_unchanged=False,error=repr(exc)))
        raise


def finish_verification(c):
    if originals_integrity()!=c['integrity_before']:raise ValueError('Originals changed')
    verify_hashes(c['protected_sha256'])
    if source_hashes()!=c['source_sha256'] or dependency_hashes()!=c['dependency_sha256']:raise ValueError('Source/dependency changed')
    # Derived images and labels must stay byte-identical even if library tries to repair corrupted JPEGs.
    hashes=read(OUT/'preparation_verified.json')['artifact_sha256'];verify_hashes(hashes,OUT)


def verify():
    c,samples,folds=check_prepared();state=read(OUT/'run_status.json')
    if state['state']!='completed_awaiting_visual_review' or state['training_attempts']!=5 or state['outer_images']!=7852:raise ValueError('Run not complete')
    rr=[]
    for f in folds:
        tr,records=validate_fold_output(f,samples,c['configuration']['prediction'],c['configuration']['training']);rr.extend(records)
    if list(rows(OUT/'oof_predictions.jsonl'))!=rr or len({r['id'] for r in rr})!=7852:raise ValueError('Merged record identity drift')
    if read(OUT/'oof_metrics.json')!=evaluate(rr,samples):raise ValueError('Saved-record metric drift')
    verify_hashes(read(OUT/'verification.json')['artifact_sha256'],OUT);finish_verification(c)
    print(json.dumps(dict(verified=True,images=7852,folds=5,original_files=17668,model_reexecuted=False),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','verify','_worker']);a=parser.parse_args()
    try:{'prepare':prepare,'run':run,'verify':verify,'_worker':worker}[a.command]()
    finally:
        if DEADLINE_TIMER is not None:DEADLINE_TIMER.cancel()

# Windows multiprocessing executes the main file as __mp_main__ before unpickling
# the DataLoader. Reinstall the same role-specific guard before importing torch.
if os.environ.get('OOF_CONFIG'):BOOT_CONFIG=bootstrap()
if __name__=='__main__':main()
