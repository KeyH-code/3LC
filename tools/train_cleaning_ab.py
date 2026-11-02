"""One explicitly approved local SS04 A/B experiment; never imports 3LC."""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/train-cleaning-ab-ss04-20261029'
CLEAN = ROOT/'reports/oof-label-cleaning-conf085-20261027'
PYTHON = Path(r'E:\anaconda\envs\dl-3lc\python.exe')
NAMES = {0: 'truck', 1: 'car', 2: 'van', 3: 'bus'}
AUTHORIZATION = ROOT/'logs/train-cleaning-ab-ss04-20261029.md'


def training_options():
    return dict(epochs=28, patience=100, batch=32, imgsz=640, device=0, workers=2,
                seed=0, deterministic=True, pretrained=False, resume=False, optimizer='auto',
                lr0=.01, lrf=.01, momentum=.937, weight_decay=.0005, warmup_epochs=3.,
                warmup_momentum=.8, warmup_bias_lr=.1, nbs=64, cos_lr=False,
                hsv_h=.015, hsv_s=.7, hsv_v=.4, degrees=0., translate=.1, scale=.5,
                shear=0., perspective=0., flipud=0., fliplr=.5, bgr=0., mosaic=1.,
                mixup=.05, cutmix=0., copy_paste=.1, close_mosaic=10, amp=True,
                cache=False, rect=False, compile=False, fraction=1., val=True,
                save=True, save_period=-1, plots=False, iou=.7, max_det=300,
                box=7.5, cls=.5, cls_pw=0., dfl=1.5, single_cls=False, classes=None,
                freeze=None, multi_scale=0., channels_last=None, quantize=None,
                distill_model=None, copy_paste_mode='flip')


def assert_training_contract(actual, optimizer, rates, betas, amp):
    for key, expected in training_options().items():
        if key == 'warmup_bias_lr': expected = 0.  # installed auto optimizer's documented mutation
        value = actual.get(key)
        if key == 'device' and str(value) in ['0', 'cuda:0']: value = 0
        if value != expected: raise RuntimeError('Training parameter drift: '+key)
    if optimizer != 'AdamW' or any(abs(v-.00125) > 1e-9 for v in rates) or betas != (.9, .999):
        raise RuntimeError('SS04 optimizer differs')
    if not amp: raise RuntimeError('SS04 requires AMP enabled')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_rows(rows, kinds=('image','label')):
    for row in rows:
        for kind in kinds:
            if sha(row[kind])!=row[kind+'_sha256']:
                raise RuntimeError('Experiment data drift: '+row[kind])


def assert_loader(dataset, rows, boxes, ordered=True):
    actual=[str(Path(p)) for p in dataset.im_files]
    expected=sorted(str(Path(r['image'])) for r in rows)
    if (actual if ordered else sorted(actual)) != expected:
        raise RuntimeError('Loader image membership/order drift')
    if sum(len(r['cls']) for r in dataset.labels)!=boxes:
        raise RuntimeError('Loader labels changed/dropped')


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def support():
    spec = importlib.util.spec_from_file_location('ab_offline_support', ROOT/'tools/oof_diagnostic.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_pair(source, derived):
    if len(source) != 7852 or len(derived) != 7852:
        raise ValueError('Expected 7852 train images per arm')
    by_id = {r['id']: r for r in derived}
    if len(by_id) != 7852 or {r['id'] for r in source} != set(by_id):
        raise ValueError('Image identity mismatch')
    if any(r['image_sha256'] != by_id[r['id']]['image_sha256'] for r in source):
        raise ValueError('Train image bytes differ')
    return by_id


def write_yaml(path, train, val):
    import yaml
    payload = {'path': str(OUT), 'train': str(train), 'val': str(val), 'nc': 4, 'names': NAMES}
    with path.open('x', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def validate_val(vals):
    if len(vals) != 966 or len(set(vals)) != 966:
        raise ValueError('Validation membership drift')


def prepare():
    if OUT.exists():
        raise FileExistsError('Run directory exists; preserve evidence, do not prepare again')
    source = [r for r in load(ROOT/'reports/oof-diagnostic-5fold-24e-retry1-20261026/source_manifest.json') if r['split'] == 'train']
    derived = load(CLEAN/'dataset_manifest.json')['rows']; by_id = validate_pair(source, derived)
    if sha(CLEAN/'dataset_manifest.json') != '1898d635ab9808c00cab3ccc6dcbf30a564820393b91f260775204711de2c76e':
        raise ValueError('Approved cleaned manifest drift')
    if shutil.disk_usage(ROOT).free < 2_000_000_000:
        raise RuntimeError('Need 2GB free for independent paired data/output')
    OUT.mkdir(); (OUT/'runtime').mkdir(); support().materialize_architecture(OUT)
    protected = load(CLEAN/'input_integrity.json')['protected'].copy()
    for row in derived:
        for kind in ['image','label']:
            protected[(CLEAN/row[kind]).relative_to(ROOT).as_posix()] = row[kind+'_sha256']
    for name in ['dataset_manifest.json','approval_record.json','applied_changes.json','phase-b-independent-verification.json']:
        protected[(CLEAN/name).relative_to(ROOT).as_posix()] = sha(CLEAN/name)
    source = sorted(source, key=lambda r: Path(r['image_path']).name)
    sample_manifest = []
    for arm in ['A','B']:
        images = OUT/f'data/{arm}/images/train'; labels = OUT/f'data/{arm}/labels/train'
        images.mkdir(parents=True); labels.mkdir(parents=True)
        for r in source:
            dr = by_id[r['id']]
            src_image = ROOT/r['image_path'] if arm == 'A' else CLEAN/dr['image']
            src_label = ROOT/r['label_path'] if arm == 'A' else CLEAN/dr['label']
            expected_label = r['label_sha256'] if arm == 'A' else dr['label_sha256']
            if sha(src_image) != r['image_sha256'] or sha(src_label) != expected_label:
                raise ValueError('Source hash mismatch')
            target_image, target_label = images/src_image.name, labels/src_label.name
            shutil.copyfile(src_image,target_image); shutil.copyfile(src_label,target_label)
            if sha(target_image) != r['image_sha256'] or sha(target_label) != expected_label:
                raise ValueError('Copied data mismatch')
            if target_image.stat().st_ino == src_image.stat().st_ino or target_label.stat().st_ino == src_label.stat().st_ino:
                raise ValueError('Independent copies required')
            sample_manifest.append(dict(arm=arm,id=r['id'],image=str(target_image),label=str(target_label),
                                        image_sha256=r['image_sha256'],label_sha256=expected_label))
    kit = ROOT/'multi-vehicle_kaggle_competition_starter_kit_files'
    vals = (ROOT/'reports/data-audit/random-label-review-20261024/val_use_list_excluding_16.relative.txt').read_text(encoding='utf-8').splitlines()
    validate_val(vals)
    val_images=OUT/'data/val/images/val';val_labels=OUT/'data/val/labels/val'
    val_images.mkdir(parents=True);val_labels.mkdir(parents=True)
    val_manifest=[]
    original_manifest={str(Path(r['relative'])):r for r in load(ROOT/'reports/data_manifest.json')}
    for rel in vals:
        image=kit/rel;label=kit/'data/val/labels'/image.with_suffix('.txt').name
        for src,dest in [(image,val_images/image.name),(label,val_labels/label.name)]:
            expected=original_manifest[str(src.relative_to(kit/'data'))]['sha256']
            if sha(src)!=expected:raise ValueError('Val source drift')
            shutil.copyfile(src,dest)
            if sha(dest)!=expected:raise ValueError('Val copy drift')
        val_manifest.append(dict(image=str(val_images/image.name),label=str(val_labels/label.name),image_sha256=sha(image),label_sha256=sha(label)))
    for arm in ['A','B']:
        write_yaml(OUT/f'{arm}.yaml',OUT/f'data/{arm}/images/train',val_images)
    for ref in ['A','B']:
        write_yaml(OUT/f'eval_train_{ref}.yaml',OUT/f'data/{ref}/images/train',OUT/f'data/{ref}/images/train')
    save(OUT/'train_manifest.json',sample_manifest);save(OUT/'val_manifest.json',val_manifest)
    save(OUT/'protected.json',protected)
    for name,expected in protected.items():
        if sha(ROOT/name)!=expected:raise ValueError('Protected source/history changed: '+name)
    save(OUT/'plan.json',dict(created_at=stamp(),authorization='User 2026-10-29 explicitly requests full A/B training from scratch, SS04 configuration, local Ultralytics only.',
         training=training_options(),arms=['A','B'],train_images=7852,val_images=966,original_boxes=72273,clean_boxes=72665,
         changed_files=339,added_boxes=392,changed_classes=58,trainer_seed=0,entry_initial_seed=42,
         assessment='Six native evaluations: each best on original train, cleaned train, same val966. Fixed P/R confidence .25 IoU .50 is a descriptive common reference, not a tuned threshold or success gate.',
         training_timeout_seconds=5400,evaluation_timeout_seconds=3600,total_timeout_seconds=18000,
         native_best_selection='val966 mAP50-95, same for both; no extra seeds or tuning',
         deviations_from_historical_SS04=['Both use ordinary Ultralytics, val966 not historical val982; plots=false; pretrained=false explicitly; external AMP weight self-check skipped, AMP remains enabled.'],
         source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),ROOT/'tests/test_train_cleaning_ab.py',ROOT/'tools/oof_diagnostic.py',AUTHORIZATION,
             OUT/'A.yaml',OUT/'B.yaml',OUT/'eval_train_A.yaml',OUT/'eval_train_B.yaml',OUT/'architecture/yolov8n.yaml',OUT/'train_manifest.json',OUT/'val_manifest.json',OUT/'protected.json']},
         competition_approved=False,training_runs_authorized=2))
    print('PREPARED A/B7852 val966; originals and cleaned sources unchanged',flush=True)


def allowed_file(path, writing):
    p=Path(path).resolve()
    roots=[OUT.resolve()] if writing else [OUT.resolve(),PYTHON.parent.resolve()]
    exact={Path(__file__).resolve(),(ROOT/'tools/oof_diagnostic.py').resolve()}
    return any(p==r or p.is_relative_to(r) for r in roots) or (not writing and p in exact)


def guard(work):
    import importlib.abc
    events=[]
    def deny(event, **details):
        record=dict(at=stamp(), event=event, **details); events.append(record)
        with (work/f'guard_events_{os.getpid()}.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record)+'\n')
        raise PermissionError('Offline A/B guard: '+json.dumps(record))
    def hook(event,args):
        if event in ['socket.connect','socket.connect_ex','socket.getaddrinfo','socket.bind','socket.sendto','subprocess.Popen','os.system']:
            deny(event)
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            raw=os.fsdecode(args[0])
            if raw in ['/etc/os-release','/etc/dgx-release','/proc/cpuinfo','/proc/device-tree/model']:
                raise FileNotFoundError(raw)
            if raw.lower() in ['nul',r'\\.\nul']:
                return
            p=Path(raw).resolve();mode=args[1];flags=args[2]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
            if not allowed_file(p, writing): deny('file',path=str(p),writing=bool(writing))
    class No3LC(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ['tlc','tlc_ultralytics','tlcconfig']:
                deny('3lc_import',module=fullname)
    sys.meta_path.insert(0,No3LC());sys.addaudithook(hook)
    return events


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(arm,stage):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    import ultralytics.engine.trainer as engine
    import ultralytics.data.utils as data_utils
    from ultralytics.utils.torch_utils import init_seeds
    data_utils.check_font=lambda *a,**k:None
    engine.check_amp=lambda model:True
    init_seeds(42,deterministic=True)
    model=YOLO(str(OUT/'architecture/yolov8n.yaml'),task='detect')
    if model.ckpt or model.ckpt_path:raise RuntimeError('Random initialization only')
    meta={};epochs=[]
    class PairedTrainer(DetectionTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            if weights is not None:raise RuntimeError('Pretrained/resume prohibited')
            return super().get_model(cfg=cfg,weights=None,verbose=verbose)
        def _handle_nan_recovery(self,epoch):
            values=self.loss.values() if isinstance(self.loss,dict) else [self.loss]
            if any(not torch.isfinite(v).all() for v in values):raise RuntimeError('Nonfinite loss; preserve failure')
            if self.fitness is not None and not math.isfinite(float(self.fitness)):raise RuntimeError('Nonfinite fitness')
            return False
        def final_eval(self):
            self.ab_final_eval=True
            return super().final_eval()
        def save_model(self):
            for net in (self.model, self.ema.ema):
                for value in net.state_dict().values():
                    if value.is_floating_point() and (not torch.isfinite(value).all() or not torch.isfinite(value.half()).all()):
                        raise RuntimeError('Nonfinite/FP16-overflow weight; do not silently repair')
            return super().save_model()
    def initialized(t):
        assert_training_contract(vars(t.args), type(t.optimizer).__name__,
                                 [p['lr'] for p in t.optimizer.param_groups],
                                 t.optimizer.param_groups[0]['betas'], t.amp)
        if t.model.names!=NAMES or sum(p.numel() for p in t.model.parameters())!=3011628:
            raise RuntimeError('YOLOv8n class/parameter drift')
        expected=[r for r in load(OUT/'train_manifest.json') if r['arm']==arm]
        assert_loader(t.train_loader.dataset,expected,{'A':72273,'B':72665}[arm])
        assert_loader(t.test_loader.dataset,load(OUT/'val_manifest.json'),11487,ordered=False)
        boxes=sum(len(r['cls']) for r in t.train_loader.dataset.labels)
        if boxes!={'A':72273,'B':72665}[arm]:raise RuntimeError('Train labels changed/dropped')
        meta.update(initial_state_sha256=state_hash(t.model),effective_args=vars(t.args),optimizer=type(t.optimizer).__name__,
                    initial_lr=[p['lr'] for p in t.optimizer.param_groups],betas=t.optimizer.param_groups[0]['betas'],train_boxes=boxes,
                    train_images=7852,val_images=966,val_boxes=11487,amp=bool(t.amp),parameters=3011628)
        save(stage/'initialization.json',meta)
        if arm=='B' and meta['initial_state_sha256']!=load(OUT/'A/train/initialization.json')['initial_state_sha256']:
            raise RuntimeError('A/B initial parameters differ')
    def epoch_start(t):t._oom_retries=3
    def epoch_end(t):
        if getattr(t,'ab_final_eval',False):return
        row={'epoch':t.epoch+1,'metrics':{k:float(v) for k,v in t.metrics.items()},'fitness':float(t.fitness),
             'best_updated':t.best_fitness==t.fitness,'at':stamp()}
        epochs.append(row)
        with (stage/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(f'ARM {arm} epoch {t.epoch+1}/28 AP50={t.metrics.get("metrics/mAP50(B)")} AP95={t.fitness}',flush=True)
    model.add_callback('on_train_start',initialized);model.add_callback('on_train_epoch_start',epoch_start)
    model.add_callback('on_fit_epoch_end',epoch_end)
    model.train(trainer=PairedTrainer,data=str(OUT/f'{arm}.yaml'),project=str(OUT/arm),name='model',exist_ok=False,**training_options())
    if len(epochs)!=28:raise RuntimeError('Not all 28 epochs completed')
    trainer=model.trainer
    return dict(arm=arm,epochs=28,best_epoch=max(r['epoch'] for r in epochs if r['best_updated']),
                best=str(trainer.best),last=str(trainer.last),best_sha256=sha(trainer.best),last_sha256=sha(trainer.last),
                initial_state_sha256=meta['initial_state_sha256'],external_amp_selfcheck=False,
                native_final_metrics={k:float(v) for k,v in trainer.metrics.items()})


def fixed_metrics(stats,threshold=.25):
    import numpy as np
    tp=np.concatenate(stats['tp'],axis=0);conf=np.concatenate(stats['conf']);pred=np.concatenate(stats['pred_cls']);truth=np.concatenate(stats['target_cls'])
    result=[]
    for cls in range(4):
        chosen=(pred==cls)&(conf>=threshold);true=int(tp[chosen,0].sum());count=int(chosen.sum());gt=int((truth==cls).sum())
        result.append(dict(cls=cls,name=NAMES[cls],TP=true,FP=count-true,FN=gt-true,
                           precision=true/count if count else 0.,recall=true/gt if gt else 0.))
    return dict(confidence=threshold,iou=.5,classes=result,precision_macro=sum(r['precision'] for r in result)/4,
                recall_macro=sum(r['recall'] for r in result)/4)


def scored_validator_factory(base, capture):
    class ScoredValidator(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            capture['validator'] = self
            self.fixed_stats = {k: [] for k in ['tp', 'conf', 'pred_cls', 'target_cls']}

        def update_metrics(self, preds, batch):
            # Filter first, then match again: low-confidence boxes cannot steal a match.
            for si, pred in enumerate(preds):
                prepared = self._prepare_batch(si, batch)
                filtered = {k: v[pred['conf'] >= .25] for k, v in pred.items()}
                filtered = self._prepare_pred(filtered)
                self.fixed_stats['tp'].append(self._process_batch(filtered, prepared)['tp'])
                self.fixed_stats['conf'].append(filtered['conf'].cpu().numpy())
                self.fixed_stats['pred_cls'].append(filtered['cls'].cpu().numpy())
                self.fixed_stats['target_cls'].append(prepared['cls'].cpu().numpy())
            return super().update_metrics(preds, batch)

        def init_metrics(self, model):
            if 'expected_rows' in capture:
                assert_loader(self.dataloader.dataset,capture['expected_rows'],capture['expected_boxes'])
            result=super().init_metrics(model)
            if self.args.max_det!=300:raise RuntimeError('Evaluator silently changed max_det')
            return result

        def get_stats(self):
            self.common_fixed = fixed_metrics(self.fixed_stats)
            return super().get_stats()
    return ScoredValidator


def evaluate(arm,reference,stage):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    import ultralytics.data.utils as data_utils
    data_utils.check_font=lambda *a,**k:None
    expected_rows=load(OUT/'val_manifest.json') if reference=='val966' else [r for r in load(OUT/'train_manifest.json') if r['arm']==reference]
    capture = {'expected_rows':expected_rows,'expected_boxes':11487 if reference=='val966' else {'A':72273,'B':72665}[reference]}
    ScoredValidator = scored_validator_factory(DetectionValidator, capture)
    yaml_file=OUT/f'{arm}.yaml' if reference=='val966' else OUT/f'eval_train_{reference}.yaml'
    weight=OUT/arm/'model/weights/best.pt'
    model=YOLO(str(weight),task='detect')
    metrics=model.val(validator=ScoredValidator,data=str(yaml_file),split='val',imgsz=640,batch=32,
                      device=0,workers=2,conf=.001,iou=.7,max_det=300,quantize=None,rect=False,
                      augment=False,plots=False,save_json=False,save_txt=False,verbose=False,
                      project=str(stage),name='native',exist_ok=False)
    v=capture['validator']
    expected=966 if reference=='val966' else 7852
    if v.seen!=expected:raise RuntimeError('Evaluation coverage differs')
    per=[]
    for i,cls in enumerate(metrics.box.ap_class_index):
        per.append(dict(cls=int(cls),name=NAMES[int(cls)],precision_best_f1=float(metrics.box.p[i]),
                        recall_best_f1=float(metrics.box.r[i]),AP50=float(metrics.box.all_ap[i,0]),
                        AP50_95=float(metrics.box.all_ap[i].mean())))
    return dict(arm=arm,reference=reference,images=v.seen,weight_sha256=sha(weight),
                native={k:float(value) for k,value in metrics.results_dict.items()},classes=per,
                fixed=v.common_fixed,precision='FP32 quantize=None',evaluator='Ultralytics8.4.165 native matching/AP; best-F1 P/R separately labeled')


def stage_path(role, arm, reference):
    if role == 'preflight': return OUT/'preflight'
    return OUT/arm/(role if role=='train' else 'eval_'+reference)


def preflight(stage):
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import DEFAULT_CFG_DICT
    from types import SimpleNamespace
    # Exercise the installed native matcher without constructing a neural model.
    native=object.__new__(DetectionValidator)
    native.niou=10; native.iouv=torch.linspace(.5,.95,10)
    truth={'cls':torch.tensor([1.]),'bboxes':torch.tensor([[0.,0.,10.,10.]])}
    pred={'cls':torch.tensor([1.,1.]),'bboxes':torch.tensor([[0.,0.,10.,10.],[0.,0.,9.,10.]]),
          'conf':torch.tensor([.1,.9]),'extra':torch.zeros((2,0))}
    full=native._process_batch(pred,truth)['tp']
    filtered={k:v[pred['conf']>=.25] for k,v in pred.items()}
    high=native._process_batch(filtered,truth)['tp']
    if int(full[1,0])!=0 or int(high[0,0])!=1:
        raise RuntimeError('Installed native threshold/matching semantics changed')
    capture={};factory=scored_validator_factory(DetectionValidator,capture)
    validator=factory(args={'project':str(stage),'name':'native-contract','plots':False,'conf':.001,'single_cls':False})
    validator.device=torch.device('cpu');validator.niou=10;validator.iouv=native.iouv
    batch={'batch_idx':torch.tensor([0]),'cls':torch.tensor([[1.]]),
           'bboxes':torch.tensor([[.5,.5,1.,1.]]),'ori_shape':[(10,10)],
           'img':torch.zeros((1,3,10,10)),'ratio_pad':[None],'im_file':['synthetic.jpg']}
    # Only the paired extension is tested here; native update requires dataset init.
    prepared=validator._prepare_batch(0,batch)
    fixed=native._process_batch(filtered,prepared)['tp']
    if int(fixed[0,0])!=1 or capture['validator'] is not validator:
        raise RuntimeError('Native validator extension contract differs')
    for k,v in training_options().items():
        if k not in DEFAULT_CFG_DICT:raise RuntimeError('Unsupported training option: '+k)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(4)),batch_size=1,num_workers=2)
    if sum(int(x[0].item()) for x in loader)!=6:raise RuntimeError('Spawn DataLoader failed')
    children=[load(p) for p in Path.cwd().glob('guard_ready_*.json') if load(p)['module']=='__mp_main__']
    if len(children)!=2:raise RuntimeError('Expected exactly two guarded DataLoader children')
    return dict(no_model_construction=True,no_training=True,native_filter_then_match_verified=True,
                supported_training_options=True,guarded_spawn_children=children)


def worker(role,arm,reference):
    stage=stage_path(role,arm,reference)
    events=BOOT_EVENTS if BOOT_EVENTS is not None else guard(Path.cwd());started=time.perf_counter()
    try:
        import torch,ultralytics
        from ultralytics.utils import ONLINE,SETTINGS
        if torch.__version__!='2.14.1+cu130' or ultralytics.__version__!='8.4.165' or ONLINE or SETTINGS['sync']:
            raise RuntimeError('Offline environment/version mismatch')
        result=preflight(stage) if role=='preflight' else (train(arm,stage) if role=='train' else evaluate(arm,reference,stage))
        if events or list(Path.cwd().glob('guard_events*.jsonl')):
            raise RuntimeError('Guard events detected, including caught library/child exceptions')
        result.update(state='completed',finished_at=stamp(),seconds=time.perf_counter()-started,
                      runtime={'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,
                               'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},no_3lc_import=not any(k.split('.')[0]=='tlc' for k in sys.modules))
        save(stage/'result.json',result)
    except BaseException:
        save(stage/'failure.json',dict(failed_at=stamp(),seconds=time.perf_counter()-started,traceback=traceback.format_exc()))
        raise


def stage_timeout(role, deadline, current):
    remaining = deadline - current
    if remaining <= 0:
        raise TimeoutError('A/B total time budget exhausted')
    return min(5400 if role == 'train' else 3600, remaining)


def launch(role,arm,reference=None,deadline=None):
    timeout = stage_timeout(role, deadline, time.perf_counter())
    stage=stage_path(role,arm,reference)
    stage.mkdir(parents=True,exist_ok=False)
    work=OUT/'runtime'/f'{arm}-{role}-{reference or "train"}';work.mkdir()
    helper=support();env=helper.environment(work)
    env['CLEANING_AB_GUARD_WORK']=str(work)
    args=[str(PYTHON),'-I','-B','-u',str(Path(__file__).resolve()),'_worker','--role',role]
    if arm:args+=['--arm',arm]
    if reference:args+=['--reference',reference]
    save(stage/'launch.json',dict(started_at=stamp(),role=role,arm=arm,reference=reference))
    started=time.perf_counter();job=None;code=None;timed_out=False
    with (stage/'console.log').open('x',encoding='utf-8') as f:
        process=subprocess.Popen(args,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            job=helper.WindowsJob(process)
            code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
        finally:
            if job:job.close()
            if process.poll() is None:process.kill();process.wait()
    save(stage/'exit.json',dict(exit_code=process.returncode,timeout=timed_out,timeout_seconds=timeout,seconds=time.perf_counter()-started,finished_at=stamp()))
    if timed_out:raise TimeoutError('Stage timed out; preserved '+str(stage))
    if code!=0 or not (stage/'result.json').is_file():raise RuntimeError('Stage failed; preserve '+str(stage))
    print('COMPLETED',arm,role,reference or '',flush=True)
    return load(stage/'result.json')


def run():
    if not (OUT/'plan.json').is_file() or (OUT/'run_started.json').exists():
        raise RuntimeError('Prepare first; never repeat an existing experiment')
    if load(OUT/'preflight/result.json').get('state')!='completed':
        raise RuntimeError('Offline preflight must complete before any model construction')
    verify_rows(load(OUT/'train_manifest.json')+load(OUT/'val_manifest.json'))
    deadline=time.perf_counter()+load(OUT/'plan.json')['total_timeout_seconds']
    save(OUT/'run_started.json',dict(started_at=stamp(),training_runs=2))
    for name,h in load(OUT/'execution_seal.json')['source_hashes'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Frozen executor changed')
    try:
        for arm in ['A','B']:
            launch('train',arm,deadline=deadline)
        if load(OUT/'A/train/result.json')['initial_state_sha256']!=load(OUT/'B/train/result.json')['initial_state_sha256']:
            raise RuntimeError('Initial model state not paired')
        for arm in ['A','B']:
            for reference in ['A','B','val966']:launch('evaluate',arm,reference,deadline=deadline)
        for name,h in load(OUT/'protected.json').items():
            if sha(ROOT/name)!=h:raise RuntimeError('Protected original drift')
        verify_rows(load(OUT/'train_manifest.json')+load(OUT/'val_manifest.json'))
        save(OUT/'completed.json',dict(finished_at=stamp(),training_runs=2,evaluations=6,protected_originals_unchanged=True))
        print('PAIRED EXPERIMENT COMPLETE: two trainings and six uniform evaluations',flush=True)
    except BaseException:
        save(OUT/'run_failure.json',dict(at=stamp(),traceback=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','preflight','seal','run','_worker'])
    parser.add_argument('--arm',choices=['A','B']);parser.add_argument('--role',choices=['train','evaluate','preflight'])
    parser.add_argument('--reference',choices=['A','B','val966']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='preflight':launch('preflight',None,deadline=time.perf_counter()+180)
    elif args.action=='seal':
        hashes=load(OUT/'plan.json')['source_hashes']
        changed=['tools/train_cleaning_ab.py','tests/test_train_cleaning_ab.py']
        for name,h in hashes.items():
            if name not in changed and sha(ROOT/name)!=h:raise RuntimeError('Unexpected frozen artifact drift: '+name)
        for name in changed+['tools/summarize_cleaning_ab.py','tests/test_summarize_cleaning_ab.py','reports/train-cleaning-ab-ss04-20261029/preflight/result.json']:
            hashes[name]=sha(ROOT/name)
        save(OUT/'execution_seal.json',dict(at=stamp(),reason='Pre-training review corrections; original plan preserved',source_hashes=hashes))
    elif args.action=='run':run()
    else:worker(args.role,args.arm,args.reference)


# Windows spawn re-executes __mp_main__ before unpickling Torch/DataLoader state.
# Install the same guard in every child before any third-party imports.
BOOT_EVENTS=None
if os.environ.get('CLEANING_AB_GUARD_WORK'):
    work=Path(os.environ['CLEANING_AB_GUARD_WORK']).resolve()
    if not work.is_relative_to(OUT/'runtime') or Path.cwd().resolve()!=work:
        raise RuntimeError('Invalid A/B worker bootstrap location')
    BOOT_EVENTS=guard(work)
    save(work/f'guard_ready_{os.getpid()}.json',dict(pid=os.getpid(),module=__name__,at=stamp()))
if __name__=='__main__':main()
