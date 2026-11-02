"""Bounded evaluation-only recovery; immutable trained sources, no training entrypoint."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/train-cleaning-ab-ss04-20261029'
OUT=ROOT/'reports/eval-cleaning-ab-recovery-20261029'
PYTHON=Path(r'E:\anaconda\envs\dl-3lc\python.exe')
SELF=Path(__file__).resolve()
AUTH=ROOT/'logs/eval-cleaning-ab-recovery-20261029.md'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def stamp():return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def module(name,path):
    import types
    m=types.ModuleType(name);m.__file__=str(Path(path).resolve())
    exec(compile(Path(path).read_bytes(),m.__file__,'exec'),m.__dict__)
    return m


def original():return module('frozen_ab_support',ROOT/'tools/train_cleaning_ab.py')


def weight_path(arm):
    r=load(SOURCE/arm/'train/result.json')
    if r['state']!='completed' or r['epochs']!=28:raise RuntimeError('Completed 28-epoch model required')
    p=Path(r['best']).resolve()
    if not p.is_relative_to(SOURCE.resolve()/arm) or p.suffix!='.pt':raise RuntimeError('Invalid recorded best path')
    if sha(p)!=r['best_sha256']:raise RuntimeError('Best checkpoint hash drift')
    return p


def evaluation_options():
    return dict(split='val',imgsz=640,batch=32,device=0,workers=2,conf=.001,iou=.7,
                max_det=300,quantize=None,rect=False,augment=False,plots=False,
                save_json=False,save_txt=False,verbose=False,exist_ok=False,
                agnostic_nms=False,classes=None,single_cls=False,compile=False)


def local_assets(approved):
    def resolve(file,*args,**kwargs):
        p=Path(file).resolve();expected=approved.get(str(p))
        if expected is None or not p.is_file() or sha(p)!=expected:
            raise RuntimeError('Only verified existing paired best weights may be loaded: '+str(p))
        return str(p)
    return resolve


def local_runtime():
    import ultralytics.data.utils as data_utils
    import ultralytics.utils.downloads as downloads
    data_utils.check_font=lambda *a,**k:None
    approved={str(weight_path(a)):load(SOURCE/a/'train/result.json')['best_sha256'] for a in ['A','B']}
    downloads.attempt_download_asset=local_assets(approved)


def evaluate(arm,reference,stage):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    local_runtime();support=original()
    rows=load(OUT/'val_manifest.json') if reference=='val966' else [r for r in load(OUT/'train_manifest.json') if r['arm']==reference]
    capture={'expected_rows':rows,'expected_boxes':11487 if reference=='val966' else {'A':72273,'B':72665}[reference]}
    validator=support.scored_validator_factory(DetectionValidator,capture)
    weight=weight_path(arm);model=YOLO(str(weight),task='detect')
    metrics=model.val(validator=validator,data=str(OUT/('eval_val966.yaml' if reference=='val966' else 'eval_train_'+reference+'.yaml')),
                      project=str(stage),name='native',**evaluation_options())
    v=capture['validator']
    if v.seen!=len(rows) or v.args.quantize is not None:raise RuntimeError('Coverage/FP32 mismatch')
    per=[dict(cls=int(cls),name=support.NAMES[int(cls)],precision_best_f1=float(metrics.box.p[i]),
              recall_best_f1=float(metrics.box.r[i]),AP50=float(metrics.box.all_ap[i,0]),AP50_95=float(metrics.box.all_ap[i].mean()))
         for i,cls in enumerate(metrics.box.ap_class_index)]
    return dict(arm=arm,reference=reference,images=v.seen,weight_sha256=sha(weight),weight=str(weight),
                native={k:float(value) for k,value in metrics.results_dict.items()},classes=per,
                fixed=v.common_fixed,precision='FP32 quantize=None',effective_options={k:getattr(v.args,k) for k in evaluation_options()},
                evaluator='Ultralytics8.4.165 native matching/AP; fixed confidence rematched separately')


def preflight():
    import torch
    from ultralytics import YOLO
    local_runtime();weights=[]
    for arm in ['A','B']:
        model=YOLO(str(weight_path(arm)),task='detect')
        if model.names!=original().NAMES:raise RuntimeError('Checkpoint classes differ')
        weights.append(dict(arm=arm,weight=str(weight_path(arm)),parameters=sum(p.numel() for p in model.model.parameters())))
        del model
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(4)),num_workers=2)
    if sum(int(x[0].item()) for x in loader)!=6:raise RuntimeError('Child loader failed')
    children=[load(p) for p in Path.cwd().glob('guard_ready_*.json') if load(p)['module']=='__mp_main__']
    if len(children)!=2:raise RuntimeError('Child guard missing')
    return dict(no_forward=True,no_training=True,weights=weights,guarded_children=children)


def allowed_file(path,writing):
    p=Path(path).resolve()
    roots=[OUT.resolve()] if writing else [OUT.resolve(),SOURCE.resolve(),PYTHON.parent.resolve()]
    exact=[SELF,ROOT/'tools/train_cleaning_ab.py',ROOT/'tools/oof_diagnostic.py']
    return any(p==r or p.is_relative_to(r) for r in roots) or (not writing and p in exact)


def check_mutation(event,args):
    counts={'os.remove':1,'os.rmdir':1,'os.mkdir':1,'os.rename':2,'os.link':2,'os.symlink':2,
            'os.chmod':1,'os.utime':1,'os.truncate':1}
    if event not in counts:return
    for value in args[:counts[event]]:
        if not isinstance(value,(str,bytes,os.PathLike)) or not allowed_file(os.fsdecode(value),True):
            raise PermissionError('Mutation outside recovery outputs: '+event)


def guard(work):
    import importlib.abc
    events=[]
    def deny(event,**details):
        frame=sys._getframe(1);stack=[]
        while frame is not None:
            stack.append({'file':frame.f_code.co_filename,'line':frame.f_lineno,'function':frame.f_code.co_name});frame=frame.f_back
        record=dict(at=stamp(),event=event,stack=stack,**details);events.append(record)
        with (work/f'guard_events_{os.getpid()}.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        raise PermissionError('Evaluation-only recovery guard: '+event)
    def hook(event,args):
        try:check_mutation(event,args)
        except PermissionError:deny('mutation',operation=event,arguments=repr(args)[:1000])
        if event in ['socket.connect','socket.connect_ex','socket.bind','socket.getaddrinfo','socket.sendto','socket.sendmsg','subprocess.Popen','os.system','os.posix_spawn']:
            deny(event,arguments=repr(args)[:1000])
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


def launch(role,arm=None,reference=None,deadline=None):
    remaining=deadline-time.perf_counter()
    if remaining<=0:raise TimeoutError('Recovery total deadline reached')
    timeout=min(3600 if role=='evaluate' else 180,remaining)
    stage=OUT/arm/('eval_'+reference) if role=='evaluate' else OUT/role
    stage.mkdir(parents=True,exist_ok=False)
    work=OUT/'runtime'/f'{role}-{arm or "none"}-{reference or "none"}';work.mkdir(parents=True)
    env=original().support().environment(work);env['CLEANING_AB_RECOVERY_WORK']=str(work)
    args=[str(PYTHON),'-I','-B','-u',str(SELF),'_worker','--role',role]
    if arm:args+=['--arm',arm,'--reference',reference]
    save(stage/'launch.json',dict(at=stamp(),role=role,arm=arm,reference=reference,timeout_seconds=timeout))
    job=None;timed_out=False;start=time.perf_counter()
    with (stage/'console.log').open('x',encoding='utf-8') as f:
        process=subprocess.Popen(args,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            job=original().support().WindowsJob(process)
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:timed_out=True
        finally:
            if job:job.close()
            if process.poll() is None:process.kill();process.wait()
    save(stage/'exit.json',dict(code=process.returncode,timeout=timed_out,seconds=time.perf_counter()-start,at=stamp()))
    if timed_out or process.returncode or not (stage/'result.json').is_file():raise RuntimeError('Recovery stage failed: '+str(stage))
    print('COMPLETED',role,arm or '',reference or '',flush=True)
    return load(stage/'result.json')


def worker(role,arm,reference):
    stage=OUT/arm/('eval_'+reference) if role=='evaluate' else OUT/role
    start=time.perf_counter()
    try:
        import torch,ultralytics
        from ultralytics.utils import ONLINE,SETTINGS
        if torch.__version__!='2.14.1+cu130' or ultralytics.__version__!='8.4.165' or ONLINE or SETTINGS['sync']:
            raise RuntimeError('Offline runtime mismatch')
        if role=='diagnose':
            from ultralytics.utils.downloads import attempt_download_asset
            missing=SOURCE/'A/model/weights/best.pt'
            if missing.exists():raise RuntimeError('Historical missing path unexpectedly exists')
            # No model load, only reproduce missing-local-asset fallback with every socket blocked.
            attempt_download_asset(str(missing))
            result=dict(no_model_load=True,blocked_events=BOOT_EVENTS)
        elif role=='preflight':result=preflight()
        else:result=evaluate(arm,reference,stage)
        if role!='diagnose' and list(Path.cwd().glob('guard_events*.jsonl')):raise RuntimeError('Guard rejection, including caught exceptions')
        result.update(state='completed',seconds=time.perf_counter()-start,at=stamp(),no_3lc_import=not any(k.split('.')[0] in ['tlc','tlc_ultralytics','tlcconfig'] for k in sys.modules),
                      runtime=dict(python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
        save(stage/'result.json',result)
    except BaseException:
        save(stage/'failure.json',dict(at=stamp(),traceback=traceback.format_exc()));raise


def verify_sources():
    for name,h in load(SOURCE/'execution_seal.json')['source_hashes'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Frozen source drift: '+name)
    for name,h in load(SOURCE/'protected.json').items():
        if sha(ROOT/name)!=h:raise RuntimeError('Protected source drift: '+name)
    original().verify_rows(load(SOURCE/'train_manifest.json')+load(SOURCE/'val_manifest.json'))
    for arm in ['A','B']:
        weight_path(arm)
        r=load(SOURCE/arm/'train/result.json')
        if sha(r['last'])!=r['last_sha256']:raise RuntimeError('Last checkpoint drift')


def prepare():
    if (OUT/'prepared.json').exists():raise FileExistsError('Already prepared')
    verify_sources()
    import yaml
    manifests={}
    for name in ['train_manifest','val_manifest']:
        rows=load(SOURCE/(name+'.json'));copied=[]
        for row in rows:
            new=row.copy()
            for kind in ['image','label']:
                src=Path(row[kind]);dest=OUT/src.relative_to(SOURCE)
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
                if sha(dest)!=row[kind+'_sha256'] or dest.stat().st_ino==src.stat().st_ino:raise RuntimeError('Independent data copy mismatch')
                new[kind]=str(dest)
            copied.append(new)
        save(OUT/(name+'.json'),copied);manifests[name]=copied
    for ref in ['A','B','val966']:
        images=OUT/('data/val/images/val' if ref=='val966' else 'data/'+ref+'/images/train')
        with (OUT/('eval_val966.yaml' if ref=='val966' else 'eval_train_'+ref+'.yaml')).open('x',encoding='utf-8') as f:
            yaml.safe_dump(dict(path=str(OUT),train=str(images),val=str(images),nc=4,names=original().NAMES),f,sort_keys=False)
    for arm in ['A','B']:save(OUT/arm/'train/result.json',load(SOURCE/arm/'train/result.json'))
    save(OUT/'prepared.json',dict(at=stamp(),training_runs=0,evaluations=6,source=str(SOURCE),authorization_sha256=sha(AUTH),
                                 original_initial_state_sha256=load(SOURCE/'A/train/result.json')['initial_state_sha256']))


def seal():
    files=[SELF,AUTH,ROOT/'tests/test_eval_cleaning_ab_recovery.py',ROOT/'tools/summarize_cleaning_ab.py',
           ROOT/'tools/train_cleaning_ab.py',ROOT/'tools/oof_diagnostic.py',OUT/'prepared.json',OUT/'preflight/result.json',
           OUT/'train_manifest.json',OUT/'val_manifest.json']+[OUT/n for n in ['eval_train_A.yaml','eval_train_B.yaml','eval_val966.yaml']]
    save(OUT/'seal.json',dict(at=stamp(),hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}))


def run_stages():
    if (OUT/'run_started.json').exists():raise FileExistsError('Recovery cannot be repeated')
    for name,h in load(OUT/'seal.json')['hashes'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Recovery seal drift: '+name)
    verify_sources();original().verify_rows(load(OUT/'train_manifest.json')+load(OUT/'val_manifest.json'))
    deadline=time.perf_counter()+10800
    save(OUT/'run_started.json',dict(at=stamp(),training_runs=0,evaluations=6,total_timeout_seconds=10800))
    try:
        for arm in ['A','B']:
            for reference in ['A','B','val966']:launch('evaluate',arm,reference,deadline)
        verify_sources();original().verify_rows(load(OUT/'train_manifest.json')+load(OUT/'val_manifest.json'))
        save(OUT/'completed.json',dict(at=stamp(),training_runs=0,evaluations=6,protected_originals_unchanged=True))
        summary=module('recovery_summary',ROOT/'tools/summarize_cleaning_ab.py');summary.OUT=OUT;summary.main()
        required=['RESULT.md','metrics.csv','comparison.json']
        for name in required:
            if not (OUT/name).stat().st_size:raise RuntimeError('Missing report artifact: '+name)
        save(OUT/'success.json',dict(at=stamp(),training_runs=0,evaluations=6,reports={n:sha(OUT/n) for n in required},
                                   note='completed.json marks evaluation/integrity prerequisite; this marker confirms reporting too'))
    except BaseException:
        save(OUT/'run_failure.json',dict(at=stamp(),traceback=traceback.format_exc()));raise


def run():
    # Outer supervisor bounds preparation verification, six workers, final hashes and reporting.
    if (OUT/'supervision_started.json').exists():raise FileExistsError('One recovery only')
    save(OUT/'supervision_started.json',dict(at=stamp(),timeout_seconds=10800))
    code=None;timed_out=False;job=None;process=None;error=None;started=time.perf_counter()
    try:
        with (OUT/'stages.log').open('x',encoding='utf-8') as f:
            process=subprocess.Popen([str(PYTHON),'-I','-B','-u',str(SELF),'_run'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            job=original().support().WindowsJob(process);code=process.wait(timeout=10800)
    except subprocess.TimeoutExpired:timed_out=True
    except BaseException:error=traceback.format_exc()
    finally:
        if job:job.close()
        if process is not None and process.poll() is None:process.kill();process.wait()
    save(OUT/'supervision_exit.json',dict(at=stamp(),code=process.returncode if process else None,timeout=timed_out,error=error,seconds=time.perf_counter()-started))
    if error or timed_out or code!=0 or not (OUT/'success.json').exists():
        # Independent read-only audit, with its own bounded cleanup budget, never retries evaluation.
        with (OUT/'failure-integrity.log').open('x',encoding='utf-8') as f:
            check=subprocess.Popen([str(PYTHON),'-I','-B',str(SELF),'_failure_verify'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            try:check.wait(timeout=300)
            except subprocess.TimeoutExpired:check.kill();check.wait()
        if not (OUT/'failure-integrity.json').exists():save(OUT/'failure-integrity.json',dict(at=stamp(),verified=False,check_exit=check.returncode))
        raise RuntimeError('Recovery stopped; retained failure and integrity evidence')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['diagnose','prepare','preflight','seal','run','_worker','_run','_failure_verify'])
    p.add_argument('--role',choices=['diagnose','preflight','evaluate'])
    p.add_argument('--arm',choices=['A','B']);p.add_argument('--reference',choices=['A','B','val966']);args=p.parse_args()
    if args.action=='_worker':worker(args.role,args.arm,args.reference)
    elif args.action=='diagnose':
        OUT.mkdir(exist_ok=False)
        launch('diagnose',deadline=time.perf_counter()+180)
    elif args.action=='prepare':prepare()
    elif args.action=='preflight':launch('preflight',deadline=time.perf_counter()+180)
    elif args.action=='seal':seal()
    elif args.action=='_run':run_stages()
    elif args.action=='_failure_verify':
        try:
            verify_sources();original().verify_rows(load(OUT/'train_manifest.json')+load(OUT/'val_manifest.json'))
            save(OUT/'failure-integrity.json',dict(at=stamp(),verified=True,original_sources_and_eval_copies_unchanged=True))
        except BaseException:
            save(OUT/'failure-integrity.json',dict(at=stamp(),verified=False,traceback=traceback.format_exc()));raise
    else:run()


BOOT_EVENTS=[]
if os.environ.get('CLEANING_AB_RECOVERY_WORK'):
    work=Path(os.environ['CLEANING_AB_RECOVERY_WORK']).resolve()
    if not work.is_relative_to(OUT/'runtime') or work!=Path.cwd().resolve():raise RuntimeError('Invalid recovery work directory')
    BOOT_EVENTS=guard(work)
    save(work/f'guard_ready_{os.getpid()}.json',dict(at=stamp(),pid=os.getpid(),module=__name__))
if __name__=='__main__':main()
