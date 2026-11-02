"""One requested D060 test prediction and strict sample-aligned CSV; never uploads."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types

ROOT=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
KIT=ROOT/'multi-vehicle_kaggle_competition_starter_kit_files'
OUT=ROOT/'reports/submission-d060-20261102'
TRAIN=ROOT/'reports/train-cleaning-expansion-resume-r1-20261102'
PYTHON=Path(r'E:\anaconda\envs\dl-3lc\python.exe')
HASH='b36f72c507a13cc134092f3589d181c25b2014ae8f6845113b662ce654ed1eac'
FIELDS=['id','image_id','prediction_string']
OPTIONS=dict(conf=.25,iou=.7,imgsz=640,batch=1,max_det=300,device=0,augment=False,
    agnostic_nms=False,half=False,save=False,save_txt=False,save_conf=False,verbose=False)


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def require(ok,message):
    if not ok:raise ValueError(message)


def encode(boxes):
    parts=[]
    for row in sorted(boxes,key=lambda r:-r[1]):
        require(len(row)==6,'Expected six prediction values')
        cls,confidence,*coordinates=row
        require(math.isfinite(float(cls)) and int(cls)==cls and 0<=cls<=3,'Invalid class')
        require(all(math.isfinite(float(v)) and 0<=v<=1 for v in [confidence]+coordinates),'Nonfinite/out-of-range prediction')
        parts.extend([str(int(cls))]+[f'{float(v):.6f}' for v in [confidence]+coordinates])
    return ' '.join(parts) if parts else 'no box'


def read_sample(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f);require(reader.fieldnames==FIELDS,'Sample headers differ');rows=list(reader)
    require(bool(rows) and len({r['id'] for r in rows})==len(rows) and len({r['image_id'] for r in rows})==len(rows),'Duplicate/empty sample')
    for r in rows:
        require(set(r)==set(FIELDS) and all(v is not None for v in r.values()),'Malformed sample row')
        require(str(int(r['id']))==r['id'],'Sample id must be canonical integer')
    return rows


def match_images(rows,directory):
    directory=Path(directory);by={}
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp'):
            require(p.stem not in by,'Ambiguous duplicate image stem');by[p.stem]=p
    stems=[r['image_id'] for r in rows]
    require(len(stems)==len(set(stems)) and all(s and Path(s).name==s and '/' not in s and '\\' not in s for s in stems),'Unsafe/duplicate image_id')
    require(set(stems)==set(by),'Missing/extra test image relative to sample')
    return [by[s] for s in stems]


def verify_csv(path,sample):
    with Path(path).open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);require(reader.fieldnames==FIELDS,'Submission headers differ');rows=list(reader)
    require(len(rows)==len(sample),'Submission row count differs')
    boxes=0;empty=0;classes={i:0 for i in range(4)}
    for r,s in zip(rows,sample):
        require(set(r)==set(FIELDS) and r['id']==s['id'] and r['image_id']==s['image_id'],'Submission identity/order differs')
        text=r['prediction_string'];require(isinstance(text,str) and '\n' not in text and '\r' not in text,'Multiline/empty prediction')
        if text=='no box':empty+=1;continue
        tokens=text.split();require(bool(tokens) and len(tokens)%6==0,'Incomplete six-tuple')
        for start in range(0,len(tokens),6):
            group=tokens[start:start+6];cls=int(group[0]);require(str(cls)==group[0] and 0<=cls<=3,'Invalid encoded class')
            values=[float(x) for x in group[1:]];require(all(math.isfinite(x) and 0<=x<=1 for x in values),'Invalid encoded float')
            boxes+=1;classes[cls]+=1
    return dict(verified=True,rows=len(rows),boxes=boxes,no_box_rows=empty,classes=classes,sha256=sha(path),bytes=Path(path).stat().st_size)


def support():
    p=ROOT/'tools/train_cleaning_expansion.py';m=types.ModuleType('submission_support');m.__file__=str(p)
    exec(compile(p.read_text(encoding='utf-8').split('\nBOOT_EVENTS=[]\n')[0],str(p),'exec'),m.__dict__)
    return m


def guarded(c,work):
    c.OUT=OUT;c.SELF=SELF;normal=c.allowed_file
    allowed_exact={KIT/'sample_submission.csv',KIT/'config.yaml',KIT/'dataset.yaml',
        ROOT/'tools/train_cleaning_expansion.py',TRAIN/'D060/train/result.json',TRAIN/'D060/yolov8n/weights/best.pt'}
    def allowed(path,writing):
        p=Path(path).resolve()
        if p in allowed_exact or p.is_relative_to((KIT/'data/test/images').resolve()):return not writing
        return normal(path,writing)
    c.allowed_file=allowed
    return c.guard(work)


def prepare():
    require(not OUT.exists(),'Existing submission attempt preserved')
    import yaml
    cfg=yaml.safe_load((KIT/'config.yaml').read_text(encoding='utf-8'))
    expected=dict(conf=.25,iou=.7,max_det=300,batch=1,pipeline='memory')
    require(all(cfg['predict'][k]==v for k,v in expected.items()),'Starter prediction options changed')
    require((cfg['predict']['imgsz'] or cfg['training']['image_size'])==640 and
        (cfg['training']['device'] if cfg['predict']['device'] is None else cfg['predict']['device'])==0,'Image/device config changed')
    require(yaml.safe_load((KIT/'dataset.yaml').read_text(encoding='utf-8'))['names']=={0:'truck',1:'car',2:'van',3:'bus'},'Class map differs')
    train=load(TRAIN/'D060/train/result.json');weight=Path(train['best'])
    require(train['state']=='completed' and train['best_sha256']==HASH and sha(weight)==HASH,'D060 checkpoint differs')
    rows=read_sample(KIT/'sample_submission.csv');paths=match_images(rows,KIT/'data/test/images')
    OUT.mkdir();work=OUT/'runtime';work.mkdir()
    pins={str(p):sha(p) for p in [SELF,ROOT/'tests/test_submission_d060.py',KIT/'sample_submission.csv',KIT/'config.yaml',KIT/'dataset.yaml',
        TRAIN/'D060/train/result.json',weight]+paths}
    save(OUT/'inputs.json',dict(sample_rows=len(rows),images=len(paths),weight=str(weight),weight_sha256=HASH,options=OPTIONS,pins=pins))
    save(OUT/'authorization.json',dict(user_requested_D060_test_csv=True,manual_upload_by_user=True,
        scope='One test pass using existing starter options; no training/TTA/ensemble/search or upload.',
        output=str(OUT/'submission.csv'),workflow_compliance_not_certified=True))


def verify_inputs():
    inputs=load(OUT/'inputs.json')
    for path,digest in inputs['pins'].items():require(sha(path)==digest,'Protected input changed: '+path)
    return inputs


def worker():
    start=time.perf_counter()
    try:
        import torch,ultralytics
        from ultralytics import YOLO
        from ultralytics.utils import ONLINE,SETTINGS
        import ultralytics.utils.downloads as downloads
        require(torch.__version__=='2.14.1+cu130' and ultralytics.__version__=='8.4.165' and not ONLINE and not SETTINGS['sync'],'Offline runtime mismatch')
        inputs=load(OUT/'inputs.json');weight=Path(inputs['weight'])
        def local(file,*args,**kwargs):
            p=Path(file).resolve();require(p==weight.resolve() and sha(p)==HASH,'Missing/unapproved weight; download forbidden');return str(p)
        downloads.attempt_download_asset=local
        rows=read_sample(KIT/'sample_submission.csv');paths=match_images(rows,KIT/'data/test/images')
        require(sha(weight)==HASH,'Weight drift')
        model=YOLO(str(weight),task='detect');require(model.names=={0:'truck',1:'car',2:'van',3:'bus'},'Model class map differs')
        output=OUT/'submission.csv'
        with output.open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
            for i,(row,path) in enumerate(zip(rows,paths),1):
                results=model.predict(source=[str(path)],project=str(OUT),name='predict',exist_ok=False,**OPTIONS)
                require(len(results)==1 and Path(results[0].path).resolve()==path.resolve(),'Prediction identity/coverage mismatch')
                boxes=results[0].boxes
                encoded=encode([] if boxes is None else [(cls,conf,*xywh) for cls,conf,xywh in zip(
                    boxes.cls.cpu().tolist(),boxes.conf.cpu().tolist(),boxes.xywhn.cpu().tolist())])
                writer.writerow(dict(id=row['id'],image_id=row['image_id'],prediction_string=encoded))
                if i%100==0 or i==len(rows):print('PREDICTED',i,'/',len(rows),flush=True)
        require(not BOOT_EVENTS and not list((OUT/'runtime').glob('guard_events*.jsonl')),'Guard rejection occurred')
        result=verify_csv(output,rows);result.update(seconds=time.perf_counter()-start,weight_sha256=sha(weight),options=OPTIONS,
            no_3lc_import=not any(k.split('.')[0] in ['tlc','tlc_ultralytics','tlcconfig'] for k in sys.modules),state='completed')
        save(OUT/'result.json',result)
    except BaseException:
        save(OUT/'failure.json',dict(traceback=traceback.format_exc(),seconds=time.perf_counter()-start));raise


def run():
    require(not (OUT/'started.json').exists(),'Cannot repeat test inference')
    inputs=verify_inputs();save(OUT/'started.json',dict(started_unix=time.time(),images=inputs['images']))
    helper=support().ab().support();env=helper.environment(OUT/'runtime');env['D060_SUBMISSION_WORK']=str(OUT/'runtime')
    job=None;timed_out=False;start=time.perf_counter()
    with (OUT/'console.log').open('x',encoding='utf-8') as f:
        process=subprocess.Popen([str(PYTHON),'-I','-B','-u',str(SELF),'_worker'],cwd=OUT/'runtime',env=env,stdout=f,stderr=subprocess.STDOUT)
        try:
            job=helper.WindowsJob(process);process.wait(timeout=1800)
        except subprocess.TimeoutExpired:timed_out=True
        finally:
            if job:job.close()
            if process.poll() is None:process.kill();process.wait()
    save(OUT/'exit.json',dict(code=process.returncode,timeout=timed_out,seconds=time.perf_counter()-start))
    require(not timed_out and process.returncode==0,'Prediction worker failed; preserve evidence')
    verify_inputs();result=verify_csv(OUT/'submission.csv',read_sample(KIT/'sample_submission.csv'))
    require(result['sha256']==load(OUT/'result.json')['sha256'],'Submission result hash differs')
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','verify','_worker']);args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run()
    elif args.action=='_worker':worker()
    else:
        verify_inputs();print(json.dumps(verify_csv(OUT/'submission.csv',read_sample(KIT/'sample_submission.csv')),indent=2))


BOOT_EVENTS=[]
if os.environ.get('D060_SUBMISSION_WORK'):
    work=Path(os.environ['D060_SUBMISSION_WORK']).resolve();require(work==OUT/'runtime' and work==Path.cwd().resolve(),'Invalid submission bootstrap')
    BOOT_EVENTS=guarded(support(),work)
if __name__=='__main__':main()
