#!/usr/bin/env python3
"""Authorized fixed-five-group train-only 2x heat-region pilot. No arbitrary inputs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bus-crop-expansion-retry1-20261025'
STATIC=ROOT/'reports/bus-static-review'
PREFLIGHT=ROOT/'reports/bus-crop-expansion-preflight-20261025'
GROUPS=['MVI_40181','MVI_40851','MVI_39401','MVI_40761','MVI_63521']
GROUP_SIZES=[107,72,80,114,119]
EXPECTED_ROIS={
 'MVI_40181':[[0,0,320,320],[320,0,640,320],[0,320,320,640],[320,320,640,640]],
 'MVI_40851':[[1,0,321,320],[319,0,639,320]],
 'MVI_39401':[[152,97,472,417],[320,97,640,417],[152,320,472,640],[320,320,640,640]],
 'MVI_40761':[[0,42,320,362],[320,42,640,362]],
 'MVI_63521':[[320,1,640,321]]}
N_INPUTS=1239

def load_module(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

P=load_module('expansion_pilot_helpers','tools/bus_crop_pilot.py')
RT=load_module('expansion_runtime_helpers','tools/bus_crop_runtime.py')
PARAMS=P.PARAMS
WEIGHT=P.WEIGHT
WEIGHT_HASH=P.WEIGHT_HASH
PYTHON=RT.PYTHON
SCRATCH=RT.SCRATCH
sha=P.sha
read=P.read
write=P.write
now=P.now


def coverage_rois(bounds):
    if len(bounds)!=4 or any(not isinstance(v,int) for v in bounds):raise ValueError('Integer bounds required')
    if not (0<=bounds[0]<bounds[2]<=640 and 0<=bounds[1]<bounds[3]<=640):raise ValueError('Invalid heat bounds')
    axes=[]
    for lo,hi in ((bounds[0],bounds[2]),(bounds[1],bounds[3])):
        if hi-lo<=320:positions=[min(320,max(0,math.floor((lo+hi)/2-160)))]
        else:positions=sorted({min(320,max(0,lo)),min(320,max(0,hi-320))})
        axes.append(positions)
    return [[x,y,x+320,y+320] for y in axes[1] for x in axes[0]]


def union_mask(rois):
    mask=np.zeros((640,640),dtype=bool)
    for roi in rois:
        P.validate_roi(roi)
        if roi[2]-roi[0]!=320 or roi[3]-roi[1]!=320:raise ValueError('Only approved 320 crops')
        mask[roi[1]:roi[3],roi[0]:roi[2]]=True
    return mask


def freeze_sources():
    old=read(STATIC/'verification.json');inventory=STATIC/'scene_manifest.json'
    if sha(inventory)!=old['artifact_sha256']['scene_manifest.json']:raise ValueError('Inventory changed')
    inv=read(inventory);pre=read(PREFLIGHT/'preflight.json');groups=[];samples=[]
    if pre['scope']['groups']!=GROUPS:raise ValueError('Preflight scope changed')
    for sid,size in zip(GROUPS,GROUP_SIZES):
        scene=next(s for s in inv['scenes'] if s['id']==sid)
        path=STATIC/scene['maps']['counts']
        if sha(STATIC/scene['maps']['heat'])!=old['artifact_sha256'][scene['maps']['heat']]:raise ValueError('Heat display source changed')
        if sha(path)!=old['artifact_sha256'][scene['maps']['counts']] or sha(path)!=pre['source_sha256'][scene['maps']['counts']]:raise ValueError('Heat source changed')
        with Image.open(path) as im:
            if im.size!=(640,640):raise ValueError('Heat dimensions changed')
            a=np.asarray(im)
        counts=a[:,:,0].astype(np.uint16)+256*a[:,:,1].astype(np.uint16)
        y,x=np.nonzero(counts);bounds=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
        prior=next(g for g in pre['groups'] if g['scene']==sid)
        if bounds!=prior['heat_bounds_xyxy_half_open']:raise ValueError('Heat bounds changed')
        rois=coverage_rois(bounds)
        if rois!=EXPECTED_ROIS[sid] or np.any((counts>0)&~union_mask(rois)):raise ValueError('Frozen crop coverage changed')
        selected=sorted([inv['images'][i] for i in scene['train_ids']],key=lambda s:(s['frame'],s['image_path']))
        if len(selected)!=size or scene['val_bus_images']<1:raise ValueError('Group scope changed')
        samples.extend(selected)
        groups.append(dict(id=sid,images=size,heat_bounds=bounds,rois=rois,heat_count_path=scene['maps']['counts'],
                           heat_count_sha256=sha(path),heat_url=scene['maps']['heat'],heat_sha256=sha(STATIC/scene['maps']['heat']),
                           heat_covered=True,original_val_bus_images=scene['val_bus_images'],original_val_bus_boxes=scene['val_bus_boxes'],
                           train_without_bus_label=sum(s['bus_boxes']==0 for s in selected),
                           roi_overlap_pixels=int(sum(320*320 for r in rois)-union_mask(rois).sum())))
    views=[]
    for sample in samples:
        for n,roi in enumerate(EXPECTED_ROIS[sample['scene']]):
            tile=f'T{n+1}'
            views.append(dict(id=sample['id']+':'+tile,sample_id=sample['id'],scene=sample['scene'],frame=sample['frame'],
                              tile=tile,roi=roi,scale=2,input_file=f'inputs/{sample["id"]}_{tile}.png'))
    validate_fixed(samples,views)
    return inv,groups,samples,views


def validate_fixed(samples,views):
    if len(samples)!=492 or len({s['id'] for s in samples})!=492:raise ValueError('Fixed492 unique train images required')
    if [sum(s['scene']==sid for s in samples) for sid in GROUPS]!=GROUP_SIZES:raise ValueError('Only approved five groups')
    expected=[]
    for sample in samples:
        if sample['split']!='train':raise ValueError('Train only')
        image=Path(sample['image_path']);label=Path(sample['label_path'])
        if (ROOT/image).resolve().parent!=(P.KIT/'data/train/images').resolve():raise ValueError('Outside train images')
        if f'{sample["scene"]}_img{sample["frame"]:05d}_' not in image.name:raise ValueError('Filename mismatch')
        if (ROOT/label).resolve()!=(P.KIT/'data/train/labels'/image.with_suffix('.txt').name).resolve():raise ValueError('Label path mismatch')
        for n,roi in enumerate(EXPECTED_ROIS[sample['scene']]):
            tile=f'T{n+1}';expected.append((sample['id']+':'+tile,sample['id'],sample['scene'],sample['frame'],tile,roi,2,f'inputs/{sample["id"]}_{tile}.png'))
    actual=[tuple(v[k] for k in ('id','sample_id','scene','frame','tile','roi','scale','input_file')) for v in views]
    if len(views)!=N_INPUTS or actual!=expected or len({v['id'] for v in views})!=N_INPUTS:raise ValueError('Fixed1239 view identity/order/geometry required')


def source_hashes():
    paths=[Path(__file__),ROOT/'tools/bus_crop_expansion.template.html',ROOT/'tests/test_bus_crop_expansion.py',
           ROOT/'tools/bus_crop_pilot.py',ROOT/'tools/bus_crop_runtime.py',ROOT/'tools/bus_static_audit.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def open_original(sample):
    path=ROOT/sample['image_path']
    if sha(path)!=sample['image_sha256']:raise ValueError('Original image changed')
    with Image.open(path) as im:
        if im.size!=(640,640):raise ValueError('Expected640 original')
        return im.convert('RGB')


def crop_input(image,roi):
    union_mask([roi])
    return image.crop(tuple(roi)).resize((640,640),Image.Resampling.BILINEAR)


def prior_snapshots():
    old=P.OUT
    snap=read(old/'ultralytics-run/final_checks.json')
    for name,h in snap['delivery_artifact_sha256'].items():
        if sha(old/name)!=h:raise ValueError('Prior experiment changed: '+name)
    return snap['delivery_artifact_sha256']


def prepare():
    if OUT.exists():raise FileExistsError('Preserve existing expansion; no overwrite')
    if Path(sys.executable).resolve()!=PYTHON.resolve():raise RuntimeError('Approved Python only')
    git=P.git()
    if git['branch']!='main' or git['head']!='c385eefc1b97d5334c52633232ff3aa5ec4fa908':raise ValueError('Revision changed')
    previous=prior_snapshots();weight=P.check_weight();refs=RT.checkpoint_globals(P)
    integrity=P.audit_module().integrity();protected=P.protected()
    inv,groups,samples,views=freeze_sources()
    OUT.mkdir();(OUT/'inputs').mkdir();(OUT/'roi_maps').mkdir()
    write(OUT/'preparation_status.json',dict(state='preparing',started_at=now(),model_attempts=0))
    for i,sample in enumerate(samples):
        if sha(ROOT/sample['label_path'])!=sample['label_sha256']:raise ValueError('Label changed')
        image=open_original(sample)
        for view in (v for v in views if v['sample_id']==sample['id']):
            crop=crop_input(image,view['roi']);crop.save(OUT/view['input_file'])
            view['input_rgb_sha256']=hashlib.sha256(crop.tobytes()).hexdigest();view['input_file_sha256']=sha(OUT/view['input_file'])
        if (i+1)%100==0:print(f'Prepared originals {i+1}/492',flush=True)
    for group in groups:
        first=next(s for s in samples if s['scene']==group['id']);image=open_original(first);draw=ImageDraw.Draw(image)
        for n,roi in enumerate(group['rois']):
            color=['#fb923c','#22d3ee','#a3e635','#f472b6'][n];draw.rectangle([roi[0],roi[1],roi[2]-1,roi[3]-1],outline=color,width=3)
            draw.text((roi[0]+4,roi[1]+4),f'T{n+1}',fill=color)
        image.save(OUT/f'roi_maps/{group["id"]}.png')
    identity=P.digest(dict(samples=[[s['id'],s['image_sha256'],s['label_sha256']] for s in samples],rois=EXPECTED_ROIS,params=PARAMS,weight=WEIGHT_HASH))
    config=dict(schema='bus-expansion-five-2x-v1',experiment_id=identity,groups=groups,images=492,inputs=N_INPUTS,params=PARAMS,
                weight=weight,checkpoint_globals=refs,inventory_sha256=sha(STATIC/'scene_manifest.json'),dataset_id=inv['dataset_id'],
                preflight_sha256=sha(PREFLIGHT/'preflight.json'),integrity_before=integrity,protected_before=protected,
                source_sha256=source_hashes(),git_before=git,created_at=now(),timeout_seconds=900,
                crop_rule='Per axis, <=320 center and clamp; >320 anchor left/right bounds (clamped0..320), Cartesian product. All five groups retained.',
                user_authorization='User selected方案1, then explicitly authorized启动重试即可 after the pre-worker CLI failure: one retry only, same five groups492 images1239 fixed2x inputs, same weight/params900s; preserve old failure evidence; oral group/segment feedback, no JSON form burden.',
                limits='Only selected heat regions, not all train potential bus positions. Multiple views independent; no cross-view NMS/fusion or automatic labels. Boundary truncation possible, even with heat pixel coverage.')
    write(OUT/'plan_config.json',config);write(OUT/'sample_manifest.json',samples);write(OUT/'crop_manifest.json',views)
    after=P.audit_module().integrity()
    if after!=integrity or P.protected()!=protected:raise ValueError('Original/protected file changed')
    files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()}
    write(OUT/'preparation_verified.json',dict(created_at=now(),prepared_images=492,prepared_views=N_INPUTS,integrity_after=after,
        protected_after=P.protected(),weight_after=P.check_weight(),artifact_sha256=files,prior_experiment_snapshot=previous))
    print(json.dumps(dict(prepared_images=492,prepared_views=N_INPUTS,experiment_id=identity),indent=2))


def validate_prepared(full_integrity=True):
    cfg=read(OUT/'plan_config.json');samples=read(OUT/'sample_manifest.json');views=read(OUT/'crop_manifest.json')
    validate_fixed(samples,views)
    inv,groups,expected_samples,expected_views=freeze_sources()
    if samples!=expected_samples or cfg['groups']!=groups or cfg['params']!=PARAMS:raise ValueError('Prepared scope drift')
    if cfg['source_sha256']!=source_hashes():raise ValueError('Source drift')
    if cfg['inventory_sha256']!=sha(STATIC/'scene_manifest.json') or cfg['preflight_sha256']!=sha(PREFLIGHT/'preflight.json'):raise ValueError('Source identity drift')
    if cfg['experiment_id']!=P.digest(dict(samples=[[s['id'],s['image_sha256'],s['label_sha256']] for s in samples],rois=EXPECTED_ROIS,params=PARAMS,weight=WEIGHT_HASH)):raise ValueError('Experiment identity drift')
    verification=read(OUT/'preparation_verified.json')
    for name,h in verification['artifact_sha256'].items():
        if sha(OUT/name)!=h:raise ValueError('Prepared artifact drift: '+name)
    for sample in samples:
        image=open_original(sample)
        if sha(ROOT/sample['label_path'])!=sample['label_sha256']:raise ValueError('Label changed')
        for view in (v for v in views if v['sample_id']==sample['id']):
            expected=crop_input(image,view['roi'])
            if hashlib.sha256(expected.tobytes()).hexdigest()!=view['input_rgb_sha256'] or sha(OUT/view['input_file'])!=view['input_file_sha256']:raise ValueError('Input drift')
            with Image.open(OUT/view['input_file']) as im:
                if im.size!=(640,640) or im.convert('RGB').tobytes()!=expected.tobytes():raise ValueError('PNG reconstruction drift')
    if full_integrity and P.audit_module().integrity()!=cfg['integrity_before']:raise ValueError('Original dataset changed')
    if P.protected()!=cfg['protected_before'] or P.check_weight()!=cfg['weight']:raise ValueError('Protected input changed')
    prior_snapshots()
    return cfg,samples,views


def install_guard(work):
    events=[];inputs={WEIGHT.resolve()}
    for p in (OUT/'plan_config.json',OUT/'crop_manifest.json',OUT/'sample_manifest.json',OUT/'run_config.json',
              Path(__file__),ROOT/'tools/bus_crop_pilot.py',ROOT/'tools/bus_crop_runtime.py'):inputs.add(p.resolve())
    for v in read(OUT/'crop_manifest.json'):inputs.add((OUT/v['input_file']).resolve())
    read_roots=[PYTHON.parent.resolve(),work.resolve()]
    def within(path,roots):return any(path==r or r in path.parents for r in roots)
    def hook(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo','socket.sendto','socket.sendmsg','subprocess.Popen','os.system','os.posix_spawn'):
            events.append(dict(type='denied_event',event=event));raise PermissionError('Offline worker guard')
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            raw=os.fsdecode(args[0])
            if sys.platform=='win32' and raw in ('/etc/os-release','/etc/dgx-release','/proc/device-tree/model','/proc/cpuinfo'):raise FileNotFoundError('Non-Windows metadata, no read')
            path=Path(raw).resolve();mode=args[1];flags=args[2]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            allowed=within(path,[work.resolve(),(OUT/'partial').resolve()]) if writing else within(path,read_roots) or path in inputs
            if path==(OUT/'worker_result.json').resolve():allowed=True
            if str(path).lower()==os.devnull.lower():allowed=True
            if not allowed:
                events.append(dict(type='file_denied',path=str(path),writing=bool(writing)));raise PermissionError('Outside worker scope: '+str(path))
    class No3LC(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('tlc','tlc_ultralytics','tlcconfig'):
                events.append(dict(type='3lc_import_denied',module=fullname));raise ImportError('No3LC in offline pilot')
    sys.meta_path.insert(0,No3LC());sys.addaudithook(hook)
    return events


def worker():
    cfg=read(OUT/'run_config.json');work=Path(cfg['scratch_work'])
    if os.environ.get('BUS_EXPANSION_TOKEN')!=cfg['token'] or Path.cwd()!=work:raise RuntimeError('Private one-shot worker')
    write(OUT/'worker_started.json',dict(started_at=now(),pid=os.getpid()))
    samples=read(OUT/'sample_manifest.json');views=read(OUT/'crop_manifest.json');validate_fixed(samples,views)
    events=install_guard(work);meta=dict(started_at=now(),completed=0,forward_shapes=[],preprocess_shapes=[]);started=time.perf_counter()
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect.predict import DetectionPredictor
        from ultralytics.utils import ONLINE,SETTINGS,SETTINGS_FILE
        if ONLINE or SETTINGS['sync'] or Path(SETTINGS_FILE).resolve()!=(work/'config/Ultralytics/settings.json').resolve():raise RuntimeError('Isolation not effective')
        if ultralytics.__version__!='8.4.165' or not torch.cuda.is_available():raise RuntimeError('Approved runtime/device unavailable')
        meta.update(versions=dict(python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,cuda=torch.version.cuda,numpy=np.__version__,pillow=Image.__version__),
                    device=torch.cuda.get_device_name(0),online=ONLINE,sync=SETTINGS['sync'],settings_file=str(SETTINGS_FILE))
        model=YOLO(str(WEIGHT),task='detect')
        if model.names!=RT.NAMES:raise ValueError('Four class mapping changed')
        model.model.register_forward_pre_hook(lambda module,args:meta['forward_shapes'].append(list(args[0].shape)))
        class CheckedPredictor(DetectionPredictor):
            def preprocess(self,ims):
                if len(ims)!=1 or ims[0].shape!=(640,640,3):raise ValueError('Invalid input')
                result=super().preprocess(ims)
                expected=torch.from_numpy(ims[0][...,::-1].copy()).permute(2,0,1).unsqueeze(0).float()/255
                if list(result.shape)!=[1,3,640,640] or not torch.allclose(result.cpu().float(),expected,rtol=0,atol=1e-7):raise ValueError('Preprocess shape/pixels drift')
                meta['preprocess_shapes'].append(list(result.shape));return result
        for i,view in enumerate(views):
            path=OUT/view['input_file']
            if sha(path)!=view['input_file_sha256']:raise ValueError('Input file changed')
            with Image.open(path) as im:image=im.convert('RGB')
            if hashlib.sha256(image.tobytes()).hexdigest()!=view['input_rgb_sha256']:raise ValueError('Input pixels changed')
            t=time.perf_counter();results=model.predict(source=image,predictor=CheckedPredictor,**PARAMS,classes=None,half=False,compile=False,
                verbose=False,save=False,save_txt=False,save_conf=False,save_crop=False,show=False,project=str(work/'runs'),name='fixed1239',exist_ok=True)
            if len(results)!=1 or list(results[0].orig_shape)!=[640,640]:raise ValueError('Output shape')
            effective={k:getattr(model.predictor.args,k) for k in PARAMS}
            if effective!=PARAMS:raise ValueError('Effective params changed')
            result=results[0];data=result.boxes.data.detach().cpu().tolist()
            record=dict(view,actual_tensor_shape=meta['preprocess_shapes'][-1],effective_params=effective,image_shape=list(result.orig_shape),
                elapsed_seconds=time.perf_counter()-t,speed_ms=result.speed,candidates=[RT.candidate(P,view,j,row) for j,row in enumerate(data)])
            validate_result(record,view)
            write(OUT/f'partial/{i:04d}.json',record);meta['completed']=i+1
            if (i+1)%25==0 or i+1==N_INPUTS:print(f'{i+1}/{N_INPUTS} {view["scene"]} frame={view["frame"]} tile={view["tile"]}',flush=True)
        if len(meta['preprocess_shapes'])!=N_INPUTS or len(meta['forward_shapes'])!=N_INPUTS+1:raise ValueError('Unexpected sample/warmup count')
        if any(s!=[1,3,640,640] for s in meta['forward_shapes']):raise ValueError('Unexpected model tensor')
        meta.update(state='completed',warmup_forward_calls=1,warmup_note='GPU library warmup reuses first input; synthetic random-box NMS warmup; no extra dataset images',
            three_lc_imported=any(k.split('.')[0] in ('tlc','tlc_ultralytics','tlcconfig') for k in sys.modules))
        if meta['three_lc_imported'] or events:raise RuntimeError('Isolation event')
    except BaseException:
        meta.update(state='failed',exception=traceback.format_exc());raise
    finally:
        meta.update(elapsed_seconds=time.perf_counter()-started,finished_at=now(),guard_events=events);write(OUT/'worker_result.json',meta)


def validate_result(record,view):
    for key,value in view.items():
        if record.get(key)!=value:raise ValueError('Result input identity drift: '+key)
    if record['actual_tensor_shape']!=[1,3,640,640] or record['image_shape']!=[640,640] or record['effective_params']!=PARAMS:raise ValueError('Result shape/params')
    if len(record['candidates'])>300:raise ValueError('Too many detections')
    for i,c in enumerate(record['candidates']):
        if c!=RT.candidate(P,view,i,c['xyxy_input']+[c['confidence'],c['cls']]):raise ValueError('Candidate identity/geometry')


def collect(views):
    result=[];errors=[]
    for i,view in enumerate(views):
        path=OUT/f'partial/{i:04d}.json'
        if not path.exists():
            if any((OUT/f'partial/{j:04d}.json').exists() for j in range(i+1,len(views))):errors.append(dict(index=i,error='Noncontiguous results'))
            break
        try:
            record=read(path);validate_result(record,view);result.append(record)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            errors.append(dict(index=i,error=type(exc).__name__+': '+str(exc)));break
    return result,errors


def box_iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=w*h
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.0


def label_box(label):
    cls,cx,cy,w,h=label;cx,cy,w,h=map(float,(cx,cy,w,h));return [640*(cx-w/2),640*(cy-h/2),640*(cx+w/2),640*(cy+h/2)]


def machine_summary(samples,records):
    summary=[]
    for sid in GROUPS:
        ss=[s for s in samples if s['scene']==sid];rr=[r for r in records if r['scene']==sid]
        bus=[c for r in rr for c in r['candidates'] if c['cls']==3];high={r['sample_id'] for r in rr if any(c['cls']==3 and c['band']=='high' for c in r['candidates'])}
        unlabelled={s['id'] for s in ss if s['bus_boxes']==0}
        summary.append(dict(scene=sid,images=len(ss),views=len(rr),all_class_candidates=sum(len(r['candidates']) for r in rr),
            bus_high=sum(c['band']=='high' for c in bus),bus_weak=sum(c['band']=='weak' for c in bus),
            images_with_high_bus=len(high),images_without_bus_labels=len(unlabelled),images_without_bus_labels_with_high_bus=len(high&unlabelled),
            edge_touch_bus_high=sum(c['band']=='high' and c['touches_crop_edge'] for c in bus)))
    return summary


def render(samples,records,cfg):
    if (OUT/'review.html').exists():raise FileExistsError('Preserve review page')
    grouped={s['id']:[] for s in samples}
    for record in records:grouped[record['sample_id']].append(record)
    data=dict(experiment_id=cfg['experiment_id'],groups=[dict(g) for g in cfg['groups']],summary=machine_summary(samples,records),samples=[])
    for sample in samples:
        item=dict(sample,image_url=os.path.relpath(ROOT/sample['image_path'],OUT).replace('\\','/'),views=[])
        gt=[label_box(label) for label in sample['labels'] if label[0]==3]
        for record in grouped[sample['id']]:
            view=dict(record,input_url=record['input_file'])
            view['bus_candidates']=[]
            for c in record['candidates']:
                if c['cls']==3:view['bus_candidates'].append(dict(c,max_original_bus_iou=max((box_iou(c['xyxy_original'],b) for b in gt),default=0)))
            del view['candidates'];item['views'].append(view)
        data['samples'].append(item)
    for group in data['groups']:group['heat_url']='../bus-static-review/'+group['heat_url']
    template=(ROOT/'tools/bus_crop_expansion.template.html').read_text(encoding='utf-8')
    if template.count('__EXPANSION_DATA__')!=1:raise ValueError('Template marker')
    text=template.replace('__EXPANSION_DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    with (OUT/'review.html').open('x',encoding='utf-8') as f:f.write(text)


def run():
    if (OUT/'run_config.json').exists() or (OUT/'partial').exists():raise FileExistsError('One attempt already started; no automatic retry')
    cfg,samples,views=validate_prepared()
    if Path(sys.executable).resolve()!=PYTHON.resolve():raise RuntimeError('Approved Python only')
    work=SCRATCH/('bus-expansion-'+uuid.uuid4().hex);work.mkdir();env=RT.environment(work)
    write(work/'config/Ultralytics/settings.json',RT.settings(work));token=uuid.uuid4().hex;env['BUS_EXPANSION_TOKEN']=token
    (OUT/'partial').mkdir()
    write(OUT/'run_config.json',dict(experiment_id=cfg['experiment_id'],token=token,scratch_work=str(work),timeout_seconds=900,
        model_attempts=1,planned_inputs=N_INPUTS,params=PARAMS,interface='ultralytics.YOLO',source_sha256=source_hashes(),started_at=now()))
    command=[str(PYTHON),'-I','-B',str(Path(__file__).resolve()),'_worker'];started=time.perf_counter();code=None;timed_out=False;launch_error=None
    with (OUT/'inference.log').open('x',encoding='utf-8') as log:
        try:code=subprocess.run(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900,check=False).returncode
        except subprocess.TimeoutExpired:timed_out=True
        except OSError as exc:launch_error=str(exc)
    duration=time.perf_counter()-started;records,errors=collect(views)
    success=code==0 and not timed_out and len(records)==N_INPUTS and not errors
    status=dict(experiment_id=cfg['experiment_id'],state='completed_awaiting_group_feedback' if success else 'timed_out' if timed_out else 'failed',
        model_attempts=1,model_inputs_completed=len(records),planned_inputs=N_INPUTS,inference_exit_code=code,timed_out=timed_out,
        process_elapsed_seconds=duration,result_errors=errors,launch_error=launch_error,finished_at=now())
    write(OUT/'run_status.json',status)
    write(OUT/'predictions.json',dict(experiment_id=cfg['experiment_id'],weight_sha256=WEIGHT_HASH,records=records))
    if success:
        write(OUT/'machine_summary.json',machine_summary(samples,records));render(samples,records,cfg)
    integrity=P.audit_module().integrity()
    if integrity!=cfg['integrity_before'] or P.protected()!=cfg['protected_before']:raise ValueError('Protected/original data changed')
    P.check_weight();prior_snapshots()
    result='# 已生成五组2×候选，等待组级目视反馈\n\n' if success else '# 扩样执行未完整完成，保留现场\n\n'
    result+=f'492张train，计划{N_INPUTS}视野，完成{len(records)}；模型进程1次，退出码{code}，超时{timed_out}，进程耗时{duration:.3f}秒。\n\n'
    result+='|组别|train图|视野|高bus框|弱bus框|有高bus的图|无bus原标注但有高bus的图|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for g in machine_summary(samples,records):result+=f'|{g["scene"]}|{g["images"]}|{g["views"]}|{g["bus_high"]}|{g["bus_weak"]}|{g["images_with_high_bus"]}|{g["images_without_bus_labels_with_high_bus"]}|\n'
    result+='\n高候选conf≥.25，弱候选.05≤conf<.25。框数跨视野可重复，不等于车辆数或漏标数。没有计算P/R/AP/真实误检率。原标签不完整，覆盖原标签热区不保证覆盖train中全部潜在bus；边界可截断车辆。\n\n每组ROI在预测前冻结，位置/重叠见plan_config.json。无跨视野融合、NMS或自动补标签。浏览页无需填写/导出JSON，可按组/区段直接口头反馈。只做离线学习，不用于竞赛训练/伪标签/提交。\n'
    (OUT/'RESULT.md').write_text(result,encoding='utf-8')
    (OUT/'README.md').write_text('# 五组热区2×扩样\n\n先读RESULT.md，打开review.html。所有492张按组和命名帧号保留，非视频时间/车辆身份。可筛“原图无bus标签且有高bus候选”，仅为待核对线索；不要据筛选数直接认定漏标。无需逐框填意见，直接给组级/区段反馈即可。\n\n左侧原图显示当前视野的映射候选或全部视野叠加，右侧为真实640PNG输入。蓝色原bus标签、绿色高候选、橙色弱候选；tile ID区分跨裁剪重复。与原bus标签的最大IoU只披露相似程度，不设自动真值阈值。\n\n```powershell\n& \'E:\\anaconda\\envs\\dl-3lc\\python.exe\' -I -B .\\tools\\bus_crop_expansion.py verify\n& \'E:\\anaconda\\envs\\dl-3lc\\python.exe\' -I -B -m unittest discover -s .\\tests -p \'test_bus*.py\'\n```\n\nverify只读，不再推理。run有独占保护，已启动即不自动重跑；不删除已有证据重试。所有新增缓存/工作目录在本会话scratch。历史13图试点和失败的单ROI预检均保留。\n',encoding='utf-8')
    hashes={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()}
    write(OUT/'verification.json',dict(created_at=now(),success=success,integrity_after=integrity,protected_after=P.protected(),
        weight_after=P.check_weight(),source_sha256=source_hashes(),artifact_sha256=hashes,prior_experiment_artifacts_unchanged=101,git_after=P.git()))
    print(json.dumps(dict(status=status,summary=machine_summary(samples,records)),ensure_ascii=False,indent=2))
    if not success:raise RuntimeError('Single attempt stopped; inspect evidence; no automatic retry')


def verify():
    cfg,samples,views=validate_prepared()
    v=read(OUT/'verification.json');status=read(OUT/'run_status.json');payload=read(OUT/'predictions.json')
    if not v['success'] or status['model_attempts']!=1 or status['inference_exit_code']!=0 or status['model_inputs_completed']!=N_INPUTS:raise ValueError('Run incomplete')
    if payload['experiment_id']!=cfg['experiment_id'] or payload['weight_sha256']!=WEIGHT_HASH or len(payload['records'])!=N_INPUTS:raise ValueError('Result batch mismatch')
    for record,view in zip(payload['records'],views):validate_result(record,view)
    if read(OUT/'machine_summary.json')!=machine_summary(samples,payload['records']):raise ValueError('Summary mismatch')
    for name,h in v['artifact_sha256'].items():
        if sha(OUT/name)!=h:raise ValueError('Artifact changed: '+name)
    if v['source_sha256']!=source_hashes() or v['integrity_after']!=cfg['integrity_before']:raise ValueError('Verification drift')
    meta=read(OUT/'worker_result.json')
    if meta['guard_events'] or meta['three_lc_imported'] or meta['warmup_forward_calls']!=1 or meta['completed']!=N_INPUTS:raise ValueError('Worker boundary/count')
    print(json.dumps(dict(verified=True,images=492,inputs=N_INPUTS,original_files_verified=17668,model_reexecuted=False),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','verify','_worker']);args=parser.parse_args()
    commands={'prepare':prepare,'run':run,'verify':verify,'_worker':worker}
    commands[args.command]()

if __name__=='__main__':main()
