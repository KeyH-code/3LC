#!/usr/bin/env python3
"""Fixed13/fixed39 offline pilot. Plain Ultralytics interface explicitly authorized.

Historical preparation/blocker evidence is immutable; new run has a separate directory.
Model imports occur only in a token-guarded, offline child process with a 900s limit.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/bus-crop-pilot'
STATIC = ROOT/'reports/bus-static-review'
KIT = ROOT/'multi-vehicle_kaggle_competition_starter_kit_files'
WEIGHT = ROOT/'.worktrees/score-search-20261019-150516/multi-vehicle_kaggle_competition_starter_kit_files/runs/detect/baseline_starter_val966_20261024/weights/best.pt'
WEIGHT_HASH = '3a90ef9c30824cefbe29a379814dac9cab64eeafc06c73243f24d32bf0dbe4b1'
FRAMES = [5,434,457,471,488,500,528,532,534,536,549,568,578]
CONDITIONS = {'A':[0,0,640,640], 'B':[186,0,506,320], 'C':[266,0,426,160]}
PARAMS = dict(device=0,batch=1,imgsz=640,rect=False,augment=False,conf=.05,iou=.7,max_det=300,agnostic_nms=False)
BLOCKER = 'Blocked before model import: tlc_ultralytics imports tlc, which eagerly calls ensure_key_activated(); supported credential-free/global-read-free offline initialization not established. No fallback authorized.'
PLAN = ROOT/'.pi/plan/bus局部放大小规模离线验证-13张图-39个视野-20261025-1642.md'
FEEDBACK = dict(dataset_id='4740058762988dbe0b43f9d4bc19764dc94c74122b7eb1cc3fc3da2fc625dd1b',
                sha256='1fcb3d79a4f1cd2edc632e571cbd11ebebd320bbfb7592934b8a4bdfe5291490',
                scope='User segment observations; not exhaustive image-level ground truth')


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def audit_module():
    spec=importlib.util.spec_from_file_location('static_audit',ROOT/'tools/bus_static_audit.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def protected():
    paths=[ROOT/'AGENTS.md']+[KIT/n for n in ('train.py','predict.py','register_tables.py','config.yaml','dataset.yaml')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def git():
    def run(*a):return subprocess.run(['git',*a],cwd=ROOT,check=True,capture_output=True,encoding='utf-8').stdout.strip()
    return dict(branch=run('branch','--show-current'),head=run('rev-parse','HEAD'),status=run('status','--short'))


def check_weight():
    # Only exact authorized weight: no enumeration, deserialization or fallback.
    if not WEIGHT.is_file() or WEIGHT.stat().st_size!=6245802 or sha(WEIGHT)!=WEIGHT_HASH:
        raise ValueError('Approved weight missing or size/hash mismatch')
    return dict(path=str(WEIGHT),bytes=6245802,sha256=WEIGHT_HASH,loaded=False)


def band(conf):
    if not math.isfinite(conf) or not .05<=conf<=1: raise ValueError('Invalid confidence')
    return 'high' if conf>=.25 else 'weak'


def validate_roi(roi):
    if len(roi)!=4 or not all(math.isfinite(v) for v in roi):raise ValueError('Invalid ROI')
    if not (0<=roi[0]<roi[2]<=640 and 0<=roi[1]<roi[3]<=640):raise ValueError('Invalid ROI')


def map_box(box,roi):
    validate_roi(roi)
    if len(box)!=4 or not all(math.isfinite(v) for v in box):raise ValueError('Non-finite xyxy')
    if not (0<=box[0]<=box[2]<=640 and 0<=box[1]<=box[3]<=640):raise ValueError('Outside input canvas')
    x0,y0,x1,y1=roi
    if x1<=x0 or y1<=y0:raise ValueError('Invalid ROI')
    return [x0+box[0]*(x1-x0)/640,y0+box[1]*(y1-y0)/640,
            x0+box[2]*(x1-x0)/640,y0+box[3]*(y1-y0)/640]


def inverse_box(box,roi):
    validate_roi(roi)
    if len(box)!=4 or not all(math.isfinite(v) for v in box):raise ValueError('Invalid original xyxy')
    if not (roi[0]<=box[0]<=box[2]<=roi[2] and roi[1]<=box[1]<=box[3]<=roi[3]):raise ValueError('Outside ROI')
    x0,y0,x1,y1=roi
    return [(box[0]-x0)*640/(x1-x0),(box[1]-y0)*640/(y1-y0),
            (box[2]-x0)*640/(x1-x0),(box[3]-y0)*640/(y1-y0)]


def validate_fixed(samples,views):
    if [s['frame'] for s in samples]!=FRAMES or len({s['id'] for s in samples})!=13:
        raise ValueError('Only frozen 13 unique samples permitted')
    for s in samples:
        if s['scene']!='MVI_20011' or s['split']!='train':raise ValueError('Invalid scope')
        p=Path(s['image_path'])
        if (ROOT/p).resolve().parent!=(KIT/'data/train/images').resolve() or f'MVI_20011_img{s["frame"]:05d}_' not in p.name:
            raise ValueError('Out-of-scope source')
        lp=(ROOT/s['label_path']).resolve()
        if lp!=(KIT/'data/train/labels'/p.with_suffix('.txt').name).resolve():raise ValueError('Out-of-scope label')
    expected=[(s['id'],s['frame'],c,roi) for s in samples for c,roi in CONDITIONS.items()]
    if [(v['sample_id'],v['frame'],v['condition'],v['roi']) for v in views]!=expected:raise ValueError('Only frozen 39 views permitted')
    for v in views:
        if v['scale']!=640/(v['roi'][2]-v['roi'][0]):raise ValueError('Changed scale')
    if len({v['id'] for v in views})!=39:raise ValueError('Duplicate view IDs')
    for v in views:
        expected_file=None if v['condition']=='A' else f'inputs/{v["frame"]:05d}_{v["condition"]}.png'
        if v['input_file']!=expected_file or v['id']!=v['sample_id']+':'+v['condition']:raise ValueError('Invalid input path/identity')


def make_input(sample,condition):
    if condition not in CONDITIONS:raise ValueError('Unapproved condition')
    p=(ROOT/sample['image_path']).resolve()
    if p.parent!=(KIT/'data/train/images').resolve():raise ValueError('Invalid source')
    if sha(p)!=sample['image_sha256']:raise ValueError('Source changed')
    with Image.open(p) as im:
        if im.size!=(640,640):raise ValueError('Unexpected dimensions')
        im=im.convert('RGB')
        if condition!='A':im=im.crop(tuple(CONDITIONS[condition])).resize((640,640),Image.Resampling.BILINEAR)
        return im


def source_hashes():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),ROOT/'tests/test_bus_crop_pilot.py']}


def artifact_hashes():
    paths=[OUT/n for n in ('README.md','RESULT.md','INTERFACE_BLOCKER.md','plan_config.json','sample_manifest.json','crop_manifest.json','run_status.json')]
    paths+=list((OUT/'inputs').glob('*.png'))
    return {p.relative_to(OUT).as_posix():sha(p) for p in paths}


def prepare():
    if OUT.exists():raise FileExistsError('Output exists; preserve it, no overwrite')
    snap=git()
    if snap['branch']!='main' or snap['head']!='c385eefc1b97d5334c52633232ff3aa5ec4fa908':raise ValueError('Unexpected revision')
    protected_before=protected()
    weight=check_weight();before=audit_module().integrity()
    print('Original data before: 17668/17668',flush=True)
    previous=read(STATIC/'verification.json')
    if sha(STATIC/'scene_manifest.json')!=previous['artifact_sha256']['scene_manifest.json']:raise ValueError('Inventory changed')
    inv=read(STATIC/'scene_manifest.json');scene=next(s for s in inv['scenes'] if s['id']=='MVI_20011')
    heat=STATIC/scene['maps']['counts']
    if sha(heat)!=previous['artifact_sha256'][scene['maps']['counts']]:raise ValueError('Heat source changed')
    with Image.open(heat) as im:arr=np.asarray(im)
    y,x=np.nonzero(arr[:,:,0].astype(np.uint16)+256*arr[:,:,1].astype(np.uint16))
    bounds=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
    if bounds!=[331,0,362,76]:raise ValueError('Heat ROI changed')
    for c in ('B','C'):
        x0,y0,x1,y1=CONDITIONS[c]
        if not (x0<=bounds[0]<bounds[2]<=x1 and y0<=bounds[1]<bounds[3]<=y1):raise ValueError('Heat clipped')
    samples=sorted([inv['images'][i] for i in scene['train_ids'] if inv['images'][i]['frame'] in FRAMES],key=lambda s:s['frame'])
    views=[dict(id=s['id']+':'+c,sample_id=s['id'],frame=s['frame'],condition=c,roi=roi,scale=640/(roi[2]-roi[0]),
                input_file=None if c=='A' else f'inputs/{s["frame"]:05d}_{c}.png') for s in samples for c,roi in CONDITIONS.items()]
    validate_fixed(samples,views)
    if inv['dataset_id']!=FEEDBACK['dataset_id']:raise ValueError('Feedback dataset identity changed')
    for s in samples:
        if sha(ROOT/s['label_path'])!=s['label_sha256']:raise ValueError('Label changed')
    experiment_id=digest(dict(samples=[[s['id'],s['image_sha256'],s['label_sha256']] for s in samples],conditions=CONDITIONS,params=PARAMS,weight=WEIGHT_HASH))
    OUT.mkdir();(OUT/'inputs').mkdir()
    for s in samples:
        for c in ('B','C'):make_input(s,c).save(OUT/f'inputs/{s["frame"]:05d}_{c}.png')
        roi=make_input(s,'A');d=ImageDraw.Draw(roi)
        for c,color in [('B','#22d3ee'),('C','#fb923c')]:
            d.rectangle(CONDITIONS[c],outline=color,width=2);d.text((CONDITIONS[c][0]+5,5),c,fill=color)
        roi.save(OUT/f'inputs/{s["frame"]:05d}_ROI.png')
    for v in views:
        s=next(s for s in samples if s['id']==v['sample_id'])
        v['input_rgb_sha256']=hashlib.sha256(make_input(s,v['condition']).tobytes()).hexdigest()
        v['input_file_sha256']=sha(OUT/v['input_file']) if v['input_file'] else None
    config=dict(schema='bus-crop-pilot-v1',experiment_id=experiment_id,frames=FRAMES,conditions=CONDITIONS,params=PARAMS,
                weight=weight,heat_bounds=bounds,heat_sha256=sha(heat),source_dataset_id=inv['dataset_id'],
                inventory_sha256=sha(STATIC/'scene_manifest.json'),integrity_before=before,protected_before=protected_before,
                git_before=snap,created_at=now(),model_execution_authorized_but_blocked=True,timeout_seconds_if_unblocked=900,
                approved_plan=dict(path=PLAN.relative_to(ROOT).as_posix(),sha256=sha(PLAN)),feedback_source=FEEDBACK,
                preparation_runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,pillow=Image.__version__))
    write(OUT/'plan_config.json',config);write(OUT/'sample_manifest.json',samples);write(OUT/'crop_manifest.json',views)
    write(OUT/'run_status.json',dict(experiment_id=experiment_id,state='blocked_before_model_import',reason=BLOCKER,
          prepared_views=39,model_process_started=False,model_inputs_completed=0,model_attempts=0,
          inference_exit_code=None,timed_out=False,predictions_created=False,created_at=now()))
    (OUT/'README.md').write_text('''# Bus局部放大小规模试验：准备完成，推理阻塞\n\n13图/39视野已冻结，26张局部640输入和13张ROI示意已生成。A不复制原图。B=[186,0,506,320)，C=[266,0,426,160)，双线性2×/4×；无超分辨率。\n\n**没有预测，没有人工候选审核页，不能判断想法是否有效。** run明确拒绝执行，绝不自动切换接口、读取凭据或下载。\n\n阻塞依据见INTERFACE_BLOCKER.md。只读verify不导入任何模型库，只验证数据、指定权重哈希、准备图像和坐标。\n\n```powershell\n$py='E:\\anaconda\\envs\\dl-3lc\\python.exe'\n& $py -I -B -m unittest discover -s .\\tests -p test_bus_crop_pilot.py -v\n& $py -I -B .\\tools\\bus_crop_pilot.py verify\n# prepare是独占创建，当前目录已存在，不要删除或重建覆盖。\n# run会报明确阻塞，不是推理入口。\n```\n\n原标签不完整，任何未来预测都仅作候选，不计算AP/P/R，不用于标签/训练/竞赛提交；本轮不扩大样本、不更改批准参数。\n''',encoding='utf-8')
    (OUT/'RESULT.md').write_text('# 状态：尚未进行可行性模型验证\n\n准备13图、39视野已完成；因tlc_ultralytics导入时强制认证与无凭据边界冲突，依批准计划第3节在模型导入前停止。\n\n模型进程启动0次，模型输入0项，权重仅核对字节/SHA256未加载，未产生预测。没有推理退出码（不是退出0），没有候选质量结论。\n\n未生成审核页：不存在真实候选时不生成假结果页。原图/标签/官方入口/配置未改。\n',encoding='utf-8')
    (OUT/'INTERFACE_BLOCKER.md').write_text('''# 离线接口阻塞（静态源码核查，未执行import）\n\n库根：E:\\anaconda\\envs\\dl-3lc\\Lib\\site-packages\\。\n\n- tlc_ultralytics/engine/model.py 顶层import tlc；predictor虽为普通DetectionPredictor，但导入先发生。\n- tlc/__init__.py导入_core，tlc/_core/__init__.py调用init_global_objects。\n- tlc/_core/init_global_objects.py:40起先调用ensure_key_activated，随后才配置/Telemetry/内部表初始化。\n- _tlc_auth_state.py:639起要求有效认证handle，否则走license/API-key激活；971起的_resolve_api_key在未提供key时回退到native默认凭据文件。\n- TLC_CONFIG_FILE可关闭用户配置发现，TLC_DISABLE_3LC_ROOT_DIR=1可禁根标记发现，TLC_DEFAULT_ROOT_DIR可隔离Python默认数据路径；TLC_SENTRY_ENVIRONMENT=local禁用3LC Sentry，TLC_COLLECTION_DISABLE=true可禁collection。**这些都不是跳过导入认证的开关。**\n- 凭据候选路径来自native组件，未核实它是否遵循APPDATA/LOCALAPPDATA/HOME重定向；Python socket/open保护不能证明native完全受控。没有读取该native组件中的凭据值、没有反编译或执行认证。\n\n批准计划要求：若无法在不读取全局配置/凭据、不产生未授权副作用的情况下加载，则报告阻塞，不自行切换模型栈。故本轮未导入模型库、未加载权重、未运行推理。\n\n可能的后续变更（**本轮未授权或执行**）：单独批准同环境普通ultralytics接口离线加载同一best.pt、固定样本/参数不变；或者另行明确授权3LC认证及精确凭据路径/网络范围。未修改依赖或monkeypatch认证。\n''',encoding='utf-8')
    after=audit_module().integrity()
    if before!=after:raise ValueError('Original data changed')
    if protected()!=protected_before:raise ValueError('Protected input changed')
    write(OUT/'verification.json',dict(created_at=now(),state='preparation_verified_inference_blocked',integrity_before=before,integrity_after=after,
        weight_after=check_weight(),protected_after=protected(),source_sha256=source_hashes(),artifact_sha256=artifact_hashes(),
        model_loaded=False,inference_executed=False,prepared_views=39,git_after=git()))
    print(json.dumps(dict(prepared=39,inference_executed=False,reason=BLOCKER),indent=2))


def runtime():
    spec=importlib.util.spec_from_file_location('bus_crop_runtime',ROOT/'tools/bus_crop_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run():
    runtime().run(sys.modules[__name__])


def verify_preparation():
    v=read(OUT/'verification.json');cfg=read(OUT/'plan_config.json');samples=read(OUT/'sample_manifest.json');views=read(OUT/'crop_manifest.json')
    validate_fixed(samples,views)
    if cfg['frames']!=FRAMES or cfg['params']!=PARAMS or cfg['conditions']!=CONDITIONS:raise ValueError('Config drift')
    if cfg['approved_plan']['sha256']!=sha(PLAN) or cfg['feedback_source']!=FEEDBACK:raise ValueError('Approval/source drift')
    inv=read(STATIC/'scene_manifest.json')
    if sha(STATIC/'scene_manifest.json')!=cfg['inventory_sha256']:raise ValueError('Inventory drift')
    if samples!=sorted([s for s in inv['images'].values() if s['scene']=='MVI_20011' and s['split']=='train' and s['frame'] in FRAMES],key=lambda s:s['frame']):raise ValueError('Sample identity drift')
    status=read(OUT/'run_status.json')
    if status['state']!='blocked_before_model_import' or status['model_attempts']!=0 or status['model_inputs_completed']!=0 or status['inference_exit_code'] is not None:raise ValueError('Blocked state changed')
    if any((OUT/n).exists() for n in ('predictions.json','review.html')):raise ValueError('Unexpected inference artifacts')
    if artifact_hashes()!=v['artifact_sha256']:raise ValueError('Prepared artifact hash mismatch')
    for s in samples:
        if sha(ROOT/s['image_path'])!=s['image_sha256'] or sha(ROOT/s['label_path'])!=s['label_sha256']:raise ValueError('Sample changed')
    for view in views:
        s=next(s for s in samples if s['id']==view['sample_id']);expected=make_input(s,view['condition'])
        if hashlib.sha256(expected.tobytes()).hexdigest()!=view['input_rgb_sha256']:raise ValueError('RGB mismatch')
        if view['input_file']:
            if sha(OUT/view['input_file'])!=view['input_file_sha256']:raise ValueError('PNG mismatch')
            with Image.open(OUT/view['input_file']) as im:
                if im.size!=(640,640) or im.convert('RGB').tobytes()!=expected.tobytes():raise ValueError('Crop reconstruction mismatch')
        for box in ([0.,0.,640.,640.],[.25,13.75,411.125,600.5]):
            if not np.allclose(inverse_box(map_box(box,view['roi']),view['roi']),box,rtol=0,atol=1e-10):raise ValueError('Geometry mismatch')
    if protected()!=cfg['protected_before'] or check_weight()!=v['weight_after']:raise ValueError('Protected input changed')
    if audit_module().integrity()!=v['integrity_before'] or v['integrity_before']!=v['integrity_after']:raise ValueError('Original data changed')
    print(json.dumps(dict(preparation_verified=True,original_files=17668,prepared_views=39,crop_pngs=26,roi_pngs=13,
                         inference_executed=False,model_loaded=False,run_is_blocked=True),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','verify','run','verify-run','_worker','summarize'])
    p.add_argument('feedback',nargs='?');args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='verify':verify_preparation()
    elif args.command=='run':run()
    elif args.command=='verify-run':runtime().verify_run(sys.modules[__name__])
    elif args.command=='_worker':runtime().worker(sys.modules[__name__])
    else:
        if not args.feedback:raise ValueError('Explicit in-project feedback JSON required')
        target=Path(args.feedback).resolve()
        if ROOT not in target.parents or target.suffix.lower()!='.json':raise ValueError('Feedback path outside project')
        rt=runtime();predictions=read(rt.directory(sys.modules[__name__])/'predictions.json')
        print(json.dumps(rt.summarize(sys.modules[__name__],read(target),predictions),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
