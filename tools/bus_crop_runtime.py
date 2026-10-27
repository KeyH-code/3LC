"""One authorized fixed39 offline run; imported by bus_crop_pilot, never a general predictor."""
from __future__ import annotations
import collections
import importlib.abc
import json
import math
import os
from pathlib import Path
import pickletools
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

PYTHON = Path(r'E:\anaconda\envs\dl-3lc\python.exe')
ENVROOT = PYTHON.parent
SCRATCH = Path(r'E:\pi-desktop\data\data\scratch\112b9029-dd22-4fc5-9132-2dab3956e6cc')
NAMES = {0:'truck',1:'car',2:'van',3:'bus'}
STATUSES = ('usable','adjust','wrong_class','false_positive','uncertain','outside_or_truncated','not_detected')
INTERFACE_AUTH = '学习者本次明确授权：改用普通ultralytics.YOLO进行离线推理；原批准13图/39视野、权重、参数和900秒限制不变。'


def directory(p): return p.OUT/'ultralytics-run'

def exclusive_directory(path):
    Path(path).mkdir(exist_ok=False)


def checkpoint_globals(p):
    p.check_weight()
    with zipfile.ZipFile(p.WEIGHT) as z:
        names=[n for n in z.namelist() if n.endswith('/data.pkl')]
        if len(names)!=1: raise ValueError('Unexpected checkpoint archive')
        ops=list(pickletools.genops(z.read(names[0])))
    if any(op.name=='STACK_GLOBAL' for op,_,_ in ops): raise ValueError('Unreviewed pickle protocol')
    refs=sorted({arg for op,arg,_ in ops if op.name=='GLOBAL'})
    allowed=('ultralytics.nn.','torch','collections ','__builtin__ set','builtins set')
    if any(not ref.startswith(allowed) for ref in refs): raise ValueError('Unexpected serialized module; no remapping authorized')
    return refs


def sources(p):
    return {q.relative_to(p.ROOT).as_posix():p.sha(q) for q in (
        p.ROOT/'tools/bus_crop_pilot.py',Path(__file__),p.ROOT/'tools/bus_crop_review.template.html',p.ROOT/'tests/test_bus_crop_pilot.py')}


def settings(work):
    d=dict(settings_version='0.0.8',datasets_dir=str(work/'datasets'),weights_dir=str(work/'weights'),
           runs_dir=str(work/'runs'),uuid='0'*64,sync=False,api_key='',openai_api_key='')
    d.update({k:False for k in ('clearml','comet','dvc','mlflow','raytune','tensorboard','wandb','vscode_msg','openvino_msg')})
    return d


def environment(work):
    # Only OS/runtime variables are inherited; no inherited service keys or user config paths.
    env={k:os.environ[k] for k in ('SystemRoot','WINDIR','PATH','PATHEXT','COMSPEC','NUMBER_OF_PROCESSORS','CUDA_VISIBLE_DEVICES') if k in os.environ}
    for name in ('HOME','USERPROFILE','APPDATA','LOCALAPPDATA','TEMP','TMP','XDG_CACHE_HOME','XDG_CONFIG_HOME',
                 'MPLCONFIGDIR','TORCH_HOME','TORCH_EXTENSIONS_DIR','CUDA_CACHE_PATH','NUMBA_CACHE_DIR','HF_HOME'):
        d=work/name.lower();d.mkdir();env[name]=str(d)
    config=work/'config';(config/'Ultralytics').mkdir(parents=True)
    env.update(YOLO_CONFIG_DIR=str(config),YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',
               YOLO_VERBOSE='false',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',WANDB_MODE='disabled',
               WANDB_DISABLED='true',COMET_MODE='DISABLED',TLC_COLLECTION_DISABLE='true')
    return env


def candidate(p, view, index, row):
    if len(row)!=6 or not all(math.isfinite(float(x)) for x in row): raise ValueError('Malformed detection')
    x0,y0,x1,y1,conf,cls=map(float,row)
    if cls!=int(cls) or int(cls) not in NAMES: raise ValueError('Invalid class')
    band=p.band(conf);box=[x0,y0,x1,y1];mapped=p.map_box(box,view['roi'])
    distances=[x0,y0,640-x1,640-y1]
    return dict(id=view['id']+f':p{index:03d}',cls=int(cls),class_name=NAMES[int(cls)],confidence=conf,band=band,
                xyxy_input=box,xyxy_original=mapped,edge_distances_input=distances,
                edge_distances_original=[v/view['scale'] for v in distances],
                touches_crop_edge=any(v<=0 for v in distances),edge_order=['left','top','right','bottom'])


def validate_predictions(p, payload, views, complete=True):
    if payload.get('experiment_id')!=p.read(p.OUT/'plan_config.json')['experiment_id'] or payload.get('weight_sha256')!=p.WEIGHT_HASH:
        raise ValueError('Prediction identity mismatch')
    records=payload['records']
    if len(records)>39 or (complete and len(records)!=39): raise ValueError('Unexpected result count')
    for record,view in zip(records,views):
        for k in ('id','sample_id','frame','condition','roi','scale','input_rgb_sha256','input_file_sha256'):
            if record[k]!=view[k]: raise ValueError('Result source mismatch: '+k)
        if record['actual_tensor_shape']!=[1,3,640,640] or record['image_shape']!=[640,640]: raise ValueError('Input shape mismatch')
        if record['effective_params']!=p.PARAMS: raise ValueError('Effective parameter mismatch')
        if not 0<=len(record['candidates'])<=300: raise ValueError('max_det mismatch')
        for i,c in enumerate(record['candidates']):
            expected=candidate(p,view,i,c['xyxy_input']+[c['confidence'],c['cls']])
            if c!=expected: raise ValueError('Candidate mapping/identity mismatch')
    return records


def guard(p, work, run):
    """Defense in depth for Python I/O/network/imports, not a native-code sandbox."""
    events=[]
    project_inputs={p.WEIGHT.resolve(),(p.OUT/'plan_config.json').resolve(),(p.OUT/'sample_manifest.json').resolve(),
                    (p.OUT/'crop_manifest.json').resolve(),(run/'run_config.json').resolve()}
    project_inputs.update((p.ROOT/name).resolve() for name in ('tools/bus_crop_pilot.py','tools/bus_crop_runtime.py'))
    entry=Path(sys.argv[0]).resolve()
    if SCRATCH.resolve() in entry.parents:project_inputs.add(entry)
    for s in p.read(p.OUT/'sample_manifest.json'): project_inputs.add((p.ROOT/s['image_path']).resolve())
    allowed_read=[ENVROOT.resolve(),work.resolve(),run.resolve()]
    def inside(path,roots): return any(path==root or root in path.parents for root in roots)
    def audit(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo','socket.sendto','socket.sendmsg'):
            events.append({'type':'network_denied','event':event});raise PermissionError('Offline network guard')
        if event in ('subprocess.Popen','os.system','os.posix_spawn'):
            events.append({'type':'subprocess_denied','event':event});raise PermissionError('Worker cannot spawn subprocesses')
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            if sys.platform=='win32' and os.fsdecode(args[0]) in ('/etc/os-release','/etc/dgx-release','/proc/device-tree/model','/proc/cpuinfo'):
                raise FileNotFoundError('Non-Windows metadata probe; no file read')
            path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            roots=[work.resolve(),run.resolve()] if writing else allowed_read
            if not inside(path,roots) and not (not writing and path in project_inputs) and str(path).lower()!=os.devnull.lower():
                events.append({'type':'file_denied','path':str(path),'writing':bool(writing)})
                raise PermissionError('Out-of-scope Python file access: '+str(path))
    class No3LC(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('tlc','tlc_ultralytics','tlcconfig'):
                events.append({'type':'3lc_import_denied','module':fullname});raise ImportError('3LC not authorized in plain run')
    sys.meta_path.insert(0,No3LC());sys.addaudithook(audit)
    return events


def worker(p):
    run=directory(p);cfg=p.read(run/'run_config.json')
    if os.environ.get('BUS_CROP_RUN_TOKEN')!=cfg['worker_token'] or Path.cwd()!=Path(cfg['scratch_work']):
        raise RuntimeError('Private worker: use run, not direct invocation')
    with (run/'worker_started.json').open('x',encoding='utf-8') as f: json.dump({'at':p.now(),'pid':os.getpid()},f)
    work=Path(cfg['scratch_work']);samples=p.read(p.OUT/'sample_manifest.json');views=p.read(p.OUT/'crop_manifest.json')
    p.validate_fixed(samples,views)
    events=guard(p,work,run)
    started=time.perf_counter();meta={'started_at':p.now(),'completed':0,'forward_shapes':[]}
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect.predict import DetectionPredictor
        from ultralytics.utils import ONLINE,SETTINGS,SETTINGS_FILE
        if ONLINE or SETTINGS['sync'] or Path(SETTINGS_FILE).resolve()!=(work/'config/Ultralytics/settings.json').resolve():
            raise RuntimeError('Offline settings not effective')
        if ultralytics.__version__!='8.4.165': raise RuntimeError('Unreviewed ultralytics version')
        if not torch.cuda.is_available(): raise RuntimeError('Approved device0 unavailable; no fallback')
        meta.update(versions={'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,
                             'numpy':p.np.__version__,'pillow':p.Image.__version__,'cuda_runtime':torch.version.cuda},
                    device=torch.cuda.get_device_name(0),settings_file=str(SETTINGS_FILE),online=ONLINE,sync=SETTINGS['sync'])
        model=YOLO(str(p.WEIGHT),task='detect')
        if model.names!=NAMES: raise ValueError('Expected four-class mapping truck/car/van/bus')
        def forward_hook(module,args):
            meta['forward_shapes'].append(list(args[0].shape))
        model.model.register_forward_pre_hook(forward_hook)
        preprocess_shapes=[]
        class FixedPredictor(DetectionPredictor):
            def preprocess(self,ims):
                if len(ims)!=1 or ims[0].shape!=(640,640,3): raise ValueError('Non-fixed input')
                result=super().preprocess(ims)
                if list(result.shape)!=[1,3,640,640]: raise ValueError('Unexpected model preprocessing')
                expected=torch.from_numpy(ims[0][...,::-1].copy()).permute(2,0,1).unsqueeze(0).float()/255
                if not torch.allclose(result.cpu().float(),expected,rtol=0,atol=1e-7): raise ValueError('Resize/padding/color drift')
                preprocess_shapes.append(list(result.shape));return result
        for i,view in enumerate(views):
            sample=next(s for s in samples if s['id']==view['sample_id']);image=p.make_input(sample,view['condition'])
            if p.hashlib.sha256(image.tobytes()).hexdigest()!=view['input_rgb_sha256']: raise ValueError('Input RGB drift')
            t=time.perf_counter();results=model.predict(source=image,predictor=FixedPredictor,**p.PARAMS,
                classes=None,half=False,compile=False,verbose=False,save=False,save_txt=False,save_conf=False,
                save_crop=False,show=False,project=str(work/'runs'),name='fixed39',exist_ok=True)
            if len(results)!=1 or tuple(results[0].orig_shape)!=(640,640): raise ValueError('Unexpected output shape')
            effective={k:getattr(model.predictor.args,k) for k in p.PARAMS}
            if effective!=p.PARAMS: raise ValueError('Predictor changed parameters')
            result=results[0];data=result.boxes.data.detach().cpu().tolist()
            record=dict(view,actual_tensor_shape=preprocess_shapes[-1],image_shape=list(result.orig_shape),
                        effective_params=effective,elapsed_seconds=time.perf_counter()-t,speed_ms=result.speed,
                        candidates=[candidate(p,view,j,row) for j,row in enumerate(data)])
            p.write(run/f'partial/{i:02d}.json',record)
            meta['completed']=i+1;print(f'{i+1}/39 frame={view["frame"]} condition={view["condition"]} candidates={len(data)}',flush=True)
        meta.update(state='completed',model_loaded=True,preprocess_shapes=preprocess_shapes,
                    warmup_forward_calls=len(meta['forward_shapes'])-39,warmup_note='Library PyTorch GPU warmup reuses first preprocessed input; NMS also uses random synthetic boxes. No other dataset images.')
        if len(preprocess_shapes)!=39 or len(meta['forward_shapes'])!=40: raise ValueError('Unexpected sample/warmup count')
        if any(x!=[1,3,640,640] for x in meta['forward_shapes']): raise ValueError('Unexpected forward input shape')
        if any(k.split('.')[0] in ('tlc','tlc_ultralytics','tlcconfig') for k in sys.modules): raise RuntimeError('3LC imported')
        meta['three_lc_imported']=False
    except BaseException:
        meta.update(state='failed',exception=traceback.format_exc());raise
    finally:
        meta.update(elapsed_seconds=time.perf_counter()-started,finished_at=p.now(),guard_events=events)
        p.write(run/'worker_result.json',meta)


def build_payload(p,run,views,errors=None):
    errors=[] if errors is None else errors
    cfg=p.read(p.OUT/'plan_config.json')
    payload=dict(schema='bus-crop-predictions-v1',experiment_id=cfg['experiment_id'],weight_sha256=p.WEIGHT_HASH,
                 dataset_id=cfg['source_dataset_id'],model_interface='ultralytics.YOLO',requested_params=p.PARAMS,records=[])
    for i in range(39):
        path=run/f'partial/{i:02d}.json'
        if not path.exists():
            if any((run/f'partial/{j:02d}.json').exists() for j in range(i+1,39)):
                errors.append(dict(file=str(path),error='Non-contiguous partial outputs'))
            break
        try:
            record=p.read(path)
            trial=dict(payload,records=payload['records']+[record])
            validate_predictions(p,trial,views,complete=False)
            payload['records'].append(record)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            errors.append(dict(file=str(path),error=type(exc).__name__+': '+str(exc)))
            break
    return payload


def counts(payload):
    result={c:{'all_classes':{'high':0,'weak':0},'bus':{'high':0,'weak':0}} for c in 'ABC'}
    for r in payload['records']:
        for c in r['candidates']:
            result[r['condition']]['all_classes'][c['band']]+=1
            if c['cls']==3: result[r['condition']]['bus'][c['band']]+=1
    return result


def run(p):
    dest=directory(p)
    if dest.exists(): raise FileExistsError('Single-run directory exists; preserve all evidence, no automatic retry')
    if Path(sys.executable).resolve()!=PYTHON.resolve(): raise RuntimeError('Use approved Python only')
    p.verify_preparation()
    revision=p.git()
    if revision['branch']!='main' or revision['head']!='c385eefc1b97d5334c52633232ff3aa5ec4fa908': raise ValueError('Revision drift')
    refs=checkpoint_globals(p);cfg=p.read(p.OUT/'plan_config.json');views=p.read(p.OUT/'crop_manifest.json')
    exclusive_directory(dest);(dest/'partial').mkdir()
    work=SCRATCH/('bus-crop-ultralytics-'+uuid.uuid4().hex);work.mkdir()
    env=environment(work);p.write(work/'config/Ultralytics/settings.json',settings(work))
    token=uuid.uuid4().hex;env['BUS_CROP_RUN_TOKEN']=token
    run_cfg=dict(experiment_id=cfg['experiment_id'],interface_authorization=INTERFACE_AUTH,interface='ultralytics.YOLO',
                 requested_params=p.PARAMS,weight_sha256=p.WEIGHT_HASH,timeout_seconds=900,planned_views=[v['id'] for v in views],
                 prepared_artifact_sha256=p.artifact_hashes(),source_sha256=sources(p),scratch_work=str(work),worker_token=token,
                 checkpoint_globals=refs,git_before=revision,created_at=p.now(),offline_environment={k:env[k] for k in env if k not in ('PATH','SystemRoot','WINDIR')})
    p.write(dest/'run_config.json',run_cfg)
    p.write(dest/'start_status.json',dict(state='starting',model_attempts=1,model_inputs_completed=0,started_at=p.now()))
    start=time.perf_counter();timed_out=False
    command=[str(PYTHON),'-I','-B',str(p.ROOT/'tools/bus_crop_pilot.py'),'_worker']
    with (dest/'inference.log').open('x',encoding='utf-8') as log:
        try:
            proc=subprocess.run(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900,check=False)
            code=proc.returncode
        except subprocess.TimeoutExpired:
            timed_out=True;code=None
    result_errors=[]
    payload=build_payload(p,dest,views,result_errors)
    success=code==0 and not timed_out and len(payload['records'])==39 and not result_errors
    status=dict(experiment_id=cfg['experiment_id'],state='completed_awaiting_human_review' if success else 'timed_out' if timed_out else 'failed',
                model_attempts=1,model_inputs_completed=len(payload['records']),inference_exit_code=code,timed_out=timed_out,
                process_elapsed_seconds=time.perf_counter()-start,finished_at=p.now(),command=command,result_errors=result_errors)
    p.write(dest/'run_status.json',status)
    if payload['records']: p.write(dest/'predictions.json',payload)
    if success:
        validate_predictions(p,payload,views)
        p.write(dest/'candidate_counts.json',counts(payload));render(p)
    integrity=p.audit_module().integrity()
    if integrity!=cfg['integrity_before'] or p.protected()!=cfg['protected_before']: raise ValueError('Original/protected files changed')
    p.check_weight()
    artifacts={q.relative_to(dest).as_posix():p.sha(q) for q in dest.rglob('*') if q.is_file()}
    p.write(dest/'verification.json',dict(created_at=p.now(),integrity_after=integrity,protected_after=p.protected(),
        weight_after=p.check_weight(),source_sha256=sources(p),artifact_sha256=artifacts,git_after=p.git(),successful_model_run=success))
    summary='# 已生成候选，等待人工审核\n\n' if success else '# 本次推理未完整完成\n\n'
    summary+=f'接口：普通ultralytics.YOLO；完成 {len(payload["records"])}/39 项；退出码 {code}；超时 {timed_out}；子进程耗时 {status["process_elapsed_seconds"]:.3f} 秒。\n\n'
    if success:
        summary+='|条件|四类≥0.25|四类0.05～0.25|bus≥0.25|bus0.05～0.25|\n|---|---:|---:|---:|---:|\n'
        for condition,c in counts(payload).items():
            summary+=f'|{condition}|{c["all_classes"]["high"]}|{c["all_classes"]["weak"]}|{c["bus"]["high"]}|{c["bus"]["weak"]}|\n'
        summary+='\n打开review.html审核13张图。机器产生候选不代表框可用；尚无人审，不计算AP/P/R或漏标率。A/B/C未融合；跨视野同一目标需人工填写同一目标组号，不能重复算新增监督。\n'
    summary+='\n原图/标签/官方入口/默认配置/AGENTS与权重未改。仅离线学习，不用于训练/伪标签/提交。样本单场景且强相关，不能外推bus整体或car。\n\n历史阻塞证据在上级目录，已保留；本次只按追加授权变更接口。模型warmup、版本、有效参数和保护事件见worker_result.json。verify-run只读，不重新推理。run拒绝第二次执行；无自动重试。\n'
    (dest/'RESULT.md').write_text(summary,encoding='utf-8')
    (dest/'README.md').write_text('# 固定39视野离线审核\n\n先读RESULT.md，再打开review.html。\n\n```powershell\n& \'E:\\anaconda\\envs\\dl-3lc\\python.exe\' -I -B .\\tools\\bus_crop_pilot.py verify-run\n& \'E:\\anaconda\\envs\\dl-3lc\\python.exe\' -I -B .\\tools\\bus_crop_pilot.py summarize <项目内显式反馈JSON路径>\n```\n\n页面自动本地保存、可显式导出/导入JSON，不写标签。原标签仅参考，默认≥.25、弱候选单独开关。类别四类均保存，bus未检出不可由其他车辆框替代。反馈中的目标组号由人工归组；没填组号不自动推断目标去重或局部收益。\n',encoding='utf-8')
    print(json.dumps({'status':status,'counts':counts(payload),'original_files_verified':17668},ensure_ascii=False,indent=2))
    if not success: raise RuntimeError('Single inference attempt stopped; inspect retained evidence, no retry')


def render(p):
    run=directory(p)
    if (run/'review.html').exists(): raise FileExistsError('Review page exists; do not overwrite')
    predictions=p.read(run/'predictions.json');samples=p.read(p.OUT/'sample_manifest.json');views=p.read(p.OUT/'crop_manifest.json')
    validate_predictions(p,predictions,views)
    data=dict(predictions)
    data['samples']=[dict(s,image_url=os.path.relpath(p.ROOT/s['image_path'],run).replace('\\','/'),
                          views=[dict(v,input_url=os.path.relpath(p.ROOT/s['image_path'] if v['condition']=='A' else p.OUT/v['input_file'],run).replace('\\','/')) for v in views if v['sample_id']==s['id']]) for s in samples]
    text=(p.ROOT/'tools/bus_crop_review.template.html').read_text(encoding='utf-8')
    encoded=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    if text.count('__PILOT_DATA__')!=1: raise ValueError('Invalid template placeholder')
    with (run/'review.html').open('x',encoding='utf-8') as f: f.write(text.replace('__PILOT_DATA__',encoded))


def feedback_identity(p,record,candidate_id):
    c=next((c for c in record['candidates'] if c['id']==candidate_id),None)
    if c is None and candidate_id is not None: raise ValueError('Unknown candidate')
    if c is None and any(c['cls']==3 and c['band']=='high' for c in record['candidates']):
        # A manual target might be missed despite some bus boxes: null remains allowed with a target group.
        pass
    return dict(sample_id=record['sample_id'],view_id=record['id'],condition=record['condition'],candidate_id=candidate_id,
                input_rgb_sha256=record['input_rgb_sha256'],weight_sha256=p.WEIGHT_HASH)


def validate_feedback(p,payload,predictions):
    if payload.get('schema')!='bus-crop-feedback-v1' or payload.get('experiment_id')!=predictions['experiment_id'] or payload.get('weight_sha256')!=p.WEIGHT_HASH:
        raise ValueError('Feedback batch/weight mismatch')
    allowed={r['id']:r for r in predictions['records']};seen=set();records=payload.get('records')
    if not isinstance(records,list): raise ValueError('Invalid feedback records')
    for item in records:
        if not isinstance(item,dict) or item.get('view_id') not in allowed: raise ValueError('Unknown feedback view')
        record=allowed[item['view_id']];identity=feedback_identity(p,record,item.get('candidate_id'))
        if any(item.get(k)!=v for k,v in identity.items()): raise ValueError('Feedback source mismatch')
        key=(item['view_id'],item['candidate_id'],item.get('target_group',''))
        # One judgment per candidate; null entries may describe distinct manually grouped missed targets.
        if key in seen: raise ValueError('Duplicate feedback')
        if item['candidate_id'] is not None and any(k[:2]==key[:2] for k in seen): raise ValueError('Duplicate candidate feedback')
        seen.add(key)
        if item.get('status') not in STATUSES or not isinstance(item.get('note',''),str) or len(item.get('note',''))>2000: raise ValueError('Invalid status/note')
        group=item.get('target_group','')
        if not isinstance(group,str) or len(group)>80: raise ValueError('Invalid target group')
        if item['candidate_id'] is None and item['status'] not in ('not_detected','outside_or_truncated','uncertain'): raise ValueError('No-box judgment requires appropriate status')
        if item['candidate_id'] is not None and item['status']=='not_detected': raise ValueError('A detection cannot be not_detected')
    return records


def summarize(p,payload,predictions):
    records=validate_feedback(p,payload,predictions);by_id={r['id']:r for r in predictions['records']}
    result={'reviewed_entries':len(records),'unreviewed_is_not_negative':True,'by_condition':{},'grouped_local_gain_examples':[]}
    for subset,frames in [('all13',set(p.FRAMES)),('suspected11',set(p.FRAMES[1:-1]))]:
        result['by_condition'][subset]={c:{b:{s:0 for s in STATUSES} for b in ('high','weak','no_candidate')} for c in 'ABC'}
        for item in records:
            r=by_id[item['view_id']]
            if r['frame'] not in frames: continue
            cand=next((c for c in r['candidates'] if c['id']==item['candidate_id']),None)
            result['by_condition'][subset][r['condition']][cand['band'] if cand else 'no_candidate'][item['status']]+=1
    grouped=collections.defaultdict(list)
    for item in records:
        if item.get('target_group'): grouped[(item['sample_id'],item['target_group'])].append(item)
    for (sample,group),items in grouped.items():
        a_missed=any(i['condition']=='A' and i['status']=='not_detected' for i in items)
        for band in ('high','weak'):
            local=[i for i in items if i['condition'] in ('B','C') and i['status']=='usable' and any(c['id']==i['candidate_id'] and c['band']==band for c in by_id[i['view_id']]['candidates'])]
            # For weak comparisons, require an explicit A weak-band target judgment, not a default high-threshold miss.
            if a_missed and local and band=='high':
                result['grouped_local_gain_examples'].append(dict(sample_id=sample,target_group=group,band=band,conditions=sorted({i['condition'] for i in local})))
    result['distinct_manually_grouped_targets']=len(grouped)
    result['limits']='Entries are candidate judgments, not truth objects. Empty groups are not deduplicated; no AP/P/R, no total missing-label rate. Only explicit A misses + high-band usable local judgments form examples; weak-band gains are not inferred.'
    return result


def verify_run(p):
    p.verify_preparation();run=directory(p);v=p.read(run/'verification.json');cfg=p.read(run/'run_config.json')
    if cfg['source_sha256']!=sources(p) or v['source_sha256']!=sources(p): raise ValueError('Run source drift')
    for name,h in v['artifact_sha256'].items():
        if p.sha(run/name)!=h: raise ValueError('Run artifact changed: '+name)
    if cfg['prepared_artifact_sha256']!=p.artifact_hashes(): raise ValueError('Prepared input drift')
    status=p.read(run/'run_status.json')
    if status['state']!='completed_awaiting_human_review' or status['model_inputs_completed']!=39 or status['inference_exit_code']!=0: raise ValueError('Run not completed')
    payload=p.read(run/'predictions.json');validate_predictions(p,payload,p.read(p.OUT/'crop_manifest.json'))
    if counts(payload)!=p.read(run/'candidate_counts.json'): raise ValueError('Count drift')
    if v['integrity_after']!=p.read(p.OUT/'plan_config.json')['integrity_before'] or p.protected()!=v['protected_after']: raise ValueError('Integrity drift')
    worker_result=p.read(run/'worker_result.json')
    if worker_result['completed']!=39 or worker_result['three_lc_imported'] or worker_result['guard_events']: raise ValueError('Worker boundary violation')
    print(json.dumps({'verified':True,'model_inputs':39,'counts':counts(payload),'not_reinferred':True},indent=2))
