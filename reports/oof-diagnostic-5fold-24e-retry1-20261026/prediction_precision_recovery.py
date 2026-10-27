"""Versioned process-local precision API adapter; never edits frozen sources or dependencies."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import uuid

ROOT=Path(r'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge')
REPORT=ROOT/'reports/oof-diagnostic-5fold-24e-retry1-20261026'
SOURCE=ROOT/'tools/oof_diagnostic.py'
SELF=Path(__file__).resolve()
OLD='        effective={k:getattr(model.predictor.args,k) for k in c[\'prediction\']}'
NEW='        effective=precision_effective(model.predictor,c,torch)'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def patched_prediction_source(text):
    tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predict_worker')
    part='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    if part.count(OLD)!=1:raise ValueError('Unexpected frozen prediction implementation')
    return part.replace(OLD,NEW)

def precision_effective(predictor,c,torch):
    args=predictor.args;backend=predictor.model
    if c['prediction']['half'] is not False:raise ValueError('Only approved FP32 request supported')
    if not hasattr(args,'quantize') or args.quantize is not None:raise ValueError('Native precision configuration is not FP32')
    if not hasattr(backend,'fp16') or backend.fp16 is not False:raise ValueError('Actual backend is not FP32')
    effective={k:(backend.fp16 if k=='half' else getattr(args,k)) for k in c['prediction']}
    if not getattr(predictor,'_oof_precision_audited',False):
        parameters=list(backend.model.parameters())
        if not parameters or any(p.is_floating_point() and p.dtype!=torch.float32 for p in parameters):raise ValueError('Prediction weights are not FP32')
        audit=Path(c['stage_root'])/'precision_adapter_runtime.json'
        with audit.open('x',encoding='utf-8') as f:
            json.dump(dict(adapter_sha256=c['precision_adapter_sha256'],requested_half=False,native_quantize=args.quantize,backend_fp16=backend.fp16,floating_parameter_dtypes=sorted({str(p.dtype) for p in parameters if p.is_floating_point()}),existing_preprocess_enforces_float32=True),f,indent=2)
        predictor._oof_precision_audited=True
    return effective

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def worker():
    # Read adapter/source before importing native module, whose bootstrap installs the guard.
    own_hash=digest(SELF);code=patched_prediction_source(SOURCE.read_text(encoding='utf-8'))
    m=load(SOURCE,'oof_precision_worker_native')
    c=m.BOOT_CONFIG
    if not c or c['role']!='predict' or c['precision_adapter_sha256']!=own_hash:raise RuntimeError('Private prediction adapter identity mismatch')
    m.precision_effective=precision_effective
    exec(compile(code,str(SOURCE)+' [precision API adapter]','exec'),m.__dict__)
    m.worker()

def self_test():
    text=SOURCE.read_text(encoding='utf-8');part=patched_prediction_source(text)
    assert NEW in part and OLD not in part
    try:patched_prediction_source(text.replace(OLD,'        effective={}'))
    except ValueError:pass
    else:raise AssertionError('Must reject changed source')
    # Execute the installed configuration mapping, without importing torch/Ultralytics or constructing a model.
    cfgpath=Path(r'E:\anaconda\envs\dl-3lc\Lib\site-packages\ultralytics\cfg\__init__.py')
    tree=ast.parse(cfgpath.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and any(isinstance(x,ast.Constant) and x.value=='half' for x in ast.walk(n)) and n.name.startswith('_handle_deprecation'))
    env={'deprecation_warn':lambda *a:None};exec(compile(ast.Module(body=[node],type_ignores=[]),str(cfgpath),'exec'),env)
    handle=env[node.name]
    assert handle({'half':False})['quantize'] is None
    assert handle({'half':True})['quantize']==16
    assert handle({'half':False,'quantize':8})['quantize']==8
    args=types.SimpleNamespace(quantize=None,conf=.001)
    p=types.SimpleNamespace(args=args,model=types.SimpleNamespace(fp16=False),_oof_precision_audited=True)
    c={'prediction':{'half':False,'conf':.001}}
    assert precision_effective(p,c,None)=={'half':False,'conf':.001}
    for obj,attr,bad in ((args,'quantize',16),(p.model,'fp16',True)):
        old=getattr(obj,attr);setattr(obj,attr,bad)
        try:precision_effective(p,c,None)
        except ValueError:pass
        else:raise AssertionError('Precision drift admitted')
        setattr(obj,attr,old)
    del args.conf
    try:precision_effective(p,c,None)
    except AttributeError:pass
    else:raise AssertionError('Unrelated missing field silently admitted')
    assert not any(x.split('.')[0] in ('torch','ultralytics','tlc') for x in sys.modules)
    print('10 precision adapter/native mapping checks passed; no model or data inference executed.',flush=True)

def continuation():
    old=load(REPORT/'continue_completed_fold.py','oof_prior_continuation');m=old.m
    adapter_hash=digest(SELF);native=m.launch
    def launch(role,stage,*a,**kw):
        if role!='predict':return native(role,stage,*a,**kw)
        stage=Path(stage)
        if (stage/'started.json').exists():
            status=m.read(stage/'status.json');result=m.read(stage/'worker_result.json')
            if status['exit_code']!=1 or result['state']!='failed' or result['guard_events'] or "has no attribute 'half'" not in result['exception']:raise RuntimeError('Unexpected prior prediction failure; preserve and investigate')
            output=stage.parent/'outer_predictions.jsonl'
            if not output.is_file() or output.stat().st_size!=0:raise RuntimeError('Unexpected partial prediction output; no automatic overwrite')
            history=stage.parent/('prediction-api-failure-'+uuid.uuid4().hex);history.mkdir()
            hashes={p.relative_to(stage).as_posix():m.sha(p) for p in stage.rglob('*') if p.is_file()}
            stage.rename(history/'predict_stage');output.rename(history/'outer_predictions.jsonl')
            m.write(history/'preservation.json',dict(created_at=m.now(),stage_sha256=hashes,prediction_bytes=0,empty_output_sha256=m.sha(history/'outer_predictions.jsonl'),best_model_retrained=False,recovery='Only missing legacy half field; original FP32 settings retained',adapter_sha256=adapter_hash))
        original_popen=m.subprocess.Popen
        def popen(command,*args,**kwargs):
            expected=[str(m.PYTHON),'-I','-B','-u',str(SOURCE),'_worker']
            if command!=expected:raise RuntimeError('Unexpected child command under precision adapter')
            return original_popen([str(m.PYTHON),'-I','-B','-u',str(SELF),'_worker'],*args,**kwargs)
        m.subprocess.Popen=popen
        try:return native(role,stage,*a,precision_adapter_sha256=adapter_hash,**kw)
        finally:m.subprocess.Popen=original_popen
    m.launch=launch
    audit=REPORT/'runtime_reconciliation'/('precision-adapter-'+uuid.uuid4().hex+'.json')
    m.write(audit,dict(created_at=m.now(),adapter=str(SELF),adapter_sha256=adapter_hash,frozen_executor_sha256=digest(SOURCE),dependency_evidence={str(p):digest(p) for p in [m.PKG/'cfg/__init__.py',m.PKG/'engine/predictor.py',m.PKG/'nn/autobackend.py']},change='Process-local effective half from actual backend.fp16; enforce quantize=None, FP32 weights and existing FP32 tensor/pixel checks',reuse_completed_fold01=True,training_parameters_changed=False,deadline_reset=False))
    old.main()

if __name__=='__main__':
    if os.environ.get('OOF_CONFIG'):worker()
    elif '--self-test' in sys.argv:self_test()
    elif '--continue' in sys.argv:continuation()
    else:raise SystemExit('Use --self-test or --continue; worker is private')
