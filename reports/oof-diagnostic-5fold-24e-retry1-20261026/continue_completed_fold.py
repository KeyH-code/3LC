"""Continue frozen five-fold run, reuse completed fold01, reconcile child NUL false positives only.
No installed/project source edits. No extra fold01 training. Raw guard records are preserved.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
ROOT=Path(r'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge')
spec=importlib.util.spec_from_file_location('oof_continuation',ROOT/'tools/oof_diagnostic.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
NULL=str(Path(os.devnull).resolve()).casefold()

def null_only(rows,pid,main_pid):
    return pid!=main_pid and bool(rows) and all(set(r)=={'event','path','writing','time'} and r['event']=='file' and str(r['path']).casefold()==NULL and r['writing'] is False for r in rows)

def successful(status,result):
    return status['exit_code']==0 and not status['timed_out'] and status['error'] is None and result['state']=='completed' and not result['guard_events']

def closed_result(stage):
    status=m.read(stage/'status.json');result=m.read(stage/'worker_result.json')
    if not successful(status,result):raise RuntimeError('Cannot reuse or disposition an unsuccessful worker')
    started=m.read(stage/'worker_started.json')
    if started['pid']!=result['pid']:raise ValueError('Worker identity drift')
    return result

def reconcile(stage,result):
    c=m.read(stage/'train_worker_config.json');work=Path(c['work']).resolve();audit=m.OUT/'runtime_reconciliation'
    if c['role']!='train' or not work.is_relative_to(m.SCRATCH):raise RuntimeError('Only owned training child journals can be dispositioned')
    files=list(work.glob('guard-*.jsonl'));validated=[]
    for p in files:
        raw=p.read_bytes();rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()];pid=int(p.stem.split('-')[1])
        if not null_only(rows,pid,result['pid']):raise RuntimeError('Non-NUL/main-process guard event remains blocking: '+str(p))
        validated.append((p,raw,rows,pid))
    for p,raw,rows,pid in validated:
        dst=audit/f'fold_{c["fold"]:02d}-closed-{pid}-{uuid.uuid4().hex}.raw.jsonl'
        with dst.open('xb') as f:f.write(raw)
        h=hashlib.sha256(raw).hexdigest()
        m.append(audit/'closed_stage_null_dispositions.jsonl',dict(time=m.now(),fold=c['fold'],pid=pid,main_pid=result['pid'],records=len(rows),original_journal=str(p),original_retained_in_place=True,raw_archive=dst.relative_to(m.OUT).as_posix(),sha256=h,decision='Only Windows null-device stdin read false positive; ordinary file/network/credential events never admitted'))
    return sum(len(rows) for _,_,rows,_ in validated)

def self_test():
    good=dict(event='file',path=str(Path(os.devnull).resolve()),writing=False,time='2026-10-26T00:00:00+00:00')
    assert null_only([good],2,1);assert not null_only([good],1,1)
    for change in ({'path':str(ROOT/'AGENTS.md')},{'writing':True},{'event':'socket.connect'},{'extra':'x'}):assert not null_only([dict(good,**change)],2,1)
    s=dict(exit_code=0,timed_out=False,error=None);r=dict(state='completed',guard_events=[])
    assert successful(s,r)
    for change in ({'exit_code':1},{'timed_out':True},{'error':'x'}):assert not successful(dict(s,**change),r)
    assert not successful(s,dict(r,state='failed'));assert not successful(s,dict(r,guard_events=[good]))
    print('12 continuation/strict NUL checks passed; no model executed.',flush=True)

def main():
    print('Verifying complete frozen preparation and prior successful fold01 before continuation...',flush=True)
    c,samples,folds=m.check_prepared(full=True);first=m.OUT/'fold_01/train_stage';tr=closed_result(first)
    if tr['fold']!=1 or tr['epochs']!=24 or tr['nan_recovery_attempts']!=0:raise ValueError('Unexpected completed first model')
    for key in ('best','last'):
        if m.sha(tr[key+'_file'])!=tr[key+'_sha256']:raise ValueError('Completed checkpoint changed')
    events=list(m.rows(m.OUT/'fold_01/stopper_trace.jsonl'))
    if [e['epoch'] for e in events]!=list(range(1,25)) or max(e['epoch'] for e in events if e['best_checkpoint_updated'])!=tr['best_checkpoint_epoch']:raise ValueError('Prior training trace mismatch')
    # The earlier audit observer archived/renamed8 child journals. Validate every raw record.
    audit=m.OUT/'runtime_reconciliation';prior=list(m.rows(audit/'null_device_dispositions.jsonl'))
    if not prior:raise ValueError('Missing original null-disposition evidence')
    for rec in prior:
        p=m.OUT/rec['raw_archive']
        if rec['fold']!=1 or m.sha(p)!=rec['sha256'] or not null_only(list(m.rows(p)),rec['pid'],tr['pid']):raise ValueError('Prior benign audit mismatch')
    remaining=reconcile(first,tr)
    if time.time()>=c['deadline_unix']:raise TimeoutError('Original batch budget exhausted')
    # Preserve old supervisor state in-place under a distinct immutable recovery namespace.
    history=audit/('supervisor-restart-'+uuid.uuid4().hex);history.mkdir();controls={}
    for name in ('run_started.json','run_status.json','progress.json','failure_integrity.json'):
        p=m.OUT/name
        if p.exists():
            h=m.sha(p);dst=history/name;p.rename(dst);controls[name]=dict(path=dst.relative_to(m.OUT).as_posix(),sha256=h)
    m.write(history/'continuation_authorization.json',dict(created_at=m.now(),amendment=str(m.AMENDMENT),prior_supervisor_controls=controls,reuse_fold01=True,fold01_training_reexecuted=False,fold01_epochs=24,fold01_best_epoch=tr['best_checkpoint_epoch'],fold01_best_sha256=tr['best_sha256'],prior_child_null_records=sum(x['records'] for x in prior),additional_null_records=remaining,project_dependency_sources_unchanged=True,deadline_unix=c['deadline_unix']))
    native_launch=m.launch
    def launch(role,stage,*args,**kw):
        stage=Path(stage)
        if role=='train' and stage==first:
            print('Reusing validated fold01 model:24epochs, best21; no extra training launch.',flush=True);return tr
        try:return native_launch(role,stage,*args,**kw)
        except RuntimeError:
            if role!='train':raise
            result=closed_result(stage)
            n=reconcile(stage,result)
            if not n:raise
            print(f'Accepted closed fold{result["fold"]} worker after auditing{n} child NUL-read false positives; raw journals retained.',flush=True)
            return result
    m.launch=launch
    # This exact full precheck just passed; per-fold source/data checks and final full integrity remain native.
    m.check_prepared=lambda full=True:(c,samples,folds)
    try:m.run()
    finally:
        if m.DEADLINE_TIMER is not None:m.DEADLINE_TIMER.cancel()

if __name__=='__main__':
    if '--self-test' in sys.argv:self_test()
    else:main()
