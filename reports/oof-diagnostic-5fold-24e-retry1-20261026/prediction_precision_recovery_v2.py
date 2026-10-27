"""Precision adapter v2: explicitly permit read-only introspection of exact versioned worker sources.
Preserves the complete previous prediction attempt; never ignores any guard event.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import uuid
ROOT=Path(r'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge')
REPORT=ROOT/'reports/oof-diagnostic-5fold-24e-retry1-20261026'
SELF=Path(__file__).resolve()
V1=REPORT/'prediction_precision_recovery.py'
V1_HASH='eb05052939e74cf61f81d2dbdcf86cb88a4dae59dadf3a319ac2f095be10c5c9'
if hashlib.sha256(V1.read_bytes()).hexdigest()!=V1_HASH:raise RuntimeError('Version1 adapter changed')
spec=importlib.util.spec_from_file_location('oof_precision_adapter_v1',V1);a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
a.SELF=SELF
def journal_matches(events,result,parse_time):
    normalized=[{k:v for k,v in e.items() if k!='time'} for e in events]
    prior=result['guard_events'];n=len(prior)
    if not prior or len(events)<n or normalized[:n]!=prior:return False
    if not all(set(e)=={'event','path','writing','time'} and e['event']=='file' and Path(e['path']).resolve()==V1 and e['writing'] is False for e in events):return False
    return all(parse_time(e['time'])>=parse_time(result['finished_at']) for e in events[n:])


def preserve_self_introspection_failure(m):
    stage=REPORT/'fold_01/predict_stage';output=stage.parent/'outer_predictions.jsonl'
    if not (stage/'started.json').exists():raise RuntimeError('Expected exactly the prior self-introspection failure')
    status=m.read(stage/'status.json');r=m.read(stage/'worker_result.json');config=m.read(stage/'predict_worker_config.json')
    if status['exit_code']!=1 or status['timed_out'] or status['error'] is not None or r['state']!='failed' or r.get('images')!=1595:raise RuntimeError('Unexpected previous attempt; preserve without retry')
    if not r['guard_events'] or not all(set(e)=={'event','path','writing'} and e['event']=='file' and Path(e['path']).resolve()==V1 and e['writing'] is False for e in r['guard_events']):raise RuntimeError('Unrelated guard event remains blocking')
    if r['exception'].splitlines()[-1]!='RuntimeError: Isolation event':raise RuntimeError('Unexpected exception')
    work=Path(config['work']).resolve()
    if not work.is_relative_to(m.SCRATCH):raise RuntimeError('Unowned work directory')
    journals=list(work.glob('guard-*.jsonl'))
    if len(journals)!=1 or journals[0].name!=f'guard-{r["pid"]}.jsonl':raise RuntimeError('Unexpected child journal')
    events=list(m.rows(journals[0]))
    if not journal_matches(events,r,m.datetime.fromisoformat):raise RuntimeError('Guard journal/result mismatch or unrelated/post-result event')
    tr=m.read(stage.parent/'train_stage/worker_result.json')
    if m.sha(tr['best_file'])!=tr['best_sha256'] or r['weight_sha256']!=tr['best_sha256'] or m.sha(output)!=r['predictions_sha256']:raise RuntimeError('Prior prediction or model integrity drift')
    records=list(m.rows(output))
    if [x['id'] for x in records]!=[s['id'] for s in config['samples']]:raise RuntimeError('Prior stream incomplete')
    for record,sample in zip(records,config['samples']):m.validate_prediction(record,sample,1,tr['best_sha256'],config['prediction'])
    history=stage.parent/('prediction-self-source-failure-'+uuid.uuid4().hex);history.mkdir()
    hashes={p.relative_to(stage).as_posix():m.sha(p) for p in stage.rglob('*') if p.is_file()}
    stage.rename(history/'predict_stage');output.rename(history/'outer_predictions.jsonl')
    raw=journals[0].read_bytes()
    with (history/'raw_guard_journal.jsonl').open('xb') as f:f.write(raw)
    m.write(history/'preservation.json',dict(created_at=m.now(),stage_sha256=hashes,prediction_records=1595,predictions_sha256=r['predictions_sha256'],original_main_guard_events=r['guard_events'],raw_guard_records=len(events),result_guard_records=len(r['guard_events']),raw_guard_sha256=hashlib.sha256(raw).hexdigest(),raw_scratch_retained=True,accepted_as_completed=False,retry_scope='Only prediction, unchanged frozen best; exact read allowlist for v1/v2 worker sources',model_retrained=False,adapter_v1_sha256=V1_HASH,adapter_v2_sha256=hashlib.sha256(SELF.read_bytes()).hexdigest()))
    print(f'Preserved1595 prior records plus all{len(events)} denied self-source reads; retrying prediction with exact source allowlist, no guard bypass.',flush=True)
if __name__=='__main__':
    if os.environ.get('OOF_CONFIG'):a.worker()
    elif '--self-test' in sys.argv:
        a.self_test()
        assert SELF!=V1 and SELF.parent==V1.parent
        from datetime import datetime
        event=dict(event='file',path=str(V1),writing=False,time='2026-10-26T08:00:00+00:00')
        result=dict(guard_events=[{k:v for k,v in event.items() if k!='time'}],finished_at='2026-10-26T08:00:01+00:00')
        extra=dict(event,time='2026-10-26T08:00:02+00:00')
        assert journal_matches([event],result,datetime.fromisoformat)
        assert journal_matches([event,extra],result,datetime.fromisoformat)
        assert not journal_matches([],result,datetime.fromisoformat)
        assert not journal_matches([event,event],result,datetime.fromisoformat)
        for change in ({'path':str(ROOT/'AGENTS.md')},{'writing':True},{'event':'socket.connect'},{'unexpected':1}):
            assert not journal_matches([event,dict(extra,**change)],result,datetime.fromisoformat)
        print('8 strict raw/result journal-prefix and post-result self-read checks passed.',flush=True)
        print('Versioned adapter identity and two-file exact allowlist verified.',flush=True)
    elif '--continue' in sys.argv:
        original_load=a.load
        def load(path,name):
            module=original_load(path,name)
            if Path(path)==REPORT/'continue_completed_fold.py':
                m=module.m;preserve_self_introspection_failure(m);native=m.launch
                def permitted_launch(role,stage,*args,**kw):
                    if role=='predict':kw['read_files']=[*kw.get('read_files',[]),V1,SELF]
                    return native(role,stage,*args,**kw)
                m.launch=permitted_launch
            return module
        a.load=load;a.continuation()
    else:raise SystemExit('Use --self-test or --continue; worker is private')
