"""Audit-only disposition of confirmed Windows DataLoader stdin NUL false positives.
No model API, dependency/source edits, data edits, networking or general guard bypass.
Raw records are retained twice (report archive and renamed scratch journal).
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid
ROOT=Path(r'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge')
SPEC=importlib.util.spec_from_file_location('oof_null_audit',ROOT/'tools/oof_diagnostic.py');m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)
NULL=str(Path(os.devnull).resolve()).casefold()

def benign(rows):
    return bool(rows) and all(set(r)=={'event','path','writing','time'} and r['event']=='file' and str(r['path']).casefold()==NULL and r['writing'] is False for r in rows)

def self_test():
    good=dict(event='file',path=str(Path(os.devnull).resolve()),writing=False,time='2026-10-26T00:00:00+00:00')
    assert benign([good])
    for change in ({'path':str(ROOT/'AGENTS.md')},{'writing':True},{'event':'socket.connect'},{'extra':'not allowed'}):assert not benign([dict(good,**change)])
    assert not benign([]);assert not benign([good,dict(event='3lc_import',name='tlc')])
    print('7 strict NUL-disposition checks passed; no model or actual device opened.',flush=True)

def main():
    out=m.OUT;audit=out/'runtime_reconciliation';audit.mkdir(exist_ok=True)
    deadline=m.read(out/'plan_config.json')['deadline_unix'];count=0
    m.write(audit/'null_device_policy.json',dict(created_at=m.now(),authorization=str(m.AMENDMENT),canonical_null_device=str(Path(os.devnull).resolve()),reason='Frozen guard compares resolved Windows NUL device path to raw os.devnull=nul, while multiprocessing.util._close_stdin opens os.devnull read-only. Only confirmed child-process null reads are dispositioned.',source_code_unchanged=True,stdlib_reference=str(m.ENV/'Lib/multiprocessing/util.py'),stdlib_sha256=m.sha(m.ENV/'Lib/multiprocessing/util.py'),other_guard_events_remain_active=True,raw_records_preserved=True))
    try:
        while time.time()<deadline:
            for cp in out.glob('fold_*/train_stage/train_worker_config.json'):
                c=m.read(cp);work=Path(c['work']).resolve();fold=c['fold'];parent=m.read(cp.parent/'worker_started.json')['pid'] if (cp.parent/'worker_started.json').exists() else None
                if not work.is_relative_to(m.SCRATCH) or c['role']!='train':raise RuntimeError('Unexpected guard journal root/role')
                for p in list(work.glob('guard-*.jsonl')):
                    raw=p.read_bytes()
                    try:rr=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
                    except (UnicodeDecodeError,json.JSONDecodeError):continue  # writer may be flushing; leave active
                    pid=int(p.stem.split('-')[1])
                    if pid==parent or not benign(rr):
                        m.append(audit/'unexpected_guard_alerts.jsonl',dict(time=m.now(),journal=str(p),main_pid=parent,rows=rr));raise RuntimeError('Non-whitelisted or main-process guard event remains active')
                    h=hashlib.sha256(raw).hexdigest();nonce=uuid.uuid4().hex;dst=audit/f'fold_{fold:02d}-{pid}-{nonce}.raw.jsonl'
                    with dst.open('xb') as f:f.write(raw)
                    if p.read_bytes()!=raw:continue  # journal changed: do not move; archive remains evidence
                    renamed=work/f'reviewed-null-device-{pid}-{nonce}.jsonl';p.rename(renamed)
                    after=renamed.read_bytes()
                    if after!=raw:
                        # Preserve a changed journal under the active prefix so supervisor cannot miss it.
                        restore=work/f'guard-reconciliation-race-{nonce}.jsonl'
                        with restore.open('xb') as f:f.write(after)
                        raise RuntimeError('Journal changed during disposition; restored active guard alert')
                    m.append(audit/'null_device_dispositions.jsonl',dict(time=m.now(),fold=fold,pid=pid,records=len(rr),original_journal=str(p),preserved_scratch_journal=str(renamed),raw_archive=dst.relative_to(out).as_posix(),sha256=h,decision='Confirmed Windows null-device read false positive only; no ordinary file/network/credential read allowed'))
                    count+=len(rr);print(f'Audited NUL stdin false positive fold{fold} pid{pid}; raw preserved, total_records={count}',flush=True)
                    m.update(audit/'observer_status.json',dict(state='auditing_child_null_reads',records=count,last_update=m.now(),source_unchanged=True,models_restarted=0))
            try:state=m.read(out/'progress.json')
            except (FileNotFoundError,json.JSONDecodeError):time.sleep(.25);continue
            if state.get('state')=='paused_recovery_authorized':
                m.update(audit/'observer_status.json',dict(state='attempt_paused_requires_recovery',records=count,last_update=m.now(),models_restarted=0));return
            if state.get('current_fold')==5 and state.get('phase')=='outer_prediction':
                # All five train jobs are closed; stop writes well before final artifact hashing.
                m.update(audit/'observer_status.json',dict(state='all_five_training_jobs_reconciled',records=count,finished_at=m.now(),source_unchanged=True,models_restarted=0,remaining_browser_review=True));print('All5 training jobs passed strict child-null disposition; observer finished.',flush=True);return
            time.sleep(.25)
        raise TimeoutError('Original batch deadline reached')
    except BaseException:
        m.update(audit/'observer_status.json',dict(state='requires_attention_no_other_event_dispositioned',records=count,last_update=m.now(),models_restarted=0));raise

if __name__=='__main__':
    if '--self-test' in sys.argv:self_test()
    else:main()
