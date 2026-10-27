"""Finish only interrupted final read-only preparation verification; never rerun prepare/probe/models."""
import importlib.util
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(r'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge')
spec=importlib.util.spec_from_file_location('oof_completion',ROOT/'tools/oof_diagnostic.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=m.OUT

def main():
    if (out/'preparation_verified.json').exists() or (out/'run_started.json').exists():raise RuntimeError('No reentry or model execution allowed here')
    c=m.read(out/'plan_config.json');snap=m.read(out/'preflight_snapshot.json');m.arm_deadline(c['deadline_unix'])
    m.write(out/'preparation_finalization_started.json',dict(started_at=m.now(),purpose='Complete missing final read-only verification only; no prepare/probe/copy/model retry',host_event='Original foreground Bash reached600second host wait limit and was stopped after all5 copy stages; no live OOF Python remained',training_attempts=0,deadline_unix=c['deadline_unix']))
    try:
        for key in ('configuration','source_sha256','dependency_sha256','integrity_before','protected_sha256','git_before'):
            if c[key]!=snap[key]:raise ValueError('Preflight/config drift: '+key)
        if m.cfg()!=c['configuration'] or m.source_hashes()!=c['source_sha256'] or m.dependency_hashes()!=c['dependency_sha256']:raise ValueError('Frozen source/config/dependency drift')
        status=m.read(out/'preflight/status.json');probe=m.read(out/'preflight/worker_result.json')
        if status['exit_code']!=0 or status['timed_out'] or probe['state']!='completed' or probe['model_constructed'] or probe['guard_events']:raise ValueError('Preflight was not successful/model-free')
        if list(out.glob('fold_*/train_stage/started.json')) or list(out.glob('fold_*/predict_stage/started.json')):raise ValueError('Unexpected model attempt')
        samples=m.read(out/'source_manifest.json');folds=m.read(out/'fold_manifest.json');by={s['id']:s for s in samples}
        if samples!=m.inventory() or folds!=m.split_folds(samples):raise ValueError('Frozen fold/source mismatch')
        m.validate_splits(samples,folds)
        print('Source/config/probe and deterministic splits verified; checking62816 copied files...',flush=True)
        hashes={};checked=0
        import yaml
        for fold in folds:
            directory=out/f'fold_{fold["fold"]:02d}';data=directory/'data'
            expected_yaml=dict(path=str(data),train='fit/images',val='monitor/images',names=m.NAMES)
            if yaml.safe_load((directory/'data.yaml').read_text(encoding='utf-8'))!=expected_yaml:raise ValueError('Data YAML drift')
            for role in ('fit','monitor','outer'):
                ss=[by[i] for i in fold[role]]
                if m.read(directory/(role+'_manifest.json'))!=ss:raise ValueError('Role manifest drift')
                if role=='outer':continue
                for kind,folder in [('image','images'),('label','labels')]:
                    base=data/role/folder;expected={Path(s[kind+'_path']).name for s in ss}
                    if {p.name for p in base.iterdir()}!=expected:raise ValueError('Copied file set drift')
                    for s in ss:
                        original=ROOT/s[kind+'_path'];p=base/original.name;h=m.sha(p)
                        if h!=s[kind+'_sha256'] or p.stat().st_size!=original.stat().st_size or p.stat().st_ino==original.stat().st_ino or p.stat().st_nlink!=1:raise ValueError('Copy content/independence drift: '+str(p))
                        hashes[p.relative_to(out).as_posix()]=h;checked+=1
                        if checked%10000==0:print('Copied files verified',checked,'/62816',flush=True)
            print('Copy verification complete for fold',fold['fold'],flush=True)
        if checked!=62816:raise ValueError('Unexpected total copied files')
        print('Verifying17668 originals and protected historical files...',flush=True)
        integrity=m.originals_integrity()
        if integrity!=c['integrity_before']:raise ValueError('Original data changed')
        m.verify_hashes(c['protected_sha256'])
        if m.protected_files()!=c['protected_sha256']:raise ValueError('Protected historical file-set drift')
        if m.source_hashes()!=c['source_sha256'] or m.dependency_hashes()!=c['dependency_sha256']:raise ValueError('Source/dependency changed during finalization')
        if any(n in sys.modules for n in ('torch','ultralytics','tlc','tlc_ultralytics','tlcconfig')):raise ValueError('Unexpected model/3LC import')
        m.time_left(c,1)
        m.write(out/'preparation_host_wait_reconciliation.json',dict(verified_at=m.now(),host_wait_seconds=600,preparation_command_reexecuted=False,probe_reexecuted=False,copies_recreated=False,model_training_attempts=0,model_prediction_attempts=0,original_integrity=integrity,derived_files_byte_identical_independent=checked,protected_files=len(c['protected_sha256']),source_and_dependency_hashes_unchanged=True,deadline_unix=c['deadline_unix'],note='Only the missing final hash freeze was completed from intact initial outputs within the same approved6hour budget. Original timeout and all evidence preserved.'))
        for p in out.rglob('*'):
            if p.is_file():
                key=p.relative_to(out).as_posix()
                if key not in hashes:hashes[key]=m.sha(p)
        m.write(out/'preparation_verified.json',dict(created_at=m.now(),artifact_sha256=hashes,images=7852,folds=5,training_attempts=0,reconciliation_file='preparation_host_wait_reconciliation.json'))
        print('PREPARATION_VERIFIED',len(hashes),'artifacts; no model execution.',flush=True)
    except BaseException:
        m.write(out/'preparation_finalization_failure.json',dict(failed_at=m.now(),exception=traceback.format_exc(),retry=False));raise
    finally:
        if m.DEADLINE_TIMER is not None:m.DEADLINE_TIMER.cancel()

if __name__=='__main__':main()
