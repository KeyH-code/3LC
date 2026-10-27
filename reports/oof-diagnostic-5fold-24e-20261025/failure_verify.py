"""Read-only failure acceptance; --seal adds exclusive evidence once. Never calls a model."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
ROOT=Path(r'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge')
spec=importlib.util.spec_from_file_location('oof_failure_readonly',ROOT/'tools/oof_diagnostic.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=m.OUT

def check():
    started=time.perf_counter()
    print('Rechecking approved config, code/dependencies, deterministic folds,62850 prepared artifacts and17668 originals...',flush=True)
    c,samples,folds=m.check_prepared(full=True)
    if m.protected_files()!=c['protected_sha256']:raise ValueError('Protected historical file set/hash drift')
    run=m.read(out/'run_status.json');progress=m.read(out/'progress.json');status=m.read(out/'fold_01/train_stage/status.json');worker=m.read(out/'fold_01/train_stage/worker_result.json')
    if run!=progress or run['state']!='failed_preserved_no_retry' or run['training_attempts']!=1 or run['completed_folds'] or run['outer_images']!=0:raise ValueError('Failure count/state drift')
    if status['exit_code']!=1 or status['timed_out'] or status['error'] is not None or worker['state']!='failed' or worker['guard_events']:raise ValueError('Worker exit evidence drift')
    train_marks=list(out.glob('fold_*/train_stage/started.json'));pred_marks=list(out.glob('fold_*/predict_stage/started.json'))
    if len(train_marks)!=1 or pred_marks:raise ValueError('Unexpected extra model-stage attempt')
    forbidden=list(out.glob('fold_*/training/weights/*.pt'))+list(out.glob('fold_*/outer_predictions.jsonl'))+list(out.glob('fold_*/stopper_trace.jsonl'))+list(out.glob('fold_*/training_runtime.json'))+list(out.glob('fold_*/training/results.csv'))
    if forbidden or (out/'oof_predictions.jsonl').exists():raise ValueError('Unexpected model/training/prediction output')
    if not m.read(out/'failure_integrity.json')['originals_and_protected_unchanged']:raise ValueError('Parent failure integrity did not pass')
    m.verify_hashes(c['source_sha256'],out/'failed-startup-sources')
    if m.read(out/'failed-startup-sources/source_sha256.json')!=c['source_sha256']:raise ValueError('Failure source archive identity mismatch')
    requested=c['architecture'];fallback=re.sub(r'(\d+)([nslmx])(.+)?$',r'\1\3',requested)
    if '\\dl-3lc\\' not in requested or '\\dl-3c\\' not in fallback or fallback not in worker['exception'] or 'FileNotFoundError' not in worker['exception']:raise ValueError('Path-parser failure evidence mismatch')
    if any(n in sys.modules for n in ('torch','ultralytics','tlc','tlc_ultralytics','tlcconfig')):raise ValueError('Unexpected model/3LC import during failure verification')
    result=dict(checked_at=m.now(),state='failed_preserved_no_retry',oof_completed=False,training_worker_attempts=1,completed_folds=0,completed_epochs=0,completed_models=0,outer_prediction_attempts=0,oof_images=0,worker_exit_code=1,worker_timeout=False,worker_process_seconds=status['process_elapsed_seconds'],worker_elapsed_seconds=worker['elapsed_seconds'],guard_events=worker['guard_events'],original_files_verified=17668,prepared_artifacts_verified=len(m.read(out/'preparation_verified.json')['artifact_sha256']),independent_derived_files=62816,protected_files_verified=len(c['protected_sha256']),deterministic_split_recomputed=True,outer_manifest_union_unique_images=len({i for f in folds for i in f['outer']}),source_dependency_and_frozen_archive_hashes_matched=True,model_reexecuted=False,training_retried=False,source_modified_after_preparation=False,approved_architecture_request=requested,pure_string_native_regex_fallback=fallback,wrong_environment_files_read_by_this_verifier=False,verification_seconds=time.perf_counter()-started,limits='Failure preservation acceptance only; no complete OOF verification, models, AP, real OOF browser or human label-quality result.')
    if '--seal' in sys.argv:
        m.write(out/'failure_verification.json',result)
        hashes=dict(m.read(out/'preparation_verified.json')['artifact_sha256'])
        for p in out.rglob('*'):
            if p.is_file():
                key=p.relative_to(out).as_posix()
                if key not in hashes:hashes[key]=m.sha(p)
        m.write(out/'final_checks.json',dict(state='failed_preserved_no_retry',oof_completed=False,checked_at=m.now(),artifact_sha256=hashes,acceptance=result,readonly_command="E:\\anaconda\\envs\\dl-3lc\\python.exe -I -B .\\reports\\oof-diagnostic-5fold-24e-20261025\\failure_verify.py --check-only"))
    elif '--check-only' in sys.argv:
        if (out/'final_checks.json').exists():m.verify_hashes(m.read(out/'final_checks.json')['artifact_sha256'],out)
    else:raise ValueError('Explicit --check-only or --seal required')
    print(json.dumps(result,ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':check()
