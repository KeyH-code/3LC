"""CPU-only evidence of successful continuation, not complete-five-fold acceptance."""
import importlib.util
from pathlib import Path
import sys
ROOT=Path(r'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge')
spec=importlib.util.spec_from_file_location('oof_launch_verify',ROOT/'tools/oof_diagnostic.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.read(m.OUT/'plan_config.json');samples=m.read(m.OUT/'source_manifest.json');folds=m.read(m.OUT/'fold_manifest.json')
if m.source_hashes()!=c['source_sha256'] or m.dependency_hashes()!=c['dependency_sha256'] or m.cfg()!=c['configuration']:raise ValueError('Frozen implementation/config/dependency drift')
m.validate_splits(samples,folds)
tr,records=m.validate_fold_output(folds[0],samples,c['configuration']['prediction'],c['configuration']['training'])
pstage=m.OUT/'fold_01/predict_stage';pr=m.read(pstage/'worker_result.json');status=m.read(pstage/'status.json');pc=m.read(pstage/'predict_worker_config.json')
if pr['guard_events'] or pr['state']!='completed' or status['exit_code']!=0 or status['timed_out'] or status['error'] is not None:raise ValueError('First outer prediction not cleanly completed')
if list(Path(pc['work']).glob('guard-*.jsonl')):raise ValueError('Raw prediction guard journal present')
adapter=m.OUT/'prediction_precision_recovery_v2.py'
if pc['precision_adapter_sha256']!=m.sha(adapter):raise ValueError('Adapter identity drift')
precision=m.read(pstage/'precision_adapter_runtime.json')
if precision['native_quantize'] is not None or precision['backend_fp16'] is not False or precision['floating_parameter_dtypes']!=['torch.float32']:raise ValueError('Actual precision drift')
old=list((m.OUT/'fold_01').glob('prediction-self-source-failure-*'))
if len(old)!=1:raise ValueError('Missing unique previous failure evidence')
preserved=m.read(old[0]/'preservation.json');m.verify_hashes(preserved['stage_sha256'],old[0]/'predict_stage')
if m.sha(old[0]/'outer_predictions.jsonl')!=preserved['predictions_sha256'] or preserved['accepted_as_completed']:raise ValueError('Previous failure archive changed')
if m.sha(old[0]/'raw_guard_journal.jsonl')!=preserved['raw_guard_sha256']:raise ValueError('Raw guard archive changed')
second=m.OUT/'fold_02';runtime=m.read(second/'training_runtime.json');m.assert_effective(runtime['effective_args'],c['configuration']['training'])
if runtime['optimizer']!='AdamW' or not runtime['amp_enabled'] or runtime['external_amp_check_executed']:raise ValueError('Second-fold actual training drift')
trace=list(m.rows(second/'stopper_trace.jsonl')) if (second/'stopper_trace.jsonl').exists() else []
if trace and [r['epoch'] for r in trace]!=list(range(1,len(trace)+1)):raise ValueError('Second-fold epoch trace gaps')
if any(x.split('.')[0] in ('torch','ultralytics','tlc') for x in sys.modules):raise RuntimeError('Unexpected model package import')
result=dict(checked_at=m.now(),continuation_started_and_verified=True,oof_complete=False,first_fold_epochs=tr['epochs'],first_fold_best_epoch=tr['best_checkpoint_epoch'],first_fold_model_sha256=tr['best_sha256'],first_fold_outer_records=len(records),first_fold_prediction_guard_events=pr['guard_events'],first_fold_prediction_exit_code=status['exit_code'],first_fold_fp32_verified=True,first_fold_retraining_performed=False,previous_attempt_preserved=True,previous_and_successful_stream_sha256_equal=preserved['predictions_sha256']==pr['predictions_sha256'],second_fold_training_runtime_verified=True,second_fold_completed_epochs_observed=len(trace),source_config_dependency_unchanged=True,deadline_unix=c['deadline_unix'],deadline_reset=False,model_reexecuted_by_this_check=False,training_background_task='e80d3567-63a1-4366-be29-238b509d3e92',readonly_postcheck_task='bea6469f-885a-49ab-91f3-fdbd4123caac',precision_adapter_sha256=m.sha(adapter),state_at_check=m.read(m.OUT/'progress.json'),limits='Running continuation only; remaining folds, full7852 verification and real browser acceptance not yet complete.')
if '--record' in sys.argv:m.write(m.OUT/'continuation_launch_verified.json',result)
print(m.canonical(result))
