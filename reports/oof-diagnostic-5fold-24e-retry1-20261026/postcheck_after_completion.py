"""Wait for complete native seal, then CPU-only verify; never invokes training/prediction."""
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
ROOT=Path(r'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge')
spec=importlib.util.spec_from_file_location('oof_postcheck',ROOT/'tools/oof_diagnostic.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=m.OUT;c=m.read(out/'plan_config.json');deadline=c['deadline_unix']
timer=threading.Timer(max(.001,deadline-time.time()),lambda:os._exit(124));timer.daemon=True;timer.start()
try:
    print('Waiting for sealed five-fold output. No model will be invoked; paused recovery states do not count as completion.',flush=True)
    while time.time()<deadline:
        try:
            state=m.read(out/'run_status.json')
            sealed=m.read(out/'verification.json') if (out/'verification.json').is_file() else None
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(20);continue
        if state['state']!='completed_awaiting_visual_review' or sealed is None:
            time.sleep(20);continue
        print('Native five-fold output sealed; independent read-only verify() starting.',flush=True)
        m.verify()
        result=dict(verified_at=m.now(),backend_readonly_verify_passed=True,folds=5,unique_oof_images=7852,model_reexecuted=False,browser_review_pending=True,human_label_quality_adjudication=False,limits='Only backend verification; real browser acceptance still pending. No labels changed or competition submission.')
        m.write(out/'post_run_verification.json',result)
        hashes=dict(sealed['artifact_sha256']);hashes['verification.json']=m.sha(out/'verification.json');hashes['post_run_verification.json']=m.sha(out/'post_run_verification.json')
        m.write(out/'final_checks.json',dict(state='backend_verified_browser_pending',checked_at=m.now(),acceptance=result,artifact_sha256=hashes))
        print(json.dumps(result,indent=2),flush=True);break
    else:raise TimeoutError('Original batch deadline reached before complete five-fold output')
finally:timer.cancel()
