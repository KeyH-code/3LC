"""Reuse frozen metric assembly while disclosing the approved interrupted D080 recovery."""
from pathlib import Path
import json
import types

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/train-cleaning-expansion-resume-r1-20261102'


def disclosure():
    return ('D080 resumed from the interrupted epoch-4 last.pt to 28 total epochs; D070/D060 trained from scratch. '
        'D080 resume=true is an explicitly approved protocol deviation: RNG/DataLoader streams and FP16 serialized '
        'checkpoint state need not match uninterrupted training. Its initial-state hash describes original provenance, '
        'NOT a newly randomized or uninterrupted recovery run. Twelve uniform FP32 evaluations; prior A/B reused. '
        'Technical inspection/tool retries are preserved separately; no claim that only three training processes were attempted.')


def main():
    p=ROOT/'tools/summarize_cleaning_expansion.py';m=types.ModuleType('frozen_summary');m.__file__=str(p)
    exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);m.OUT=OUT
    comparison=m.main()
    report=OUT/'RESULT.md';text=report.read_text(encoding='utf-8')
    text=text.replace('Three from-scratch 28-epoch YOLOv8n runs; twelve uniform FP32 evaluations. Prior A/B results reused, not rerun.',disclosure())
    text=text.replace('no network/3LC or automatic retry.','no network/3LC; technical retry amendment preserved in retry_amendment.json.')
    text+='\n## Recovery provenance and limitations\n\n'+disclosure()+'\n'
    text+='\nD080 epoch 1-4 traces and historical checkpoint hashes are retained with the original interrupted attempt; epochs 5-28 are the resumed segment. Recovery elapsed times exclude prior interrupted training; see interrupted_attempt.json.\n'
    report.write_text(text,encoding='utf-8')
    path=OUT/'comparison.json';value=json.loads(path.read_text(encoding='utf-8'))
    value['recovery_protocol']=dict(disclosure=disclosure(),source='reports/train-cleaning-expansion-ss04-20261101',
        D080_resume=True,interrupted_completed_epochs=4,additional_epochs=24,uninterrupted_equivalence=False)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return comparison


if __name__=='__main__':main()
