"""Build/verify a read-only confidence browser from the one completed crop run. No model imports."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/bus-crop-expansion-retry1-20261025'
OUT=ROOT/'reports/bus-crop-confidence-review-20261025'
LEVELS=[.25,.40,.50,.60,.70,.80,.90,.95,.99]
GROUPS=['MVI_40181','MVI_40851','MVI_39401','MVI_40761','MVI_63521']
TEMPLATE=ROOT/'tools/bus_confidence_review.template.html'
JS=ROOT/'tools/bus_confidence_review.js'
TEST=ROOT/'tests/test_bus_confidence_review.py'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()

def extract_data(text,script_id='expansion-data'):
    pattern=r'<script id="'+re.escape(script_id)+r'" type="application/json">(.*?)</script>'
    matches=re.findall(pattern,text,re.S)
    if len(matches)!=1:raise ValueError('Exactly one embedded data script required')
    return json.loads(matches[0])

def summarize(data,threshold):
    if not isinstance(threshold,(int,float)) or isinstance(threshold,bool) or not math.isfinite(threshold) or not .25<=threshold<=1:raise ValueError('Display threshold must be finite in [.25,1]')
    groups=[]
    for group in data['groups']:
        samples=[s for s in data['samples'] if s['scene']==group['id']]
        kept=[c for s in samples for v in s['views'] for c in v['bus_candidates'] if c['confidence']>=threshold]
        has=lambda s:any(c['confidence']>=threshold for v in s['views'] for c in v['bus_candidates'])
        groups.append(dict(scene=group['id'],images=len(samples),boxes=len(kept),images_with_candidates=sum(has(s) for s in samples),
                           lead_images=sum(s['bus_boxes']==0 and has(s) for s in samples),edge_boxes=sum(c['touches_crop_edge'] for c in kept)))
    total={k:sum(g[k] for g in groups) for k in ('boxes','images_with_candidates','lead_images','edge_boxes')}
    return dict(threshold=threshold,groups=groups,total=total)

def validate_data(data,predictions,expected_images=492,expected_views=1239):
    if data['experiment_id']!=predictions['experiment_id']:raise ValueError('Batch identity mismatch')
    samples=data['samples'];views=[v for s in samples for v in s['views']];records=predictions['records']
    if len(samples)!=expected_images or len({s['id'] for s in samples})!=expected_images:raise ValueError('Image identity/count mismatch')
    if len(views)!=expected_views or len(records)!=expected_views:raise ValueError('View count mismatch')
    if len({v['id'] for v in views})!=expected_views:raise ValueError('Duplicate view identity')
    record_by_id={r['id']:r for r in records}
    if len(record_by_id)!=expected_views:raise ValueError('Duplicate prediction identity')
    seen=set()
    for s in samples:
        if s['split']!='train':raise ValueError('Train only')
        for view in s['views']:
            record=record_by_id.get(view['id'])
            if record is None:raise ValueError('Missing prediction view')
            for key in ('sample_id','scene','frame','tile','roi','scale','input_file','input_file_sha256','input_rgb_sha256'):
                if view[key]!=record[key]:raise ValueError('View source mismatch: '+key)
            if view['sample_id']!=s['id'] or view['scene']!=s['scene']:raise ValueError('Wrong sample ownership')
            expected=[c for c in record['candidates'] if c['cls']==3]
            actual=view['bus_candidates']
            if len(actual)!=len(expected):raise ValueError('Bus candidate count mismatch')
            for a,b in zip(actual,expected):
                if {k:v for k,v in a.items() if k!='max_original_bus_iou'}!=b:raise ValueError('Candidate record changed')
                if a['id'] in seen:raise ValueError('Duplicate candidate identity')
                seen.add(a['id'])
                if not math.isfinite(a['confidence']) or not .05<=a['confidence']<=1:raise ValueError('Invalid confidence')
                if not math.isfinite(a['max_original_bus_iou']) or not 0<=a['max_original_bus_iou']<=1:raise ValueError('Invalid label overlap')
    return len(seen)

def page_data(original):
    data=copy.deepcopy(original)
    data['confidence_levels']=LEVELS[:];data['source_report']=SOURCE.relative_to(ROOT).as_posix()
    for sample in data['samples']:
        for view in sample['views']:
            if view['input_url']!=view['input_file'] or not view['input_file'].startswith('inputs/') or '..' in Path(view['input_file']).parts:raise ValueError('Unexpected source input URL')
            view['input_url']='../'+SOURCE.name+'/'+view['input_file']
    return data

def source_files():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),TEMPLATE,JS,TEST)}

def check_source():
    final=read(SOURCE/'final_checks.json')
    for name,h in final['delivery_artifact_sha256'].items():
        if sha(SOURCE/name)!=h:raise ValueError('Sealed source artifact changed: '+name)
    status=read(SOURCE/'run_status.json')
    if status['inference_exit_code']!=0 or status['model_inputs_completed']!=1239 or status['timed_out']:raise ValueError('Incomplete source run')
    return final

def build():
    if OUT.exists():raise FileExistsError('Preserve existing confidence review')
    final=check_source();original=extract_data((SOURCE/'review.html').read_text(encoding='utf-8'))
    count=validate_data(original,read(SOURCE/'predictions.json'))
    if [g['id'] for g in original['groups']]!=GROUPS:raise ValueError('Fixed group scope changed')
    if count!=1967 or summarize(original,.25)['total']!={'boxes':1026,'images_with_candidates':394,'lead_images':104,'edge_boxes':528}:raise ValueError('Baseline changed')
    data=page_data(original);summaries=[summarize(data,t) for t in LEVELS]
    template=TEMPLATE.read_text(encoding='utf-8')
    if template.count('__CONFIDENCE_DATA__')!=1:raise ValueError('Template marker')
    html=template.replace('__CONFIDENCE_DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    OUT.mkdir()
    with (OUT/'index.html').open('x',encoding='utf-8') as f:f.write(html)
    with (OUT/'confidence.js').open('xb') as f:f.write(JS.read_bytes())
    write(OUT/'threshold_summaries.json',summaries)
    write(OUT/'verification.json',dict(created_at=now(),source_report=SOURCE.relative_to(ROOT).as_posix(),source_final_checks_sha256=sha(SOURCE/'final_checks.json'),
        sealed_source_files_verified=len(final['delivery_artifact_sha256']),source_sha256=source_files(),images=492,views=1239,bus_candidates=count,
        confidence_levels=LEVELS,inference_executed=False,labels_modified=False,training_executed=False,
        delivery_artifact_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps(dict(page=str(OUT/'index.html'),images=492,views=1239,bus_candidates=count,thresholds=[s['total']|{'threshold':s['threshold']} for s in summaries]),ensure_ascii=False,indent=2))

def verify():
    final=check_source();v=read(OUT/'verification.json')
    if sha(SOURCE/'final_checks.json')!=v['source_final_checks_sha256'] or source_files()!=v['source_sha256']:raise ValueError('Source drift')
    for name,h in v['delivery_artifact_sha256'].items():
        if sha(OUT/name)!=h:raise ValueError('Confidence artifact changed: '+name)
    original=extract_data((SOURCE/'review.html').read_text(encoding='utf-8'))
    validate_data(original,read(SOURCE/'predictions.json'))
    actual=extract_data((OUT/'index.html').read_text(encoding='utf-8'),'confidence-data')
    if actual!=page_data(original) or read(OUT/'threshold_summaries.json')!=[summarize(actual,t) for t in LEVELS]:raise ValueError('Display data/summary changed')
    if sha(OUT/'confidence.js')!=sha(JS):raise ValueError('JavaScript copy mismatch')
    print(json.dumps(dict(verified=True,source_artifacts_unchanged=len(final['delivery_artifact_sha256']),images=492,views=1239,inference_executed=False),indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['build','verify']);args=parser.parse_args()
    {'build':build,'verify':verify}[args.command]()
if __name__=='__main__':main()
