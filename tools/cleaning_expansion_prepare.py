"""Freeze approved three-band review candidates only. No inference, apply or training."""
import argparse
import collections
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import types

ROOT=Path(__file__).resolve().parents[1]
OOF=ROOT/'reports/oof-diagnostic-5fold-24e-retry1-20261026'
CLEAN=ROOT/'reports/oof-label-cleaning-conf085-20261027'
LADDER=ROOT/'reports/cleaning-threshold-ladder-20261029'
OUT=ROOT/'reports/cleaning-expansion-080-070-060-20261029'
PYTHON=Path(r'E:\anaconda\envs\dl-3lc\python.exe')
BANDS={'D080':(.8,.85),'D070':(.7,.8),'D060':(.6,.7)}
EXPECTED={'D080':(1197,937),'D070':(2543,1855),'D060':(2448,1853)}


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def module(path):
    m=types.ModuleType(Path(path).stem);m.__file__=str(path)
    exec(compile(Path(path).read_bytes(),str(path),'exec'),m.__dict__);return m


def band(confidence):
    if not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Invalid confidence')
    return next((name for name,(lo,hi) in BANDS.items() if lo<=confidence<hi),None)


def partition(rows):
    result={name:[] for name in BANDS};seen=set()
    for row in rows:
        if row['key'] in seen:raise ValueError('Duplicate candidate identity')
        seen.add(row['key']);name=band(row['confidence'])
        if name:result[name].append(row)
    for rows in result.values():
        rows.sort(key=lambda r:(r['scene'],r['frame'],r['id'],r['signal'],r['prediction_id'],r.get('gt_index',-1)))
    return result


def target_reference(source,lines,original,index):
    if type(index) is not int or not 0<=index<len(lines):raise ValueError('Target outside D085 label')
    text=lines[index];tokens=text.split()
    if len(tokens)!=5:raise ValueError('Invalid target line')
    if index>=len(original):status='approved_added_box'
    else:
        old=list(map(float,original[index]));new=list(map(float,tokens))
        if old[1:]!=new[1:]:raise ValueError('Original geometry unexpectedly changed')
        status='approved_class_correction' if old[0]!=new[0] else 'unchanged_original'
    stable=hashlib.sha256(json.dumps([source['id'],source['label_sha256'],index,text],separators=(',',':')).encode()).hexdigest()
    return dict(stable_target_id=stable,reference_label_sha256=source['label_sha256'],
                d085_gt_index=index,original_label_line=text,d085_target_status=status)


def unreviewed(key):
    return dict(key=key,decision='unreviewed',source=None,proposed_class=None,geometry_ready=False,
                final_xyxy=None,object_key=None,reason='',observation='',uncertainty='',evidence=[],
                visual_completed=False,review_model=None,actual_model=None,reviewer_role=None)


def batches(grouped,size=48):
    if size<=0:raise ValueError('Invalid batch size')
    result=[]
    for name,rows in grouped.items():
        for i in range(0,len(rows),size):
            result.append(dict(batch_id=f'{name}-{i//size+1:03}',band=name,
                               candidate_keys=[r['key'] for r in rows[i:i+size]]))
    return result


def preparation_sources(plan):
    paths=[plan,Path(__file__).resolve(),ROOT/'tests/test_cleaning_expansion_prepare.py',
           ROOT/'tools/oof_label_cleaning.py',ROOT/'tools/oof_label_cleaning_review.template.html',
           ROOT/'tools/train_cleaning_ab.py',ROOT/'tools/eval_cleaning_ab_recovery.py',
           OOF/'source_manifest.json',OOF/'oof_predictions.jsonl',CLEAN/'dataset_manifest.json',
           CLEAN/'review.json',CLEAN/'candidates.json',CLEAN/'approval_record.json',CLEAN/'applied_changes.json',
           CLEAN/'phase-b-independent-verification.json',LADDER/'candidate_records.json',LADDER/'independent-verification.json',
           ROOT/'reports/eval-cleaning-ab-recovery-20261029/comparison.json']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def prepare():
    if OUT.exists():raise FileExistsError('Existing expansion evidence cannot be overwritten')
    plans=list((ROOT/'.pi/plan').glob('*080-070-060*20261029.md'))
    if len(plans)!=1:raise ValueError('Expected one approved plan')
    pins=preparation_sources(plans[0])
    if sha(CLEAN/'dataset_manifest.json')!='1898d635ab9808c00cab3ccc6dcbf30a564820393b91f260775204711de2c76e':
        raise ValueError('D085 approved manifest drift')
    expected_sha=load(LADDER/'independent-verification.json')['report_sha256']['candidate_records.json']
    if sha(LADDER/'candidate_records.json')!=expected_sha:raise ValueError('Ladder candidate source drift')
    approved=load(CLEAN/'approval_record.json')
    if not approved['approved_by_user'] or approved['proposal_sha256']!='0f7aebb2d0dd68c8adc6f79f8f2bb4564356f5ed4754e06385e49bf9f72259e6':
        raise ValueError('D085 application provenance mismatch')
    protected=load(CLEAN/'input_integrity.json')['protected'].copy()
    original={s['id']:s for s in load(OOF/'source_manifest.json') if s['split']=='train'}
    derived={s['id']:s for s in load(CLEAN/'dataset_manifest.json')['rows']}
    if len(original)!=7852 or set(original)!=set(derived):raise ValueError('Source membership drift')
    helper=module(ROOT/'tools/oof_label_cleaning.py')
    label_lines={};samples={};bytes_total=0
    for ident,row in derived.items():
        source=original[ident]
        if row['image_sha256']!=source['image_sha256']:raise ValueError('D085 image identity changed')
        image=CLEAN/row['image'];label=CLEAN/row['label']
        for p,kind in [(image,'image'),(label,'label')]:
            protected[p.relative_to(ROOT).as_posix()]=row[kind+'_sha256'];bytes_total+=p.stat().st_size
        lines=label.read_text(encoding='utf-8').splitlines()
        if not all(line.strip() for line in lines):raise ValueError('Blank line invalidates stable row reference')
        label_lines[ident]=lines
        sample=source.copy();sample.update(image_path=image.relative_to(ROOT).as_posix(),label_path=label.relative_to(ROOT).as_posix(),
                                          label_sha256=row['label_sha256'],labels=[line.split() for line in lines])
        samples[ident]=sample
    for arm in ['A','B']:
        result_path=ROOT/f'reports/train-cleaning-ab-ss04-20261029/{arm}/train/result.json'
        r=load(result_path);pins[result_path.relative_to(ROOT).as_posix()]=sha(result_path)
        for kind in ['best','last']:
            path=Path(r[kind]);protected[path.relative_to(ROOT).as_posix()]=r[kind+'_sha256']
    for name,expected in {**protected,**pins}.items():
        if sha(ROOT/name)!=expected:raise ValueError('Protected source drift: '+name)
    thin=load(LADDER/'candidate_records.json')['current_clean']
    expected_by_key={r['key']:r for r in thin if band(r['confidence'])}
    candidates=[];seen=set()
    with (OOF/'oof_predictions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            rec=json.loads(line);ident=rec['id']
            if ident not in original or ident in seen:raise ValueError('OOF duplicate or unknown image')
            seen.add(ident);old=original[ident]
            if any(rec[k]!=old[k] for k in ['scene','frame','image_path','label_path','image_sha256','label_sha256']):
                raise ValueError('OOF/source identity drift')
            for row in helper.signals(samples[ident],rec,.6):
                name=band(row['confidence'])
                if not name:continue
                thin_row={k:row[k] for k in expected_by_key[row['key']]}
                if thin_row!=expected_by_key[row['key']]:raise ValueError('Candidate rehydration differs from frozen ladder')
                row['band']=name;row['reference_dataset']='D085';row['oof_source_label_sha256']=old['label_sha256']
                if 'gt_index' in row:
                    row.update(target_reference(samples[ident],label_lines[ident],old['labels'],row['gt_index']))
                candidates.append(row)
    if seen!=set(original) or {r['key'] for r in candidates}!=set(expected_by_key):raise ValueError('Frozen candidate coverage mismatch')
    grouped=partition(candidates);ordered=[];summary={}
    for name,rows in grouped.items():
        if (len(rows),len({r['id'] for r in rows}))!=EXPECTED[name]:raise ValueError('Band count drift')
        for row in rows:row['number']=len(ordered)+1;ordered.append(row)
        summary[name]=dict(candidates=len(rows),images=len({r['id'] for r in rows}),scenes=len({r['scene'] for r in rows}),
                           signals=dict(collections.Counter(r['signal'] for r in rows)),
                           predicted_classes=dict(collections.Counter(r['predicted_name'] for r in rows)),
                           targets_previously_changed_in_D085=sum(r.get('d085_target_status') in ['approved_class_correction','approved_added_box'] for r in rows))
    if len(ordered)!=6188 or len({r['id'] for r in ordered})!=3360:raise ValueError('Expected 6188 signals/3360 images')
    for name,expected in {**protected,**pins}.items():
        if sha(ROOT/name)!=expected:raise ValueError('Inputs changed during freeze: '+name)
    OUT.mkdir(parents=True)
    save(OUT/'AUTHORIZATION.json',dict(at=now(),user_message='审阅通过，我磁盘空间已经经过了一次清理，现在空间足够，可以继续实验',
        source_reference='Current conversation: accepted three-band plan and asked to continue',approved_plan=str(plans[0]),plan_sha256=sha(plans[0]),
        phase='P0/P1 mechanical freeze and review preparation',thresholds=[.8,.7,.6],new_trainings_planned=3,
        preserve_stop_A=True,preserve_stop_B=True,label_application_authorized=False,training_started=False,
        visual_reviewer_model_confirmation_pending=True))
    save(OUT/'candidates.json',dict(schema='cleaning-expansion-candidates-v1',reference_dataset='D085',rows=ordered))
    save(OUT/'review.json',dict(schema='cleaning-expansion-review-v1',rows=[unreviewed(r['key']) for r in ordered]))
    save(OUT/'batches.json',dict(batch_size=48,note='Mechanical queue only; not delegation or completed review',rows=batches(grouped)))
    for name,rows in grouped.items():save(OUT/f'candidates/{name}.json',dict(rows=rows))
    save(OUT/'protected.json',dict(files=protected,sources=pins))
    training=module(ROOT/'tools/train_cleaning_ab.py').training_options()
    evaluation=module(ROOT/'tools/eval_cleaning_ab_recovery.py').evaluation_options()
    save(OUT/'experiment_spec.json',dict(at=now(),state='candidates_frozen_review_not_started',bands=BANDS,counts=summary,
        cumulative_versions=['D085','D080','D070','D060'],training=training,evaluation=evaluation,
        training_runs_planned=3,evaluations_planned=12,label_application_authorized=False,
        main_metric='val966 AP50-95',fixed_train_references=['D0','D085'],self_reference_per_model=True,
        frozen_candidates=6188,frozen_images=3360,new_visual_reviews=0,label_writes=0,model_inference_calls=0,
        disk_free_bytes=shutil.disk_usage(ROOT).free,d085_image_label_bytes=bytes_total,
        three_dataset_minimum_copy_bytes=bytes_total*3,
        storage_note='Dataset-only minimum, not full peak estimate; review evidence/caches/runtime/checkpoints and reserve must be budgeted before copying/training',
        review_roles='Pending confirmation for this new scope; no old model/parent/subagent authorization silently inherited'))
    helper.render(ROOT,OUT,ordered,[unreviewed(r['key']) for r in ordered])
    immutable=['AUTHORIZATION.json','candidates.json','batches.json','protected.json','experiment_spec.json']+[f'candidates/{name}.json' for name in BANDS]
    save(OUT/'freeze.json',dict(at=now(),files={n:sha(OUT/n) for n in immutable},review_initial_sha256=sha(OUT/'review.json'),
                              initial_index_sha256=sha(OUT/'index.html')))
    (OUT/'STATUS.md').write_text('# 三档清洗准备：候选已冻结，视觉审核未开始\n\n'
        '- 6188条新增候选，3360张图；D080 1197/937图、D070新增2543/1855图、D060新增2448/1853图。\n'
        '- 新视觉审核0；标签应用0；新训练/YOLO推理0。所有review为unreviewed，页面存在不等于已看图。\n'
        '- 累计的是未来通过审核的修订，不是自动采用预测框。旧>=.85保留/暂缓项目未纳入本轮。\n'
        '- 候选绑定当前D085标签及原OOF来源；原GT行内容/稳定ID/既有修改标记均保留，后续不能按漂移行号直接改标。\n'
        '- 冻结三档提案后汇报实际规模，学习者批准后才应用标签；三份版本验收后再确认训练执行。\n'
        '- 下一步确认本轮审核角色/实际模型/图像证据方式；不能沿用上轮限定授权冒充本轮已明确模型。\n'
        '- index.html为复用模板的准备页面，尚未作本轮浏览器交互/显示验收，不作真实视觉审核证据。\n'
        '- 只读验证命令：`E:\\anaconda\\envs\\dl-3lc\\python.exe -I -B tools/cleaning_expansion_prepare.py verify`。\n',encoding='utf-8')
    print('FROZEN 6188 candidates / 3360 images; visual reviews=0; label writes=0; training=0',flush=True)


def verify():
    freeze=load(OUT/'freeze.json')
    for name,h in freeze['files'].items():
        if sha(OUT/name)!=h:raise ValueError('Frozen artifact drift: '+name)
    info=load(OUT/'protected.json')
    for name,h in {**info['files'],**info['sources']}.items():
        if sha(ROOT/name)!=h:raise ValueError('Protected source drift: '+name)
    candidates=load(OUT/'candidates.json')['rows'];reviews=load(OUT/'review.json')['rows']
    if len(candidates)!=6188 or {r['key'] for r in candidates}!={r['key'] for r in reviews} or len(reviews)!=6188:
        raise ValueError('Review identity drift')
    if sha(OUT/'review.json')!=freeze['review_initial_sha256']:raise ValueError('Preparation verifier only handles initial unreviewed state')
    if any(r['visual_completed'] or r['decision']!='unreviewed' for r in reviews):raise ValueError('Unexpected completed reviews')
    if sha(OUT/'index.html')!=freeze['initial_index_sha256']:raise ValueError('Initial review page drift')
    bs=load(OUT/'batches.json')['rows'];flat=[k for b in bs for k in b['candidate_keys']]
    if len(flat)!=len(set(flat)) or set(flat)!={r['key'] for r in candidates}:raise ValueError('Batch coverage drift')
    result=dict(at=now(),verified=True,protected_files=len(info['files']),source_files=len(info['sources']),
                candidates=6188,images=len({r['id'] for r in candidates}),batches=len(bs),visual_reviews=0,
                label_writes=0,model_inference_calls=0,scope='Read-only preparation verification; no visual adjudication')
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','verify'])
    a=p.parse_args();prepare() if a.action=='prepare' else verify()
