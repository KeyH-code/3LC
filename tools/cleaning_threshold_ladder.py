"""Read-only saved-OOF candidate size ladder; no inference, review or label edits."""
import argparse
import collections
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import types

ROOT=Path(__file__).resolve().parents[1]
OOF=ROOT/'reports/oof-diagnostic-5fold-24e-retry1-20261026'
CLEAN=ROOT/'reports/oof-label-cleaning-conf085-20261027'
OUT=ROOT/'reports/cleaning-threshold-ladder-20261029'
THRESHOLDS=[.95,.9,.85,.8,.75,.7,.65,.6,.55,.5,.45,.4,.35,.3,.25]
EXTRA='prediction_without_label_coverage'
CONFLICT='class_disagreement'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def count(rows):
    if any(not math.isfinite(r['confidence']) or not 0<=r['confidence']<=1 for r in rows):
        raise ValueError('Invalid confidence')
    return dict(signals=len(rows),prediction_boxes=len({(r['id'],r['prediction_id']) for r in rows}),
                images=len({r['id'] for r in rows}),
                uncovered_signals=sum(r['signal']==EXTRA for r in rows),
                class_conflict_signals=sum(r['signal']==CONFLICT for r in rows),
                by_predicted_class={name:sum(r['predicted_class']==i for r in rows) for i,name in enumerate(['truck','car','van','bus'])})


def ladder(rows,thresholds,reviewed,reviewed_images):
    result=[];above=None;base_images={r['id'] for r in rows if r['confidence']>=.85}
    for threshold in sorted(set(thresholds),reverse=True):
        selected=[r for r in rows if r['confidence']>=threshold]
        band=[r for r in selected if above is None or r['confidence']<above]
        unseen=[r for r in selected if r['key'] not in reviewed]
        result.append(dict(threshold=threshold,total=count(selected),band=count(band),
                           band_upper_exclusive=above,below085=count([r for r in selected if r['confidence']<.85]),
                           unreviewed=count(unseen),never_reviewed_images=len({r['id'] for r in selected}-reviewed_images),
                           new_images_vs085=len({r['id'] for r in selected}-base_images),
                           reviewed_decisions=dict(collections.Counter(reviewed[r['key']] for r in selected if r['key'] in reviewed))))
        above=threshold
    return result


def main(out=OUT):
    if out.exists():raise FileExistsError('Preserve prior ladder; output directory already exists')
    helper_path=ROOT/'tools/oof_label_cleaning.py'
    helper=types.ModuleType('frozen_label_signals');helper.__file__=str(helper_path)
    exec(compile(helper_path.read_bytes(),str(helper_path),'exec'),helper.__dict__)
    fixed=[OOF/'source_manifest.json',OOF/'oof_predictions.jsonl',CLEAN/'dataset_manifest.json',
           CLEAN/'candidates.json',CLEAN/'review.json',CLEAN/'applied_changes.json',
           ROOT/'reports/oof-label-diagnostic-20261026/summary.json',helper_path,
           Path(__file__).resolve(),ROOT/'tests/test_cleaning_threshold_ladder.py']
    before={p.relative_to(ROOT).as_posix():sha(p) for p in fixed}
    if sha(CLEAN/'dataset_manifest.json')!='1898d635ab9808c00cab3ccc6dcbf30a564820393b91f260775204711de2c76e':
        raise ValueError('Approved cleaned manifest drift')
    source=load(OOF/'source_manifest.json');source=[s for s in source if s['split']=='train']
    by_id={s['id']:s for s in source};derived={s['id']:s for s in load(CLEAN/'dataset_manifest.json')['rows']}
    if len(by_id)!=7852 or set(derived)!=set(by_id):raise ValueError('Train membership drift')
    old_candidates=load(CLEAN/'candidates.json')['rows'];old_keys={r['key'] for r in old_candidates}
    reviewed={r['key']:r['decision'] for r in load(CLEAN/'review.json')['rows'] if r['visual_completed']}
    if set(reviewed)!=old_keys or len(old_keys)!=744:raise ValueError('Prior 744 review identities drift')
    reviewed_images={r['id'] for r in old_candidates}
    seen=set();original_rows=[];clean_rows=[];pred_counts={t:0 for t in THRESHOLDS};label_pins={}
    fields=['key','id','prediction_id','confidence','signal','predicted_class','gt_index']
    with (OOF/'oof_predictions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            rec=json.loads(line);ident=rec['id']
            if ident in seen or ident not in by_id:raise ValueError('OOF duplicate/unknown image')
            seen.add(ident);src=by_id[ident];dr=derived[ident]
            if any(rec[k]!=src[k] for k in ['scene','frame','image_path','label_path','image_sha256','label_sha256']):
                raise ValueError('OOF/source identity mismatch')
            original_label=ROOT/src['label_path'];clean_label=CLEAN/dr['label']
            for path,expected in [(original_label,src['label_sha256']),(clean_label,dr['label_sha256'])]:
                if sha(path)!=expected:raise ValueError('Label source changed: '+str(path))
                label_pins[path.relative_to(ROOT).as_posix()]=expected
            clean_src=src.copy();clean_src['label_sha256']=dr['label_sha256']
            clean_src['labels']=[line.split() for line in clean_label.read_text(encoding='utf-8').splitlines() if line.strip()]
            # Each conflict selects the highest-confidence qualifying prediction per GT.
            # Eligibility geometry is per prediction; filtering this minimum-threshold
            # result is equivalent to rerunning at any higher reporting threshold.
            for dst,sample in [(original_rows,src),(clean_rows,clean_src)]:
                dst.extend({k:r[k] for k in fields if k in r} for r in helper.signals(sample,rec,min(THRESHOLDS)))
            for t in THRESHOLDS:pred_counts[t]+=sum(p['confidence']>=t for p in rec['candidates'])
    if seen!=set(by_id):raise ValueError('Incomplete OOF coverage')
    if {r['key'] for r in original_rows if r['confidence']>=.85}!=old_keys:
        raise ValueError('0.85 candidate identities fail exact historical anchor')
    old_summary=load(ROOT/'reports/oof-label-diagnostic-20261026/summary.json')['thresholds']
    for t in [.25,.5,.7,.9]:
        rr=[r for r in original_rows if r['confidence']>=t];anchor=old_summary[str(t)]
        for signal,field in [(EXTRA,'extra_separated'),(CONFLICT,'class_conflict')]:
            chosen=[r for r in rr if r['signal']==signal]
            if [sum(r['predicted_class']==i for r in chosen) for i in range(4)]!=anchor[field]:
                raise ValueError('Historical anchor count drift')
            if len({r['id'] for r in chosen})!=anchor['image_counts'][field]:raise ValueError('Historical image anchor drift')
        if pred_counts[t]!=sum(anchor['predictions']):raise ValueError('Raw prediction anchor drift')
    for name,h in {**before,**label_pins}.items():
        if sha(ROOT/name)!=h:raise ValueError('Input mutated during counting: '+name)
    original_ladder=ladder(original_rows,THRESHOLDS,reviewed,reviewed_images)
    clean_ladder=ladder(clean_rows,THRESHOLDS,reviewed,reviewed_images)
    out.mkdir(parents=True)
    for name,data in [('original',original_ladder),('current_clean',clean_ladder)]:
        save(out/(name+'_ladder.json'),data)
        with (out/(name+'_ladder.csv')).open('x',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f);w.writerow(['threshold','all_prediction_boxes','candidate_signals','unique_candidate_prediction_boxes','candidate_images',
                'uncovered_signals','class_conflict_signals','below085_signals','below085_images','unreviewed_signals','unreviewed_images','never_reviewed_images',
                'band_signals','band_images','truck_signals','car_signals','van_signals','bus_signals'])
            for r in data:
                total=r['total'];w.writerow([r['threshold'],pred_counts[r['threshold']],total['signals'],total['prediction_boxes'],total['images'],
                    total['uncovered_signals'],total['class_conflict_signals'],r['below085']['signals'],r['below085']['images'],
                    r['unreviewed']['signals'],r['unreviewed']['images'],r['never_reviewed_images'],r['band']['signals'],r['band']['images']]+list(total['by_predicted_class'].values()))
    save(out/'candidate_records.json',dict(original=original_rows,current_clean=clean_rows))
    save(out/'verification.json',dict(at=datetime.now(timezone.utc).isoformat(),train_images=7852,
         thresholds=THRESHOLDS,reporting_grid_only_not_selected_cleaning_threshold=True,previous_review_signals=744,
         previous_review_images=len(reviewed_images),previous_decisions=dict(collections.Counter(reviewed.values())),
         exact085_candidate_identity_anchor=True,historical_count_image_class_anchors=[.25,.5,.7,.9],
         fixed_inputs_sha256=before,label_files_verified_unchanged=len(label_pins),label_manifest_sha256=hashlib.sha256(json.dumps(label_pins,sort_keys=True).encode()).hexdigest(),
         model_execution=False,label_writes=0,new_reviews=0,source_predictions='Original-label-trained frozen OOF, not newly cleaned-model inference',
         reviewed_identity_note='Current-clean previously-reviewed identity match is provenance only; GT indices remain original lines plus approved additions, not a new review decision',
         limitations='Candidates are model-label disagreements, not confirmed errors. Counts exclude unmatched GT and moderate geometric disagreements. Images overlap across signals/classes/bands; do not sum image columns.'))
    lines=['# 后续清洗候选阈值阶梯（只读规模统计）','',
           '本表不选择下一阈值，不授权改标签/审核/训练。阈值为统计展示网格；固定旧OOF不重新推理。',
           '“候选信号”沿用原规则：无覆盖预测(max IoU<.1且预测面积被GT覆盖比例<.2)，或异类IoU>=.7且同类最大IoU<.3、每个GT取最高置信度预测。',
           '同一预测可能关联多个GT；信号数与去重预测框数分列。图片数按image id去重，不得跨档相加。',
           '当前清洗口径仅把相同旧预测与已批准清洗标签重新比较；不是B模型新OOF。此前defer/keep不因降阈值自动变为可改。','']
    for title,data in [('原始标签口径：复现旧候选池',original_ladder),('当前清洗标签口径：本轮后残余候选线索',clean_ladder)]:
        lines += ['## '+title,'','| conf≥ | 候选信号 | 独立预测框 | 图片 | 无覆盖线索 | 异类冲突线索 | <.85新增信号 | <.85涉及图片 | 未审核信号 | 未审核涉及图片 |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in data:
            t=r['total'];lines.append(f"| {r['threshold']:.2f} | {t['signals']} | {t['prediction_boxes']} | {t['images']} | {t['uncovered_signals']} | {t['class_conflict_signals']} | {r['below085']['signals']} | {r['below085']['images']} | {r['unreviewed']['signals']} | {r['unreviewed']['images']} |")
        lines+=['']
    lines+=['## 核验与复现','',
            '- .85档候选key逐项复现此前744项；.25/.50/.70/.90各信号类型、类别、图片数和原始预测框数与旧诊断锚点一致。',
            '- 原始/清洗15704标签文件统计前后SHA256不变；来源JSON/OOF/工具哈希不变；未修改val或标签。',
            '- JSON含每档累计、相邻置信区间增量、类别分布、旧审核决策与新图片计数；CSV便于选择规模。',
            '- 无覆盖/异类冲突只是线索；不是已确认漏标/错标。真实修改量须视觉审核及单独批准，不能由候选数推定。',
            '- 不包括全部可能标签问题，不把缺少预测的GT当作应删除目标。','',
            '```powershell',"& 'E:\\anaconda\\envs\\dl-3lc\\python.exe' -I -B -m unittest discover -s tests -p test_cleaning_threshold_ladder.py",
            "& 'E:\\anaconda\\envs\\dl-3lc\\python.exe' -I -B tools/cleaning_threshold_ladder.py --out reports/cleaning-threshold-ladder-20261029-recheck",'```','',
            '复现必须使用新输出目录，不覆盖当前证据。问题解决来源：AI 直接解决（机械统计）；下一阈值/修改范围由学习者决定。']
    (out/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('LADDER WRITTEN',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();target=args.out.resolve()
    if not target.is_relative_to(ROOT):raise ValueError('Output must remain in this project')
    main(target)
