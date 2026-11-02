"""Read completed paired artifacts only; never loads a model or changes a run."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/train-cleaning-ab-ss04-20261029'
KEYS=['metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)']


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def comparisons(rows):
    keys=list(rows['A']['val966']['native'])
    keys=[k for k in keys if k in KEYS]
    gaps={arm:{ref:{k:rows[arm][ref]['native'][k]-rows[arm]['val966']['native'][k]
                        for k in keys} for ref in ['A','B']} for arm in ['A','B']}
    return dict(val_B_minus_A={k:rows['B']['val966']['native'][k]-rows['A']['val966']['native'][k] for k in keys},
                gaps=gaps,gap_B_minus_A={ref:{k:gaps['B'][ref][k]-gaps['A'][ref][k] for k in keys} for ref in ['A','B']})


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not read(OUT/'completed.json')['protected_originals_unchanged']:
        raise RuntimeError('Completed, protected experiment required')
    rows={arm:{ref:read(OUT/arm/('eval_'+ref)/'result.json') for ref in ['A','B','val966']} for arm in ['A','B']}
    training={arm:read(OUT/arm/'train/result.json') for arm in ['A','B']}
    if training['A']['initial_state_sha256']!=training['B']['initial_state_sha256']:
        raise RuntimeError('Initial states differ')
    for arm in ['A','B']:
        if training[arm]['epochs']!=28:raise RuntimeError('Incomplete epochs')
        if sha(Path(training[arm]['best']))!=training[arm]['best_sha256']:raise RuntimeError('Weight drift')
        for row in rows[arm].values():
            if row['state']!='completed' or row['weight_sha256']!=training[arm]['best_sha256']:
                raise RuntimeError('Evaluation not tied to final best checkpoint')
    compare=comparisons(rows)
    with (OUT/'comparison.json').open('x',encoding='utf-8') as f:
        json.dump(dict(training=training,evaluations=rows,comparison=compare),f,ensure_ascii=False,indent=2,allow_nan=False)
    with (OUT/'metrics.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['model','label_reference','images']+KEYS+['fixed_P_macro_conf025','fixed_R_macro_conf025'])
        for arm in ['A','B']:
            for ref,row in rows[arm].items():
                writer.writerow([arm,ref,row['images']]+[row['native'][k] for k in KEYS]+[row['fixed']['precision_macro'],row['fixed']['recall_macro']])
    lines=['# SS04 本地完整A/B对照结果','',
           'A：原始训练标签；B：已批准清洗训练标签。两组均重新随机初始化训练28轮，普通Ultralytics入口，无3LC。',
           '共同val966用于选best及本次比较；不是额外独立test。历史SS04的val982分数不参与差值。','',
           '## 共同val966：原生指标','',
           '| 模型 | best epoch | AP50 | AP50–95 | P（最佳F1点） | R（最佳F1点） |',
           '|---|---:|---:|---:|---:|---:|']
    for arm in ['A','B']:
        r=rows[arm]['val966']['native'];lines.append(f"| {arm} | {training[arm]['best_epoch']} | {r[KEYS[2]]:.6f} | {r[KEYS[3]]:.6f} | {r[KEYS[0]]:.6f} | {r[KEYS[1]]:.6f} |")
    lines+=['', 'B−A（百分点）：'+ '；'.join(f'{k}: {v*100:+.3f}' for k,v in compare['val_B_minus_A'].items()),'',
            '## 六次统一评估与相同标签参照的差距','',
            '| 模型 | 标签参照 | AP50 | AP50–95 | P@conf.25/IoU.50（宏平均） | R@conf.25/IoU.50（宏平均） |',
            '|---|---|---:|---:|---:|---:|']
    for arm in ['A','B']:
        for ref,r in rows[arm].items():
            lines.append(f"| {arm} | {ref} | {r['native'][KEYS[2]]:.6f} | {r['native'][KEYS[3]]:.6f} | {r['fixed']['precision_macro']:.6f} | {r['fixed']['recall_macro']:.6f} |")
    lines+=['','差距定义：同一模型、指定train标签参照的指标减去共同val966指标。以下为AP50–95差距（百分点）。','',
            '| train标签参照 | A的差距 | B的差距 | 差距变化B−A |','|---|---:|---:|---:|']
    for ref in ['A','B']:
        k=KEYS[3];lines.append(f"| {ref} | {compare['gaps']['A'][ref][k]*100:+.3f} | {compare['gaps']['B'][ref][k]*100:+.3f} | {compare['gap_B_minus_A'][ref][k]*100:+.3f} |")
    lines+=['','## val966分类指标','', '| 类别 | A AP50 | B AP50 | A AP50–95 | B AP50–95 | AP50–95变化（百分点） |','|---|---:|---:|---:|---:|---:|']
    a={r['cls']:r for r in rows['A']['val966']['classes']};b={r['cls']:r for r in rows['B']['val966']['classes']}
    for cls in sorted(a):
        x,y=a[cls],b[cls];lines.append(f"| {x['name']} | {x['AP50']:.6f} | {y['AP50']:.6f} | {x['AP50_95']:.6f} | {y['AP50_95']:.6f} | {(y['AP50_95']-x['AP50_95'])*100:+.3f} |")
    lines+=['','## 成本、核验与限制','',f"- 初始参数SHA256相同：`{training['A']['initial_state_sha256']}`。",
            '- A/B各7852图；原始72273框、清洗72665框；共同val966为11487框。源数据及实验副本末尾哈希检查通过，见completed.json。',
            '- 训练耗时：'+ '；'.join(f"{arm} {training[arm]['seconds']:.1f}秒" for arm in ['A','B'])+'。',
            '- 六次统一评估总耗时：'+f"{sum(r['seconds'] for v in rows.values() for r in v.values()):.1f}秒。",
            '- 统一FP32，conf=.001、NMS IoU=.7、max_det=300、imgsz640、batch32、无TTA。固定工作点先过滤conf>=.25再按原生规则重匹配，IoU=.50；不是调参或成功门槛。',
            '- 单种子、单次配对观测，不提供显著性结论。deterministic=true不保证所有CUDA算子和运行环境绝对可复现。',
            '- train集本身不是泛化评估；更小差距可能来自train下降。必须同时看共同val绝对指标；P/R的最佳F1点可能不同，不能当成共同固定工作点。',
            '- 清洗来自既有模型诊断和视觉审核；本次没有独立test盲测，不冒充未经选择的测试集结论或竞赛流程合规认证。',
            '- 不代做错误根因结论，不据本结果自动选择模型、追加训练或设计后续实验。',
            '- 问题解决来源：AI 直接解决（执行器、训练监督、评估及报告）；A/B及SS04配置由学习者指定。','',
            '## 权重','']
    for arm in ['A','B']:
        lines += [f"- {arm} best：`{training[arm]['best']}`；SHA256 `{training[arm]['best_sha256']}`。"]
    with (OUT/'RESULT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print('SUMMARY WRITTEN',OUT/'RESULT.md')


if __name__=='__main__':main()
