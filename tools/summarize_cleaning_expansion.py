"""Read-only metric/hypothesis assembly for the approved single-seed expansion run."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/train-cleaning-expansion-ss04-20261101'
OLD=ROOT/'reports/train-cleaning-ab-ss04-20261029'
RECOVERY=ROOT/'reports/eval-cleaning-ab-recovery-20261029'
ORDER=('A','B','D080','D070','D060')
AP='metrics/mAP50-95(B)'
AP50='metrics/mAP50(B)'
P='metrics/precision(B)'
R='metrics/recall(B)'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def f1(p,r):return 2*p*r/(p+r) if p+r else 0.
def compare(rows):
    val={b:rows[b]['val966']['native'][AP] for b in ORDER}
    delta={b:val[b]-val[ORDER[i-1]] for i,b in enumerate(ORDER) if i}
    gaps={b:{ref:rows[b][ref]['native'][AP]-val[b] for ref in ('D0','D085')} for b in ORDER}
    turn={}
    for ref in ('D0','D085'):
        scores=[rows[b][ref]['native'][AP] for b in ORDER]
        turn[ref]=any(scores[i]<scores[i-1] and any(scores[j]>scores[j-1] for j in range(i+1,len(scores)))
                      for i in range(1,len(scores)-1))
    strict=all(delta[b]>0 for b in ORDER[2:])
    decline=any(delta[b]<0 for b in ORDER[2:])
    diminishing=delta['D070']<delta['D080'] or delta['D060']<delta['D070']
    gap_changes={b:{ref:abs(gaps[b][ref])-abs(gaps['B'][ref]) for ref in ('D0','D085')} for b in ORDER[2:]}
    status=lambda observed:'supported_descriptively' if observed else 'not_supported'
    return dict(val=val,delta_previous=delta,delta_B={b:val[b]-val['B'] for b in ORDER[2:]},gaps=gaps,
        absolute_gaps={b:{ref:abs(v) for ref,v in g.items()} for b,g in gaps.items()},
        H1_strict_monotonic_val=strict,
        H1_all_new_above_B=all(val[b]>val['B'] for b in ORDER[2:]),
        H2_any_adjacent_decline=decline,H2_diminishing_adjacent_gain=diminishing,
        H3_fixed_train_down_then_up=turn,
        H4_absolute_gap_change_vs_B=gap_changes,
        hypothesis_status=dict(H1_strict_monotonic=status(strict),
            H1_all_new_above_B=status(all(val[b]>val['B'] for b in ORDER[2:])),
            H2_observed_diminishing_or_decline=status(decline or diminishing),
            H3_fixed_reference={ref:status(v) for ref,v in turn.items()},
            H4_gap_shrink_vs_B={b:{ref:status(v<0) for ref,v in g.items()} for b,g in gap_changes.items()},
            cross_seed_stability_and_causal_mechanism='insufficient_evidence'))


def main():
    rows={};training={}
    for band in ORDER:
        if band in ('A','B'):
            training[band]=load(OLD/band/'train/result.json')
            rows[band]={ref:load(RECOVERY/band/('eval_'+old)/'result.json')
                        for ref,old in [('D0','A'),('D085','B'),('val966','val966')]}
        else:
            training[band]=load(OUT/band/'train/result.json')
            rows[band]={ref:load(OUT/band/('eval_'+ref)/'result.json') for ref in ('D0','D085','self','val966')}
        t=training[band]
        if t['epochs']!=28:raise RuntimeError('Incomplete training')
        for row in rows[band].values():
            if row['state']!='completed' or row['weight_sha256']!=t['best_sha256']:raise RuntimeError('Unbound evaluation')
    if len({r['initial_state_sha256'] for r in training.values()})!=1:raise RuntimeError('Initial states differ')
    c=compare(rows)
    save(OUT/'comparison.json',dict(training=training,evaluations=rows,comparison=c))
    with (OUT/'metrics.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['model','reference','images','AP50','AP50_95','P_best_F1','R_best_F1','F1_at_native_point',
            'fixed_P_macro','fixed_R_macro','fixed_F1_macro','train_minus_val_AP95','absolute_gap'])
        for b in ORDER:
            for ref,row in rows[b].items():
                fixed=row['fixed'];fm=sum(f1(x['precision'],x['recall']) for x in fixed['classes'])/4
                gap=row['native'][AP]-rows[b]['val966']['native'][AP] if ref!='val966' else ''
                writer.writerow([b,ref,row['images'],row['native'][AP50],row['native'][AP],row['native'][P],row['native'][R],f1(row['native'][P],row['native'][R]),
                    fixed['precision_macro'],fixed['recall_macro'],fm,gap,abs(gap) if gap!='' else ''])
    with (OUT/'per_class.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['model','reference','class','AP50','AP50_95','P_best_F1','R_best_F1','F1_at_native_point',
            'TP_conf025_IoU050','FP','FN','P_fixed','R_fixed','F1_fixed'])
        for b in ORDER:
            for ref,row in rows[b].items():
                fixed={x['cls']:x for x in row['fixed']['classes']}
                for cls in row['classes']:
                    v=fixed[cls['cls']];p=cls['precision_best_f1'];r=cls['recall_best_f1']
                    writer.writerow([b,ref,cls['name'],cls['AP50'],cls['AP50_95'],p,r,f1(p,r),v['TP'],v['FP'],v['FN'],
                                     v['precision'],v['recall'],f1(v['precision'],v['recall'])])
    with (OUT/'class_deltas.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['model','reference','class','AP50_95','delta_B_pp','delta_previous_pp','fixed_FP','fixed_FN','delta_B_FP','delta_B_FN'])
        for i,b in enumerate(ORDER[2:],2):
            for ref in ('D0','D085','val966'):
                baseline={x['cls']:x for x in rows['B'][ref]['classes']}
                previous={x['cls']:x for x in rows[ORDER[i-1]][ref]['classes']}
                fixed={x['cls']:x for x in rows[b][ref]['fixed']['classes']}
                base_fixed={x['cls']:x for x in rows['B'][ref]['fixed']['classes']}
                for cls in rows[b][ref]['classes']:
                    k=cls['cls'];v=fixed[k];a=base_fixed[k]
                    writer.writerow([b,ref,cls['name'],cls['AP50_95'],(cls['AP50_95']-baseline[k]['AP50_95'])*100,
                        (cls['AP50_95']-previous[k]['AP50_95'])*100,v['FP'],v['FN'],v['FP']-a['FP'],v['FN']-a['FN']])
    scale=load(ROOT/'reports/cleaning-expansion-080-070-060-20261029/datasets-20261101/application_seal.json')['versions']
    with (OUT/'change_scale.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['version','anchor','candidate_cumulative','actions_relative_D085','net_new_actions','added_boxes','class_changes','changed_files','train_boxes','val_AP95','delta_B_pp','delta_previous_pp'])
        prior=0
        for b,anchor in [('D080',.8),('D070',.7),('D060',.6)]:
            d=scale[b];writer.writerow([b,anchor,{'D080':1197,'D070':3740,'D060':6188}[b],d['actions'],d['actions']-prior,d['added_boxes'],d['changed_classes'],d['changed_label_files'],d['boxes'],c['val'][b],c['delta_B'][b]*100,c['delta_previous'][b]*100]);prior=d['actions']
    lines=['# SS04 three-band cumulative cleaning: complete results','',
        'Three from-scratch 28-epoch YOLOv8n runs; twelve uniform FP32 evaluations. Prior A/B results reused, not rerun.',
        'All labels/proposals frozen before these runs. val966 is a development/best-selection set, not an independent test.','',
        '| Model | best epoch | val AP50 | val AP50-95 | delta vs B (pp) | delta previous (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    for b in ORDER:
        n=rows[b]['val966']['native'];lines.append(f"| {b} | {training[b]['best_epoch']} | {n[AP50]:.6f} | {n[AP]:.6f} | {(n[AP]-c['val']['B'])*100:+.3f} | {c['delta_previous'].get(b,0)*100:+.3f} |")
    lines+=['','## Hypothesis checks (descriptive, one seed)','',
        f"- Evidence classification: `{json.dumps(c['hypothesis_status'],ensure_ascii=False)}`.",
        f"- H1 strict B -> D080 -> D070 -> D060 val improvement: {c['H1_strict_monotonic_val']}.",
        f"- All new models above B: {c['H1_all_new_above_B']}; adjacent decline exists: {c['H2_any_adjacent_decline']}.",
        f"- H3 fixed-reference train down-then-up observed: {c['H3_fixed_train_down_then_up']} (self scores are NOT used for this check).",
        f"- H4 absolute train-val gap changes vs B: {c['H4_absolute_gap_change_vs_B']}.",
        '- No significance, cross-seed stability, causal mechanism or label-perfection claim. A decline does not identify a root cause.',
        '- AP/PR at native best-F1 points and common conf>=.25 / IoU=.50 rematched metrics are distinct; see per_class.csv.',
        '- H2 diminishing refers to raw adjacent AP increments, not a linear per-action effect; candidate composition also changes.',
        '- Class gains/losses and common-threshold FP/FN deltas are reported without hiding regressions in class_deltas.csv.',
        '', '## Fixed-reference train/val AP50-95','', '| Model | D0 train | D085 train | self train | val | D0 gap | D085 gap |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for b in ORDER:
        d=rows[b];self_score=d['self']['native'][AP] if 'self' in d else d['D0' if b=='A' else 'D085']['native'][AP]
        lines.append(f"| {b} | {d['D0']['native'][AP]:.6f} | {d['D085']['native'][AP]:.6f} | {self_score:.6f} | {d['val966']['native'][AP]:.6f} | {c['gaps'][b]['D0']:+.6f} | {c['gaps'][b]['D085']:+.6f} |")
    lines+=['','## Cost / provenance','',f"- Shared initial state SHA256: `{training['A']['initial_state_sha256']}`.",
        '- Train seconds: '+', '.join(f"{b}={training[b]['seconds']:.1f}" for b in ORDER[2:]),
        '- Evaluation seconds (12 new): '+str(sum(v['seconds'] for b in ORDER[2:] for v in rows[b].values())),
        '- Original sources and runtime copies verified before final success marker; no network/3LC or automatic retry.',
        '- AI directly implemented/executed the specifically authorized workflow. Experiment choices were set by the learner.','', '## New weights','']
    for b in ORDER[2:]:lines.append(f"- {b}: `{training[b]['best']}` SHA256 `{training[b]['best_sha256']}`")
    with (OUT/'RESULT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    return c


if __name__=='__main__':main()
