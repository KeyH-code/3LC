"""Correct legacy scope text into a NEW page; preserve original preparation freeze."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/cleaning-expansion-080-070-060-20261029'


def correct_scope(page,count):
    replacements={
        '<title>OOF ≥0.85 候选核查</title>':'<title>三档OOF候选核查：D085参照</title>',
        'OOF ≥0.85：744 条候选核查':f'三档OOF：{count} 条候选核查（0.60 ≤ conf &lt; 0.85）',
        '蓝虚线 GT＝原始标签':'蓝虚线 GT＝D085 已清洗标签参照',
        '无原框充分覆盖':'无D085框充分覆盖',
        '无原标签充分覆盖':'无D085标签充分覆盖',
        '`原标 ${c.original_name} → 模型 ${c.predicted_name}`':'`D085标签 ${c.original_name} → 模型 ${c.predicted_name}`',
        '原框索引：':'D085框索引：',
        '整图＋原框':'整图＋D085框',
        '原框整图':'D085框整图',
        '可按状态/场景筛选。':'候选分档固定为[0.80,0.85)、[0.70,0.80)、[0.60,0.70)，旧≥0.85项目不在本轮范围。可按状态/场景筛选。',
    }
    for old,new in replacements.items():
        if page.count(old)!=1:raise ValueError('Unexpected legacy scope marker: '+old)
        page=page.replace(old,new,1)
    return page


def main():
    source=OUT/'index.html';target=OUT/'review_index_v2.html'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    freeze=json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))
    if sha(source)!=freeze['initial_index_sha256']:raise ValueError('Original frozen page drift')
    page=source.read_text(encoding='utf-8');updated=correct_scope(page,6188)
    with target.open('x',encoding='utf-8') as f:f.write(updated)
    if sha(source)!=freeze['initial_index_sha256']:raise ValueError('Original page changed')
    with (OUT/'review_page_v2_verification.json').open('x',encoding='utf-8') as f:
        json.dump(dict(at=datetime.now(timezone.utc).isoformat(),original_preserved=True,
            original_sha256=sha(source),new_page=target.name,new_sha256=sha(target),
            source_sha256=sha(Path(__file__)),tests_sha256=sha(ROOT/'tests/test_cleaning_expansion_page.py'),
            static_scope_corrected=True,browser_interaction_tested=False,new_visual_reviews=0,label_writes=0),f,ensure_ascii=False,indent=2)
    print('NEW PAGE WRITTEN; original frozen page preserved:',target)


if __name__=='__main__':main()
