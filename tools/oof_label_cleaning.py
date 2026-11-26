"""Saved-OOF label review workflow. No torch/YOLO/network calls.

prepare and summarize NEVER write training labels. apply requires a separately
supplied phase-B approval bound to a complete, frozen proposal. Review/model
confirmation records are evidence/provenance, not an authorization bypass.
"""
import argparse
import collections
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
OOF = 'reports/oof-diagnostic-5fold-24e-retry1-20261026'
DEFAULT_OUT = 'reports/oof-label-cleaning-conf085-20261027'
PLAN = '.pi/plan/oof置信度-0-85候选视觉审核与标签清洗-先汇总停下-再确认修改-20261027-2326.md'
NAMES = ['truck', 'car', 'van', 'bus']
EXTRA = 'prediction_without_label_coverage'
CONFLICT = 'class_disagreement'
HISTORY = [
    ('reports/oof-ai-preliminary-review-20261026', 'reports/oof-human-review-20261027'),
    ('reports/oof-ai-review-30-20261027', 'reports/oof-human-review-30-20261027'),
]


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.writing')
    require(not tmp.exists(), f'Unfinished write exists: {tmp}')
    tmp.write_bytes(encoded(value))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'Expected safe relative path')
    target = (Path(root) / p).resolve()
    require(target.is_relative_to(Path(root).resolve()), 'Path escapes root')
    return target


def table(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()})


def checked_box(box):
    require(isinstance(box, list) and len(box) == 4, 'xyxy must contain four coordinates')
    require(all(type(v) in (int, float) and math.isfinite(v) for v in box), 'Nonfinite box')
    x1, y1, x2, y2 = box
    require(0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 640, 'Invalid 640x640 box')
    return box


def label_box(row):
    c, cx, cy, w, h = map(float, row)
    require(c in range(4) and c == int(c), 'Invalid class')
    return [(cx-w/2)*640, (cy-h/2)*640, (cx+w/2)*640, (cy+h/2)*640]


def overlap(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    aa = (a[2]-a[0])*(a[3]-a[1]); ba = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa+ba-inter), inter / aa


def identity(row):
    return (row['id'], row['signal'], row['prediction_id'], row.get('gt_index'))


def candidate_key(row):
    return hashlib.sha256(json.dumps(identity(row), separators=(',', ':')).encode()).hexdigest()


def signals(src, rec, threshold):
    """Exact old matching convention, numpy imported only for saved-record computation."""
    import numpy as np
    pred = [p for p in rec['candidates'] if p['confidence'] >= threshold]
    if not pred:
        return []
    pb = np.asarray([checked_box(p['xyxy']) for p in pred], dtype=np.float64)
    pc = np.asarray([p['cls'] for p in pred], dtype=np.int64)
    labels = np.asarray(src['labels'], dtype=np.float64).reshape(-1, 5)
    pa = np.prod(pb[:, 2:]-pb[:, :2], axis=1)
    gb = np.asarray([label_box(r) for r in labels]).reshape(-1, 4)
    if len(labels):
        gc = labels[:, 0].astype(np.int64)
        inter = np.prod(np.maximum(0, np.minimum(pb[:, None, 2:], gb[None, :, 2:]) -
                                   np.maximum(pb[:, None, :2], gb[None, :, :2])), axis=2)
        ga = np.prod(gb[:, 2:]-gb[:, :2], axis=1)
        iou = inter/(pa[:, None]+ga[None, :]-inter)
        maxiou = iou.max(axis=1); covered = (inter/pa[:, None]).max(axis=1)
        same = pc[:, None] == gc[None, :]
        sameiou = np.where(same, iou, 0).max(axis=1)
        valid = (~same) & (iou >= .7) & (sameiou[:, None] < .3)
    else:
        maxiou = covered = sameiou = np.zeros(len(pred))
        valid = np.zeros((len(pred), 0), dtype=bool)
        iou = valid.astype(float)
    result = []

    def make(k, signal, j=None):
        p = pred[k]
        require(p['cls'] in range(4) and p['class_name'] == NAMES[p['cls']], 'Prediction class mismatch')
        r = {field: src[field] for field in ['id', 'scene', 'frame', 'image_path', 'label_path', 'image_sha256', 'label_sha256']}
        r.update(fold=rec['fold'], model_sha256=rec['model_sha256'], prediction_id=p['id'],
                 predicted_class=p['cls'], predicted_name=p['class_name'], confidence=p['confidence'],
                 xyxy=p['xyxy'], signal=signal, max_iou_any_gt=float(maxiou[k]),
                 max_fraction_prediction_covered_by_gt=float(covered[k]),
                 gt_boxes=[{'class': int(g[0]), 'xyxy': b.tolist()} for g, b in zip(labels, gb)],
                 touches_image_edge=bool(min(p['xyxy'][:2]) <= 1 or max(p['xyxy'][2:]) >= 639))
        if j is not None:
            support = [q['confidence'] for q in rec['candidates'] if q['cls'] == int(labels[j, 0])
                       and overlap(q['xyxy'], gb[j])[0] >= .5]
            r.update(gt_index=j, original_class=int(labels[j, 0]), original_name=NAMES[int(labels[j, 0])],
                     gt_xyxy=gb[j].tolist(), iou_other_class=float(iou[k, j]), max_iou_same_class=float(sameiou[k]),
                     best_confidence_supporting_original_class_at_iou50=max(support, default=0))
        r['key'] = candidate_key(r)
        return r

    for k in np.flatnonzero((maxiou < .1) & (covered < .2)):
        result.append(make(int(k), EXTRA))
    for j in range(len(labels)):
        ks = np.flatnonzero(valid[:, j])
        if len(ks):
            k = max(ks, key=lambda i: pred[i]['confidence'])
            result.append(make(int(k), CONFLICT, j))
    return result


def historical_seeds(root, candidates):
    lookup = {identity(r): r for r in candidates}; found = {}; unmatched = []
    for ai_dir, human_dir in HISTORY:
        aa = {r['case']: r for r in load(root/ai_dir/'review.json')['rows']}
        for h in load(root/human_dir/'review.json')['rows']:
            a = aa[h['case']]
            require(all(h[k] == a[k] for k in ['scene', 'frame', 'prediction_id']), 'Historical identity mismatch')
            key = identity(a)
            if key not in lookup:
                unmatched.append({'batch': human_dir, 'case': h['case'], 'identity': list(key)})
                continue
            c = lookup[key]
            require(c['image_sha256'] == a['image_sha256'] and c['label_sha256'] == a['label_sha256'], 'Historical source drift')
            evidence = a.get('evidence_screenshot', a.get('screenshot'))
            require(bool(evidence), 'Missing historical screenshot')
            ep = inside(root/ai_dir, evidence)
            e = {'path': ep.relative_to(root).as_posix(), 'sha256': sha(ep), 'root': 'workspace'}
            status = h['adjudication_status']
            decision = {'missing_label': 'add_box', 'wrong_class': 'change_class',
                        'original_label_correct_model_wrong': 'keep', 'not_required_to_annotate': 'keep'}.get(status, 'defer')
            row = dict(key=c['key'], decision=decision, source='human_verified',
                       proposed_class=NAMES.index(h['human_class']) if h.get('human_class') else None,
                       geometry_ready=False, final_xyxy=None, object_key=None,
                       reason=h['raw_feedback'], observation=a['visual_observation'], uncertainty=a.get('limitations', ''),
                       evidence=[e], visual_completed=True, review_model='unknown (historical AI; human adjudication retained)',
                       historical_status=status, historical_batch=human_dir, historical_case=h['case'])
            if decision == 'change_class':
                require(h['human_class'] != c['original_name'], 'Class correction equals original class')
                row['geometry_ready'] = True
            require(c['key'] not in found, 'Duplicate historical candidate')
            found[c['key']] = row
    return found, unmatched


def protected_paths(root, source):
    files = {root/OOF/'source_manifest.json', root/OOF/'oof_predictions.jsonl',
             root/'reports/oof-label-diagnostic-20261026/summary.json',
             root/'reports/oof-label-diagnostic-20261026/high_confidence_candidates.json',
             root/'AGENTS.md', root/'RULES.md', root/PLAN}
    files.update(p for p in (root/'multi-vehicle_kaggle_competition_starter_kit_files').glob('*.py'))
    files.update([root/'multi-vehicle_kaggle_competition_starter_kit_files/config.yaml',
                  root/'multi-vehicle_kaggle_competition_starter_kit_files/dataset.yaml'])
    files.update(p for p in (root/'tools').glob('oof_diagnostic.*'))
    for aa, hh in HISTORY:
        for folder in [aa, hh]:
            files.update(p for p in (root/folder).rglob('*') if p.is_file())
    for r in source:
        files.update([inside(root, r['image_path']), inside(root, r['label_path'])])
    # The OOF manifest contains train only. Protect original val too, without
    # using any val annotation to make cleaning decisions or reading test.
    original_manifest = root/'reports/data_manifest.json'
    files.add(original_manifest)
    for entry in load(original_manifest):
        relative = Path(entry['relative'])
        if relative.parts[0] == 'val':
            path = inside(root/'multi-vehicle_kaggle_competition_starter_kit_files/data', str(relative))
            require(sha(path) == entry['sha256'], 'Original val integrity drift')
            files.add(path)
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(files)}


def current_code(root):
    return {str(p.relative_to(root)): sha(p) for p in [root/'tools/oof_label_cleaning.py',
            root/'tools/oof_label_cleaning_review.template.html', root/'tests/test_oof_label_cleaning.py']}


def prepare(root, out):
    require(not out.exists(), 'Report directory exists; use a new version, never overwrite')
    source = [r for r in load(root/OOF/'source_manifest.json') if r['split'] == 'train']
    sm = {r['id']: r for r in source}
    require(len(sm) == len(source) == 7852, 'Expected exactly 7852 unique train images')
    before = protected_paths(root, source)
    candidates = []; seen = set(); anchors = {t: [] for t in [.7, .9]}
    with (root/OOF/'oof_predictions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line); ident = rec['id']
            require(ident in sm and ident not in seen, 'Unknown or duplicate OOF image')
            seen.add(ident); src = sm[ident]
            require(all(rec[k] == src[k] for k in ['scene', 'frame', 'image_path', 'label_path', 'image_sha256', 'label_sha256']), 'OOF/source mismatch')
            for t in anchors:
                anchors[t].extend(signals(src, rec, t))
            candidates.extend(signals(src, rec, .85))
    require(seen == set(sm), 'OOF coverage incomplete')
    old = load(root/'reports/oof-label-diagnostic-20261026/summary.json')['thresholds']
    for t, rows in anchors.items():
        for signal, field in [(EXTRA, 'extra_separated'), (CONFLICT, 'class_conflict')]:
            selected = [r for r in rows if r['signal'] == signal]
            require(len(selected) == sum(old[str(t)][field]), f'Anchor count drift {t}/{signal}')
            require(len({r['id'] for r in selected}) == old[str(t)]['image_counts'][field], 'Anchor image drift')
            counts = collections.Counter(r['predicted_class'] for r in selected)
            require([counts[i] for i in range(4)] == old[str(t)][field], 'Anchor class drift')
    high = load(root/'reports/oof-label-diagnostic-20261026/high_confidence_candidates.json')['rows']
    require({identity(r) for r in anchors[.9]} == {identity(r) for r in high}, 'Old 176 candidate identities changed')
    counts = collections.Counter(r['signal'] for r in candidates)
    require(counts == {EXTRA: 521, CONFLICT: 223}, 'Expected 521+223 candidates')
    require(len({r['id'] for r in candidates}) == 591 and len({r['scene'] for r in candidates}) == 69, 'Expected 591 images/69 scenes')
    candidates.sort(key=lambda r: (r['scene'], r['frame'], r['id'], r['signal'], r['prediction_id'], r.get('gt_index', -1)))
    for i, c in enumerate(candidates, 1):
        c['number'] = i
        raw = inside(root, c['label_path']).read_text(encoding='utf-8').splitlines()
        parsed = [line.split() for line in raw if line.strip()]
        require(len(parsed) == len(c['gt_boxes']), 'Label count drift')
        require(all(int(float(a[0])) == b['class'] and max(abs(x-y) for x, y in zip(label_box(a), b['xyxy'])) < .01
                    for a, b in zip(parsed, c['gt_boxes'])), 'Label geometry drift')
        if 'gt_index' in c:
            require(all(line.strip() for line in raw), 'Blank lines would invalidate historical row index')
            c['original_label_line'] = raw[c['gt_index']]
        for field in ['image_path', 'label_path']:
            require(before[c[field]] == c[field.replace('_path', '_sha256')], 'Original data hash drift')
    inherited, unmatched = historical_seeds(root, candidates)
    require(len(inherited) == 38 and not unmatched, 'All historical 38 targets must map exactly')
    reviews = [inherited.get(c['key'], dict(key=c['key'], decision='unreviewed', source=None,
               proposed_class=None, geometry_ready=False, final_xyxy=None, object_key=None, reason='',
               observation='', uncertainty='', evidence=[], visual_completed=False, review_model=None)) for c in candidates]
    out.mkdir(parents=True)
    (out/'evidence').mkdir()
    save(out/'input_integrity.json', {'protected': before, 'code': current_code(root), 'created_at': now()})
    save(out/'selection_config.json', {'confidence_min': .85, 'uncovered_max_iou_strict_lt': .1,
         'uncovered_max_fraction_strict_lt': .2, 'other_class_iou_ge': .7, 'same_class_iou_strict_lt': .3,
         'per_gt': 'highest confidence qualifying prediction', 'expected_signals': 744, 'expected_images': 591,
         'requested_review_model': 'gpt-6-luna', 'actual_new_review_model': None, 'model_entry_verified': False,
         'phase_b_authorized': False, 'anchor_070_090_passed': True, 'old_176_identities_match': True})
    save(out/'candidates.json', {'schema': 'oof085-candidates-v1', 'rows': candidates})
    save(out/'inherited.json', {'rows': list(inherited.values()), 'unmatched': unmatched})
    save(out/'review.json', {'schema': 'oof085-review-v1', 'rows': reviews})
    save(out/'freeze.json', {'files': {n: sha(out/n) for n in ['input_integrity.json', 'selection_config.json', 'candidates.json', 'inherited.json']}, 'frozen_at': now()})
    table(out/'candidates.csv', candidates, ['number', 'key', 'id', 'scene', 'frame', 'signal', 'prediction_id',
          'gt_index', 'original_name', 'predicted_name', 'confidence', 'xyxy', 'gt_xyxy', 'image_path', 'label_path'])
    summarize(root, out)


def check_integrity(root, out, full=False):
    for name, expected in load(out/'freeze.json')['files'].items():
        require(sha(inside(out, name)) == expected, f'Frozen artifact modified: {name}')
    info = load(out/'input_integrity.json')
    if full:
        for name, expected in info['protected'].items():
            require(sha(inside(root, name)) == expected, f'Protected input changed: {name}')
    return info


def confirmed_model(out):
    p = out/'model_confirmation.json'
    require(p.is_file(), 'Review model image entry not confirmed; new visual decisions blocked')
    m = load(p)
    require(m.get('confirmed_by_user') is True and m.get('image_input_capable') is True, 'Model confirmation incomplete')
    require(bool(m.get('evidence_reference')), 'Missing review model evidence')
    if m.get('actual_model') == 'current-session-inherited':
        authorization_path = out/'review_model_authorization.json'
        require(authorization_path.is_file(), 'Current-model subagent choice not explicitly authorized')
        authorization = load(authorization_path)
        require(m.get('selection_mode') == 'inherit_current_session_subagent'
                and authorization.get('mode') == 'inherit_current_session_subagent'
                and authorization.get('approved_by_user') is True
                and bool(authorization.get('message')) and bool(authorization.get('source_reference')),
                'Current-model subagent choice not explicitly authorized')
    else:
        require(m.get('actual_model') == 'gpt-6-luna', 'Unverified or different review model')
    return m


def evidence_check(root, out, record):
    require(record.get('evidence'), 'Visual evidence missing')
    for e in record['evidence']:
        base = root if e.get('root') == 'workspace' else out
        p = inside(base, e['path'])
        require(sha(p) == e['sha256'], 'Visual evidence hash mismatch')


def parent_review_model(out, row):
    """One explicit, scoped current-session reviewer; never impersonate the pinned pic model."""
    require(row.get('review_model') == 'current-session-parent'
            and row.get('reviewer_role') == 'parent', 'Review model does not match confirmed model')
    path = out/'parent_review_authorization.json'
    require(path.is_file(), 'Current parent reviewer not explicitly authorized')
    authorization = load(path)
    require(authorization.get('approved_by_user') is True
            and authorization.get('selection_mode') == 'any_actual_image_capable_agent_for_frozen_scope'
            and bool(authorization.get('user_message')) and bool(authorization.get('source_reference'))
            and authorization.get('image_input_verified') is True
            and bool(authorization.get('evidence_reference')),
            'Current parent authorization or image capability incomplete')
    require(row['key'] in authorization.get('candidate_keys', []), 'Parent review outside authorized frozen scope')
    require(any(v.get('actual_image_received') is True and bool(v.get('tool'))
                for v in row.get('vision_inputs', [])), 'Parent review requires actual image input record')


def session_review_model(root, out, row, candidate):
    """Keep new session reviewers separate from historical pinned-model evidence."""
    require((row.get('review_model'), row.get('reviewer_role')) in
            [('current-session-subagent', 'subagent'), ('current-session-parent', 'parent')],
            'Session reviewer identity mismatch')
    path = out/'session_review_authorization.json'
    require(path.is_file(), 'Session reviewer not explicitly authorized')
    authorization = load(path)
    require(authorization.get('approved_by_user') is True and authorization.get('phase') == 'A'
            and authorization.get('selection_mode') == 'actual_image_capable_session_agents'
            and bool(authorization.get('user_message')) and bool(authorization.get('source_reference'))
            and authorization.get('label_application_authorized') is False,
            'Session review authorization incomplete')
    require(row['key'] in authorization.get('candidate_keys', []), 'Session review outside authorized frozen scope')
    require(bool(row.get('actual_model')) and row['actual_model'] in authorization.get('allowed_models', []),
            'Session reviewer model not authorized')
    def resolved(item):
        return inside(root if item.get('root') == 'workspace' else out, item['path'])
    evidence_paths = {resolved(e) for e in row.get('evidence', [])}
    received = {resolved(v) for v in row.get('vision_inputs', [])
                if v.get('actual_image_received') is True and bool(v.get('tool')) and v.get('path')}
    linked = evidence_paths & received
    require(bool(linked),
            'Session review requires actual image input linked to evidence')
    original = inside(root, candidate['image_path'])
    bound = original in linked and sha(original) == candidate['image_sha256']
    manifest = out/'chrome-mcp/static-plates-manifest.json'
    if not bound and manifest.is_file():
        bound = any(candidate['key'] in p.get('keys', []) and inside(out, p['path']) in linked
                    and sha(inside(out, p['path'])) == p['sha256'] for p in load(manifest)['plates'])
    require(bound, 'Session review requires candidate-bound image evidence')


def validate_reviews(root, out):
    check_integrity(root, out)
    candidates = load(out/'candidates.json')['rows']; cm = {c['key']: c for c in candidates}
    rows = load(out/'review.json')['rows']; require(len(rows) == len(cm), 'Review count mismatch')
    require(len({r['key'] for r in rows}) == len(rows) and {r['key'] for r in rows} == set(cm), 'Review identity mismatch')
    inherited = {r['key']: r for r in load(out/'inherited.json')['rows']}
    for r in rows:
        c = cm[r['key']]; d = r['decision']; seed = inherited.get(r['key'])
        require(d in ['unreviewed', 'blocked', 'defer', 'keep', 'add_box', 'change_class'], 'Unknown decision')
        changed = seed is None or r != seed
        if seed:
            # Previously uncertain targets stay deferred until discussed; never silently overrule human feedback.
            for field in ['source', 'historical_status', 'historical_batch', 'historical_case', 'decision', 'proposed_class', 'reason']:
                require(r.get(field) == seed.get(field), f'Human adjudication override: {field}')
        if d == 'unreviewed':
            require(not seed and not r.get('visual_completed') and not r.get('source'), 'Invalid unreviewed provenance')
            continue
        if d == 'blocked':
            require(not r.get('visual_completed') and bool(r.get('reason')), 'Blocked is not completed visual review')
            continue
        require(r.get('visual_completed') is True and bool(r.get('observation')) and bool(r.get('reason')), 'Missing actual visual observation')
        evidence_check(root, out, r)
        if changed:
            m = confirmed_model(out)
            if r.get('review_model') != m['actual_model']:
                if (r.get('review_model') == 'current-session-subagent' or
                        (r.get('review_model') == 'current-session-parent' and r.get('actual_model'))):
                    session_review_model(root, out, r, c)
                else:
                    parent_review_model(out, r)
            require(r.get('source') == ('human_verified' if seed else 'agent_adopted'), 'Invalid provenance')
            require(r.get('evidence_sufficiency') in ['clear', 'uncertain'], 'Evidence sufficiency must be explicit')
            # Retain the human's missing-label verdict without pretending its geometry is final.
            # proposal() excludes these rows as pending_geometry; new Agent decisions stay strict.
            pending_human_geometry = (seed is not None and seed.get('source') == 'human_verified'
                                      and d == 'add_box' and r.get('geometry_ready') is False)
            if d in ['add_box', 'change_class', 'keep'] and not pending_human_geometry:
                require(r['evidence_sufficiency'] == 'clear' and not r.get('uncertainty'), 'Uncertain substantive decision must be deferred')
        if d in ['add_box', 'change_class']:
            require(type(r.get('proposed_class')) is int and r['proposed_class'] in range(4), 'Invalid proposed class')
            if d == 'change_class':
                require(c['signal'] == CONFLICT and r['proposed_class'] != c['original_class'], 'Invalid class change')
            else:
                require(c['signal'] == EXTRA, 'Addition must come from uncovered candidate')
                if r.get('geometry_ready'):
                    checked_box(r.get('final_xyxy'))
                    require(overlap(r['final_xyxy'], c['xyxy'])[0] > 0,
                            'Final box disjoint from selected candidate; explicit target remapping required')
                    require(bool(r.get('object_key')), 'Addition needs visual object identity for deduplication')
    return candidates, rows


def proposal(candidates, rows):
    cm = {c['key']: c for c in candidates}; groups = collections.defaultdict(list); deferred = []
    for r in rows:
        c = cm[r['key']]; d = r['decision']
        if d not in ['add_box', 'change_class']:
            if d in ['defer', 'blocked', 'unreviewed']:
                deferred.append({'key': r['key'], 'reason': r.get('reason'), 'status': d})
            continue
        if d == 'add_box' and not r.get('geometry_ready'):
            deferred.append({'key': r['key'], 'reason': 'human-confirmed missing target; final geometry pending', 'status': 'pending_geometry'})
            continue
        object_key = f"gt:{c['gt_index']}" if d == 'change_class' else 'new:' + r['object_key']
        groups[(c['id'], object_key)].append((c, r))
    changes = []
    for (image_id, object_key), entries in groups.items():
        c, r = entries[0]; signatures = {(r1['decision'], r1['proposed_class'], tuple(r1.get('final_xyxy') or [])) for _, r1 in entries}
        if len(signatures) != 1:
            deferred.extend({'key': c1['key'], 'reason': 'conflicting recommendations for same object', 'status': 'conflict'} for c1, _ in entries)
            continue
        item = {k: c[k] for k in ['id', 'scene', 'frame', 'image_path', 'label_path', 'image_sha256', 'label_sha256']}
        item.update(action=r['decision'], object_key=object_key, new_class=r['proposed_class'],
                    candidate_keys=[c1['key'] for c1, _ in entries], sources=sorted({r1['source'] for _, r1 in entries}),
                    gt_index=c.get('gt_index'), old_class=c.get('original_class'),
                    original_label_line=c.get('original_label_line'), final_xyxy=r.get('final_xyxy'),
                    historical=any('historical_case' in r1 for _, r1 in entries), evidence=[e for _, r1 in entries for e in r1['evidence']])
        item['change_id'] = hashlib.sha256(json.dumps([image_id, object_key], separators=(',', ':')).encode()).hexdigest()
        changes.append(item)
    # Mechanical overlap detects conflicts, never proves distinct objects or silently merges them.
    risky = set()
    additions = [x for x in changes if x['action'] == 'add_box']
    for x in additions:
        cs = [cm[k] for k in x['candidate_keys']]
        if any(overlap(x['final_xyxy'], g['xyxy'])[0] >= .5 for g in cs[0]['gt_boxes']):
            risky.add(x['change_id'])
        for y in additions:
            if x['id'] == y['id'] and x['change_id'] != y['change_id'] and overlap(x['final_xyxy'], y['final_xyxy'])[0] >= .5:
                risky.update([x['change_id'], y['change_id']])
    for x in changes:
        if x['change_id'] in risky:
            deferred.extend({'key': k, 'reason': 'overlap needs explicit object/box resolution; not automatic merge', 'status': 'overlap_conflict'} for k in x['candidate_keys'])
    return [x for x in changes if x['change_id'] not in risky], deferred


def render(root, out, candidates, rows):
    from os.path import relpath
    from urllib.parse import quote
    view_candidates = [{**r, 'image_url': quote(Path(relpath(inside(root, r['image_path']), out)).as_posix(), safe='/')}
                       for r in candidates]
    data = json.dumps({'candidates': view_candidates, 'reviews': rows}, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    template = (root/'tools/oof_label_cleaning_review.template.html').read_text(encoding='utf-8')
    (out/'index.html').write_text(template.replace('__DATA__', data), encoding='utf-8')


def summarize(root, out):
    candidates, rows = validate_reviews(root, out)
    changes, deferred = proposal(candidates, rows)
    decisions = dict(collections.Counter(r['decision'] for r in rows))
    incomplete = sum(r['decision'] in ['unreviewed', 'blocked'] for r in rows)
    model_ready = (out/'model_confirmation.json').is_file()
    state = 'awaiting_user_scope_confirmation' if incomplete == 0 else ('review_incomplete' if model_ready else 'blocked_model_confirmation')
    if incomplete and not model_ready and (out/'review_model_authorization.json').is_file():
        state = 'blocked_subagent_visual_input'
    pp = {'schema': 'oof085-proposal-v1', 'state': state, 'phase_b_authorized': False,
          'candidates_sha256': sha(out/'candidates.json'), 'review_sha256': sha(out/'review.json'), 'rows': changes}
    save(out/'proposed_changes.json', pp); save(out/'deferred.json', {'rows': deferred})
    counts = collections.Counter(x['action'] for x in changes)
    cm = {c['key']: c for c in candidates}
    resolved_keys = {key for change in changes for key in change['candidate_keys']}
    deferred_keys = {r['key'] for r in deferred}
    keep_keys = {r['key'] for r in rows if r['decision'] == 'keep'}
    require(len(resolved_keys | deferred_keys | keep_keys) == len(candidates)
            and not (resolved_keys & deferred_keys or resolved_keys & keep_keys or deferred_keys & keep_keys),
            'Candidate dispositions do not reconcile')
    summary = {'state': state, 'candidate_signals': len(candidates), 'candidate_images': len({c['id'] for c in candidates}),
               'candidate_scenes': len({c['scene'] for c in candidates}), 'signal_counts': dict(collections.Counter(c['signal'] for c in candidates)),
               'decision_counts': decisions, 'historical_targets_carried_forward': sum('historical_case' in r for r in rows),
               'new_visual_decisions': sum(r['source'] == 'agent_adopted' and r['visual_completed'] for r in rows),
               'historical_missing_geometry_pending': sum(r['decision'] == 'add_box' and not r.get('geometry_ready') and r['source'] == 'human_verified' for r in rows),
               'ready_proposed_changes': len(changes), 'ready_proposed_label_files': len({x['label_path'] for x in changes}),
               'ready_add_boxes': counts['add_box'], 'ready_change_classes': counts['change_class'],
               'ready_historical_changes': sum(x['historical'] for x in changes),
               'ready_new_agent_changes': sum(not x['historical'] for x in changes),
               'addition_classes': dict(collections.Counter(NAMES[x['new_class']] for x in changes if x['action'] == 'add_box')),
               'class_transitions': dict(collections.Counter(f"{NAMES[x['old_class']]}->{NAMES[x['new_class']]}" for x in changes if x['action'] == 'change_class')),
               'ready_changes_by_scene': dict(collections.Counter(x['scene'] for x in changes)),
               'deferred_reasons': dict(collections.Counter(x['status'] for x in deferred)),
               'resolved_change_signals': len(resolved_keys), 'keep_signals': len(keep_keys),
               'deferred_or_unreviewed_signals': len(deferred_keys),
               'actually_visually_completed_signals': sum(r.get('visual_completed') is True for r in rows),
                'newly_inspected_historical_geometry': sum('historical_case' in r and r.get('review_model') in ['gpt-6-luna', 'current-session-subagent', 'current-session-parent']
                    and r['decision'] == 'add_box' for r in rows),
               'ready_modified_image_percent': 100*len({x['id'] for x in changes})/7852,
               'ready_candidate_touches_edge_signals': sum(cm[k]['touches_image_edge'] for k in resolved_keys),
                'proposal_sha256': sha(out/'proposed_changes.json'), 'label_writes': 0, 'model_calls_by_mechanical_tool': 0,
                'model_entry_verified': False if not model_ready else confirmed_model(out)['actual_model'],
                'reviewer_counts': dict(collections.Counter(r.get('review_model') for r in rows if r.get('visual_completed'))),
                'actual_session_models': dict(collections.Counter(r['actual_model'] for r in rows
                    if r.get('visual_completed') and r.get('review_model') in ['current-session-subagent', 'current-session-parent']
                    and r.get('actual_model'))),
               'scope_note': ('Visual review complete; proposed scale only. Await separate phase-B confirmation; no labels applied.'
                              if incomplete == 0 else 'Partial progress is NOT a completed visual review or final modification scale.')}
    save(out/'summary.json', summary)
    table(out/'review.csv', [{**next(c for c in candidates if c['key'] == r['key']), **r} for r in rows],
          ['number', 'key', 'scene', 'frame', 'signal', 'decision', 'source', 'proposed_class', 'geometry_ready', 'reason', 'observation', 'uncertainty', 'review_model'])
    (out/'SUMMARY.md').write_text('# OOF ≥0.85 清洗准备状态\n\n'
        f"- 当前状态：`{state}`。模型选择授权与实际图像输入能力分别核验，不能互相替代。\n"
        f"- 候选：{len(candidates)}条 / {summary['candidate_images']}张 / {summary['candidate_scenes']}场景。\n"
        f"- 逐项目标状态：{json.dumps(decisions, ensure_ascii=False)}。\n"
         f"- 旧人工目标关联：{summary['historical_targets_carried_forward']}；新增视觉裁定：{summary['new_visual_decisions']}。\n"
        f"- 审核来源分布：{json.dumps(summary['reviewer_counts'], ensure_ascii=False)}；当前会话实际模型：{json.dumps(summary['actual_session_models'], ensure_ascii=False)}。\n"
        f"- 当前具备原框定位条件的改类建议：{counts['change_class']}；补框建议：{counts['add_box']}。这不是最终批准规模。\n"
        f"- 人工已确认漏标但坐标仍待定稿：{summary['historical_missing_geometry_pending']}。\n"
        f"- {summary['scope_note']} 没有标签修订数据集。\n"
        '- 先验证获准审核路径实际接收图像，再完成审核；最终汇总后仍必须等待第二次确认才可apply。\n'
        '- 核查页index.html三列显示，图片原始上下文与目标局部均可查看；加载成功不代表视觉审核完成。\n'
        '- 模型置信度仅作筛选，不是标签错误概率。人审意见按具体目标关联，不能推广到同图/邻帧其他目标。\n', encoding='utf-8')
    save(out/'proposal_seal.json', {'state': state, 'created_at': now(), 'files': {n: sha(out/n) for n in
         ['proposed_changes.json', 'review.json', 'deferred.json', 'summary.json']}})
    render(root, out, candidates, rows)
    return summary


def import_review(root, out, batch):
    confirmed_model(out)
    payload = load(batch); incoming = payload['rows']
    current = load(out/'review.json'); rm = {r['key']: r for r in current['rows']}
    require(len({r['key'] for r in incoming}) == len(incoming), 'Duplicate batch candidate')
    for r in incoming:
        require(r['key'] in rm, 'Unknown review candidate')
    before = (out/'review.json').read_bytes()
    for r in incoming:
        rm[r['key']] = r
    current['rows'] = [rm[r['key']] for r in current['rows']]
    save(out/'review.json', current)
    try:
        validate_reviews(root, out)
    except Exception:
        (out/'review.json').write_bytes(before)
        raise
    summarize(root, out)


def verify_review(root, out):
    check_integrity(root, out, full=True)
    candidates, rows = validate_reviews(root, out)
    recomputed, deferred = proposal(candidates, rows)
    require(load(out/'proposed_changes.json')['rows'] == recomputed, 'Proposal not synchronized with review')
    require(load(out/'deferred.json')['rows'] == deferred, 'Deferred records not synchronized')
    for name, expected in load(out/'proposal_seal.json')['files'].items():
        require(sha(inside(out, name)) == expected, f'Proposal seal mismatch: {name}')
    result = {'verified_at': now(), 'candidate_count': len(candidates), 'all_review_identities_checked': len(rows),
              'protected_sha256_files': len(load(out/'input_integrity.json')['protected']),
              'source_and_historical_files_unchanged': True, 'proposal_recomputed_exactly': True,
              'state': load(out/'summary.json')['state'], 'new_visual_decisions': load(out/'summary.json')['new_visual_decisions'],
              'code_hashes': current_code(root), 'label_writes': 0, 'model_calls': 0}
    save(out/'verification.json', result)
    return result


def authorize_apply(root, out, approval):
    require(approval is not None and Path(approval).is_file(), 'Separate phase-B approval required; nothing written')
    seal = load(out/'proposal_seal.json')
    require(seal['state'] == 'awaiting_user_scope_confirmation', 'Review not complete; cannot apply')
    for name, expected in seal['files'].items():
        require(sha(inside(out, name)) == expected, 'Frozen proposal changed after summary')
    p = load(out/'proposed_changes.json'); a = load(approval)
    require(a.get('phase') == 'B' and a.get('approved_by_user') is True, 'No phase-B approval')
    require(a.get('proposal_sha256') == sha(out/'proposed_changes.json'), 'Approval bound to another proposal')
    require(bool(a.get('user_message')) and bool(a.get('source_reference')), 'Approval provenance required')
    require(datetime.fromisoformat(a['approved_at']) > datetime.fromisoformat(seal['created_at']), 'Approval must follow the scale report')
    require(a.get('deferred_policy') == 'leave_unchanged', 'Unresolved targets require a new frozen proposal')
    require(set(a.get('change_ids', [])) == {x['change_id'] for x in p['rows']} and
            len(a['change_ids']) == len(p['rows']), 'Approval scope must exactly match frozen proposal')
    verify_review(root, out)
    return a, p['rows']


def edited_label(raw, changes):
    lines = raw.decode('utf-8').splitlines(keepends=True)
    newline = '\r\n' if b'\r\n' in raw else '\n'
    used = set()
    require(all(x.get('action') in ['add_box', 'change_class'] and type(x.get('new_class')) is int
                and x['new_class'] in range(4) for x in changes), 'Unsupported label edit')
    for c in changes:
        if c['action'] == 'change_class':
            j = c['gt_index']; require(j not in used and 0 <= j < len(lines), 'Duplicate/invalid GT line')
            used.add(j)
            require(lines[j].rstrip('\r\n') == c['original_label_line'], 'Original label line changed')
            m = re.match(r'^(\s*)(\d+)(\s+.*)$', lines[j], flags=re.DOTALL)
            require(m and int(m[2]) == c['old_class'], 'GT class token mismatch')
            lines[j] = m[1] + str(c['new_class']) + m[3]
    text = ''.join(lines)
    for c in changes:
        if c['action'] == 'add_box':
            x1, y1, x2, y2 = checked_box(c['final_xyxy'])
            if text and not text.endswith(('\n', '\r')):
                text += newline
            coords = [(x1+x2)/1280, (y1+y2)/1280, (x2-x1)/640, (y2-y1)/640]
            text += str(c['new_class']) + ' ' + ' '.join(f'{v:.12f}' for v in coords) + newline
    for line in text.splitlines():
        if line.strip():
            parts = line.split(); require(len(parts) == 5, 'Malformed YOLO row')
            box = label_box(parts)
            require(all(math.isfinite(v) for v in box) and box[0] < box[2] and box[1] < box[3]
                    and min(box) >= -1e-6 and max(box) <= 640+1e-6, 'Invalid YOLO geometry')
    return text.encode('utf-8')


def apply(root, out, approval):
    a, changes = authorize_apply(root, out, approval)  # No filesystem mutation before ALL guards.
    destination = out/'dataset'
    require(not destination.exists() and not (out/'approval_record.json').exists(), 'Never overwrite an applied dataset')
    source = [r for r in load(root/OOF/'source_manifest.json') if r['split'] == 'train']
    by_path = collections.defaultdict(list)
    require({c['label_path'] for c in changes}.issubset({r['label_path'] for r in source}), 'Unknown train label target')
    require(len({Path(r['image_path']).name for r in source}) == len(source)
            and len({Path(r['label_path']).name for r in source}) == len(source), 'Nonunique source basename')
    for c in changes:
        by_path[c['label_path']].append(c)
    prepared = {}; expected_classes = collections.Counter(); original_count = 0
    for r in source:
        raw = inside(root, r['label_path']).read_bytes()
        require(sha(inside(root, r['label_path'])) == r['label_sha256'], 'Original label drift')
        original_count += len([x for x in raw.splitlines() if x.strip()])
        if r['label_path'] not in by_path:
            require(edited_label(raw, []) == raw, 'Unchanged label byte preservation failed')
        new = edited_label(raw, by_path.get(r['label_path'], []))
        prepared[r['id']] = new
        expected_classes.update(int(line.split()[0]) for line in new.splitlines() if line.strip())
    additions = sum(c['action'] == 'add_box' for c in changes)
    require(sum(expected_classes.values()) == original_count+additions, 'Unexpected box total')
    original_classes = collections.Counter(int(line[0]) for r in source for line in r['labels'])
    wanted_classes = original_classes.copy()
    for change in changes:
        if change['action'] == 'change_class':
            wanted_classes[change['old_class']] -= 1
        wanted_classes[change['new_class']] += 1
    require(expected_classes == wanted_classes, 'Class transition totals differ from proposal')
    bytes_needed = sum(inside(root, r['image_path']).stat().st_size+len(prepared[r['id']]) for r in source)
    require(shutil.disk_usage(out).free > bytes_needed+100_000_000, 'Insufficient disk space')
    images = destination/'images/train'; labels = destination/'labels/train'
    images.mkdir(parents=True); labels.mkdir(parents=True)
    manifest = []
    for r in source:
        src = inside(root, r['image_path']); im = images/src.name
        target = labels/Path(r['label_path']).name
        require(not im.exists() and not target.exists(), 'Nonunique destination basename')
        shutil.copyfile(src, im); target.write_bytes(prepared[r['id']])
        require(sha(im) == r['image_sha256'], 'Copied image differs')
        expected = hashlib.sha256(prepared[r['id']]).hexdigest()
        require(sha(target) == expected, 'Copied label differs')
        manifest.append({'id': r['id'], 'image': im.relative_to(out).as_posix(), 'label': target.relative_to(out).as_posix(),
                         'image_sha256': r['image_sha256'], 'label_sha256': expected, 'original_label_sha256': r['label_sha256']})
    check_integrity(root, out, full=True)
    save(out/'approval_record.json', a)
    save(out/'applied_changes.json', {'rows': changes, 'proposal_sha256': sha(out/'proposed_changes.json')})
    save(out/'dataset_manifest.json', {'images': len(source), 'original_boxes': original_count,
         'boxes': sum(expected_classes.values()), 'classes': dict(expected_classes), 'rows': manifest})
    changed_labels = [r for r in manifest if r['label_sha256'] != r['original_label_sha256']]
    require(len(changed_labels) == len(by_path), 'Actual changed label count differs from proposal')
    require(sum(1 for p in images.iterdir() if p.is_file()) == len(source)
            and sum(1 for p in labels.iterdir() if p.is_file()) == len(source), 'Dataset pair count mismatch')
    save(out/'applied_verification.json', {'image_count': len(source), 'label_count': len(source),
         'changed_label_files': len(changed_labels), 'added_boxes': additions,
         'final_boxes': sum(expected_classes.values()), 'class_totals_match_proposal': True,
         'all_original_hashes_unchanged': True, 'applied_at': now()})
    (destination/'README.md').write_text('Derived train only; no val/test or table registration. Original data unchanged.\n'
         'Rollback: use original dataset/table, never overwrite it with this derived copy.\n', encoding='utf-8')
    return {'destination': str(destination), 'images': len(source), 'new_boxes': additions, 'changed_files': len(by_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'summarize', 'verify-review', 'import-review', 'apply'])
    p.add_argument('--out', default=DEFAULT_OUT)
    p.add_argument('--batch'); p.add_argument('--approval')
    args = p.parse_args(); out = inside(ROOT, args.out)
    require(out.is_relative_to(ROOT/'reports'), 'Reports must remain inside workspace/reports')
    if args.action == 'prepare':
        prepare(ROOT, out); result = verify_review(ROOT, out)
    elif args.action == 'summarize':
        result = summarize(ROOT, out)
    elif args.action == 'verify-review':
        result = verify_review(ROOT, out)
    elif args.action == 'import-review':
        require(bool(args.batch), '--batch required'); import_review(ROOT, out, inside(ROOT, args.batch)); result = verify_review(ROOT, out)
    else:
        result = apply(ROOT, out, inside(ROOT, args.approval) if args.approval else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
