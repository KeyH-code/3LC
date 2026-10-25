"""Build one self-contained human-review HTML; never modify dataset files."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import base64
import hashlib
import json
import math
from PIL import Image

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
ROOT = AUDIT.parent.parent
DATA = ROOT / 'multi-vehicle_kaggle_competition_starter_kit_files' / 'data'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def build():
    candidates_path = AUDIT / 'supplemental/supplemental_data_audit_near_duplicate_candidates.json'
    candidates = load(candidates_path)
    assert len(candidates) == 4862
    known = load(AUDIT / 'cleaning_audit.json')
    known_pairs = {(p['train']['rel'], p['val']['rel']) for p in known['pairs']}
    assert len(known_pairs) == 16
    assert not known_pairs.intersection((p['train'], p['val']) for p in candidates)
    assert len({(p['train'], p['val']) for p in candidates}) == len(candidates)
    count = int((Decimal(len(candidates)) * Decimal('0.05')).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    rank = lambda p: digest((p['train'] + '\n' + p['val']).encode('utf-8'))
    ranked = sorted(enumerate(candidates, 1), key=lambda ip: (rank(ip[1]), ip[1]['train'], ip[1]['val']))[:count]
    ranked.sort(key=lambda ip: (ip[1]['distance'], ip[1]['train'], ip[1]['val']))
    near = [{'id': 'near-' + str(i).zfill(3), 'sample_index': i, 'candidate_index': ci,
             'train': p['train'], 'val': p['val'], 'distance': p['distance'],
             'same_video_prefix': p['same_video_prefix'], 'selection_sha256': rank(p)}
            for i, (ci, p) in enumerate(ranked, 1)]
    edges = load(AUDIT / 'supplemental/supplemental_data_audit_edge_overflow_decimal.json')
    large = [e for e in edges if Decimal(e['decimal_max_excess']) > Decimal('0.000001')]
    grouped = defaultdict(list)
    for e in large:
        grouped[e['image']].append(e)
    edge_rows = []
    for i, (rel, items) in enumerate(sorted(grouped.items(), key=lambda kv: (-max(Decimal(e['decimal_max_excess']) for e in kv[1]), kv[0])), 1):
        edge_rows.append({'id': 'edge-' + str(i).zfill(3), 'image': rel,
                          'lines': [e['line'] for e in items], 'issues': items,
                          'max_overflow_pixels': float(max(Decimal(e['decimal_max_excess']) for e in items) * 640)})
    empty = [{'id': 'empty-' + str(i).zfill(2), 'image': e['image']['rel']}
             for i, e in enumerate(known['empty_labels'], 1)]
    assert len(near) == 243 and len(edge_rows) == 23 and len(large) == 26 and len(empty) == 2
    manifest = {x['relative'].replace('\\', '/'): x for x in load(ROOT / 'reports/data_manifest.json')}
    required = {rel for p in near for rel in (p['train'], p['val'])}
    required.update(e['image'] for e in edge_rows + empty)
    images = {}
    embedded = {}
    for rel in sorted(required):
        assert rel.startswith(('train/images/', 'val/images/'))
        path = DATA / rel
        blob = path.read_bytes()
        assert len(blob) == manifest[rel]['bytes'] and digest(blob) == manifest[rel]['sha256'], rel
        label_rel = rel.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        lb = (DATA / label_rel).read_bytes()
        assert len(lb) == manifest[label_rel]['bytes'] and digest(lb) == manifest[label_rel]['sha256'], label_rel
        boxes = []
        for line in lb.decode('utf-8-sig').splitlines():
            if line.strip():
                tokens = line.split()
                assert len(tokens) == 5
                coords = list(map(float, tokens[1:]))
                assert all(math.isfinite(v) for v in coords)
                boxes.append([int(tokens[0]), *coords])
        with Image.open(path) as im:
            im.load()
            assert im.size == (640, 640)
        info = {'relative': rel, 'label_relative': label_rel, 'name': path.stem.split('_jpg.rf.')[0],
                'split': rel.split('/')[0], 'box_count': len(boxes), 'boxes': boxes,
                'sha256': digest(blob), 'label_sha256': digest(lb), 'bytes': len(blob)}
        images[rel] = info
        embedded[rel] = {**info, 'src': 'data:image/jpeg;base64,' + base64.b64encode(blob).decode('ascii')}
    assert all(images[e['image']]['box_count'] == 0 for e in empty)
    sample_id = digest('\n'.join(p['selection_sha256'] for p in near).encode('ascii'))
    meta = {'created_at': datetime.now().astimezone().isoformat(), 'sample_id': sample_id,
            'population': len(candidates), 'selected_pairs': count, 'requested_fraction': 0.05,
            'actual_fraction': count / len(candidates),
            'sampling': 'SHA256(UTF-8(train_relative + newline + val_relative)); ascending digest; first round_half_up(N*0.05); display sorted by distance and paths',
            'candidate_file_sha256': digest(candidates_path.read_bytes()),
            'sample_distance_histogram': dict(sorted(Counter(str(p['distance']) for p in near).items())),
            'unique_embedded_images': len(images), 'edge_images': len(edge_rows), 'edge_boxes': len(large),
            'micro_overflow_boxes_not_in_review': len(edges) - len(large), 'empty_images': len(empty),
            'dataset_modified': False, 'confirmed_move_status': 'pending clarification; no data moved',
            'deferred_label_conflicts': True}
    review_manifest = {'metadata': meta, 'near_pairs': near, 'empty': empty, 'edges': edge_rows, 'images': images}
    payload = {**review_manifest, 'images': embedded}
    template = (HERE / 'review.template.html').read_text(encoding='utf-8')
    assert template.count('__REVIEW_DATA__') == 1
    packed = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    output = template.replace('__REVIEW_DATA__', packed)
    (HERE / 'review.html').write_text(output, encoding='utf-8')
    (HERE / 'review_manifest.json').write_text(json.dumps(review_manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    move = {'status': 'not_executed', 'request': 'move all 16 val records identical to train into train; keep original train records',
            'reason_pending': 'official split / validation-label-use rule conflict; clarify non-competition scope before move',
            'train_before': 7852, 'val_before': 982, 'train_if_keep_both_records': 7868, 'val_if_move': 966,
            'labels_merged_or_selected': False,
            'records': [{'train_image': p['train']['rel'], 'val_image': p['val']['rel'],
                         'train_label': p['train']['label_rel'], 'val_label': p['val']['label_rel'],
                         'train_boxes': p['train']['boxes'], 'val_boxes': p['val']['boxes']}
                        for p in known['pairs']]}
    (HERE / 'confirmed_move_pending.json').write_text(json.dumps(move, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({**meta, 'html_bytes': (HERE / 'review.html').stat().st_size,
                      'html': str(HERE / 'review.html')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    build()
