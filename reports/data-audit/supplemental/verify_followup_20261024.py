"""Verify audit artifacts and unchanged train/val; never execute model entrypoints."""
from pathlib import Path
from datetime import datetime
from collections import Counter
from decimal import Decimal
import ast
import hashlib
import json
import re
import statistics
import sys
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
ROOT = AUDIT.parent.parent
DATA = ROOT / 'multi-vehicle_kaggle_competition_starter_kit_files' / 'data'
OUTPUT = AUDIT / 'followup_verification_20261024.json'
PREFIX = 'supplemental_data_audit'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now().astimezone().isoformat()


def main():
    assert Path(sys.executable).resolve() == Path(r'E:\anaconda\envs\dl-3lc\python.exe').resolve()
    result = {'started': now(), 'status': 'running', 'checks': {}, 'selected_pairs': [], 'artifacts': []}
    try:
        manifest = load(ROOT / 'reports/data_manifest.json')
        expected = {x['relative'].replace('\\', '/'): x for x in manifest
                    if x['relative'].replace('\\', '/').split('/')[0] in ('train', 'val')}
        actual = {p.relative_to(DATA).as_posix() for split in ('train', 'val')
                  for folder in ('images', 'labels') for p in (DATA / split / folder).iterdir() if p.is_file()}
        assert actual == set(expected) and len(actual) == 17668
        for rel, item in expected.items():
            path = DATA / rel
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], rel
        result['checks']['manifest_files_unchanged'] = len(actual)

        report = load(HERE / (PREFIX + '_report.json'))
        records = load(HERE / (PREFIX + '_per_image.json'))
        candidates = load(HERE / (PREFIX + '_near_duplicate_candidates.json'))
        byimage = {r['image']: r for r in records}
        assert len(records) == len(byimage) == 8834
        for split, count, boxes in (('train', 7852, 72273), ('val', 982, 11615)):
            rr = [r for r in records if r['split'] == split]
            assert len(rr) == count and sum(r['lines'] for r in rr) == boxes
            assert not report['inventory'][split]['missing_labels']
            assert not report['inventory'][split]['orphan_labels']
            assert all(r['size'] == [640, 640] and not r.get('decode_error') for r in rr)
            assert all(not r['duplicate_text'] and not r['duplicate_numeric'] and not r['overlaps'] for r in rr)
            assert all(i['kind'] == 'box_edges_outside_image' for r in rr for i in r['issues'])
        assert sum(r['empty_label'] for r in records) == 2
        edges = load(HERE / (PREFIX + '_edge_overflow_decimal.json'))
        assert len(edges) == 199
        assert len({e['image'] for e in edges}) == 171
        assert sum(Decimal(e['decimal_max_excess']) == Decimal('0.0000005') for e in edges) == 173
        large = [e for e in edges if Decimal(e['decimal_max_excess']) > Decimal('0.000001')]
        assert len(large) == 26 and len({e['image'] for e in large}) == 23
        assert max(Decimal(e['decimal_max_excess']) for e in edges) * 640 == Decimal('2.399360')
        assert load(HERE / (PREFIX + '_all_high_overlap_pairs.json')) == []
        for name in ('_exitcode.txt', '_postprocess_exitcode.txt'):
            assert int((HERE / (PREFIX + name)).read_text(encoding='utf-8-sig').strip()) == 0
        result['checks']['label_decode_report_consistency'] = 'passed'

        known_audit = load(AUDIT / 'cleaning_audit.json')
        known = {(p['train']['rel'], p['val']['rel']) for p in known_audit['pairs']}
        trains = [(r['image'], int(r['dhash64'], 16)) for r in records if r['split'] == 'train']
        vals = [(r['image'], int(r['dhash64'], 16)) for r in records if r['split'] == 'val']
        got = {(t, v) for t, a in trains for v, b in vals if (a ^ b).bit_count() <= 4} - known
        assert len(got) == len(candidates) == 4862
        assert got == {(p['train'], p['val']) for p in candidates}
        assert Counter(p['distance'] for p in candidates) == {0: 321, 1: 597, 2: 896, 3: 1249, 4: 1799}
        for p in candidates:
            assert (int(byimage[p['train']]['dhash64'], 16) ^ int(byimage[p['val']]['dhash64'], 16)).bit_count() == p['distance']
        assert len({p['train'] for p in candidates}) == 2139
        assert len({p['val'] for p in candidates}) == 571
        result['checks']['all_7710664_dhash_distances_recomputed_from_archived_hashes'] = 'passed'

        for p in load(HERE / (PREFIX + '_top20_review.json'))[:3]:
            arrays = []
            for side in ('train', 'val'):
                with Image.open(DATA / p[side]) as image:
                    image.load()
                    arrays.append(np.asarray(image.convert('RGB'), dtype=np.int16))
            equal = np.array_equal(*arrays)
            result['selected_pairs'].append({'train': p['train'], 'val': p['val'],
                'bytes_equal': (DATA / p['train']).read_bytes() == (DATA / p['val']).read_bytes(),
                'rgb_pixels_equal': equal,
                'rgb_mean_absolute_difference_0_255': float(np.abs(arrays[0] - arrays[1]).mean())})
            assert not equal
        result['checks']['three_visual_pairs_not_pixel_identical'] = 'passed'

        official_path = ROOT / 'reports/OFFICIAL_REFERENCE_20261024.json'
        official = load(official_path)
        scores = official['private']['scores_in_displayed_rank_order']
        assert len(scores) == 97 and scores == sorted(scores, reverse=True)
        assert statistics.median(scores) == 0.80147
        assert [sum(x >= t for x in scores) for t in (0.8, 0.85, 0.9)] == [49, 22, 15]
        assert official['observed_date_local'] == '2026-10-24'
        result['checks']['leaderboard_arithmetic'] = 'passed'

        evidence = sorted((HERE / 'evidence').glob('*.jpg'))
        assert len(evidence) == 4
        for path in evidence:
            with Image.open(path) as image:
                image.load()
                assert image.width > 0 and image.height > 0
        docs = [ROOT / 'reports/EVALUATION_STANDARD_DRAFT.md', AUDIT / 'FOLLOWUP_20261024.md']
        for path in docs:
            text = path.read_text(encoding='utf-8')
            assert '\ufffd' not in text and '\x00' not in text
            assert not re.search(r'^(?:<<<<<<< |=======$|>>>>>>> )', text, re.M)
            assert all(line.rstrip() == line for line in text.splitlines())
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if not target.startswith(('https://', 'http://', '#')):
                    resolved = (path.parent / target.split('#')[0]).resolve()
                    assert resolved == OUTPUT or resolved.is_file(), (path, target)
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        result['checks']['evidence_ast_utf8_whitespace_links'] = 'passed'
        old = load(AUDIT / 'verification.json')
        assert all(sha(ROOT / a['path']) == a['sha256'] for a in old['artifacts'])
        result['checks']['previous_audit_artifacts_unchanged'] = 'passed'
        for path in sorted(HERE.iterdir()) + evidence + docs + [official_path]:
            if path.is_file():
                result['artifacts'].append({'path': str(path.relative_to(ROOT)),
                    'bytes': path.stat().st_size, 'sha256': sha(path)})
        result['status'] = 'passed'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        raise
    finally:
        result['ended'] = now()
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': result['checks'],
                      'selected_pairs': result['selected_pairs'], 'output': str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
