"""Verify self-contained review deliverable; no training or data edits."""
from pathlib import Path
from datetime import datetime
from collections import Counter
from decimal import Decimal
from html.parser import HTMLParser
import ast
import base64
import hashlib
import json
import re
import sys

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
ROOT = AUDIT.parent.parent
DATA = ROOT / 'multi-vehicle_kaggle_competition_starter_kit_files' / 'data'
OUTPUT = HERE / 'verification.json'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


class ResourceCheck(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag in ('script', 'img', 'iframe', 'link', 'object'):
            for attr in ('src', 'href', 'data'):
                if attr in values and not values[attr].startswith(('data:', '#')):
                    self.external.append((tag, attr, values[attr]))


def main():
    assert Path(sys.executable).resolve() == Path(r'E:\anaconda\envs\dl-3lc\python.exe').resolve()
    result = {'started': datetime.now().astimezone().isoformat(), 'checks': {}, 'status': 'running'}
    try:
        text = (HERE / 'review.html').read_text(encoding='utf-8')
        found = re.search(r'<script id="review-data" type="application/json">(.*?)</script>', text, re.S)
        assert found and '__REVIEW_DATA__' not in text
        packed = load(HERE / 'review_manifest.json')
        payload = json.loads(found[1])
        assert payload['metadata'] == packed['metadata']
        meta = packed['metadata']
        assert meta['selected_pairs'] == 243 and meta['population'] == 4862
        assert meta['actual_fraction'] == 243 / 4862
        candidates_path = AUDIT / 'supplemental/supplemental_data_audit_near_duplicate_candidates.json'
        assert sha(candidates_path.read_bytes()) == meta['candidate_file_sha256']
        candidates = load(candidates_path)
        rank = lambda p: sha((p['train'] + '\n' + p['val']).encode('utf-8'))
        expected = sorted(candidates, key=lambda p: (rank(p), p['train'], p['val']))[:243]
        got = packed['near_pairs']
        assert len(got) == len({(r['train'], r['val']) for r in got}) == 243
        assert {(p['train'], p['val']) for p in got} == {(p['train'], p['val']) for p in expected}
        assert Counter(str(p['distance']) for p in got) == {'0': 16, '1': 23, '2': 40, '3': 74, '4': 90}
        assert sha('\n'.join(p['selection_sha256'] for p in got).encode('ascii')) == meta['sample_id']
        for r in got:
            source = candidates[r['candidate_index'] - 1]
            assert (r['train'], r['val'], r['distance']) == (source['train'], source['val'], source['distance'])
        result['checks']['deterministic_sample_243_of_4862'] = 'passed'
        assert len(packed['empty']) == 2
        assert len(packed['edges']) == 23 and sum(len(e['issues']) for e in packed['edges']) == 26
        assert all(Decimal(e['decimal_max_excess']) > Decimal('0.000001') for r in packed['edges'] for e in r['issues'])
        assert all(packed[k] == payload[k] for k in ('near_pairs', 'empty', 'edges'))
        assert len(packed['images']) == len(payload['images']) == meta['unique_embedded_images'] == 413
        full = {r['relative'].replace('\\', '/'): r for r in load(ROOT / 'reports/data_manifest.json')}
        for rel, info in packed['images'].items():
            embedded = payload['images'][rel]
            assert {k: v for k, v in embedded.items() if k != 'src'} == info
            blob = base64.b64decode(embedded['src'].split(',', 1)[1], validate=True)
            assert sha(blob) == info['sha256'] == full[rel]['sha256']
            assert blob == (DATA / rel).read_bytes() and len(blob) == info['bytes']
            lb = (DATA / info['label_relative']).read_bytes()
            assert sha(lb) == info['label_sha256'] == full[info['label_relative']]['sha256']
            rows = [line.split() for line in lb.decode('utf-8-sig').splitlines() if line.strip()]
            assert len(rows) == info['box_count'] == len(info['boxes'])
            assert [[int(r[0]), *map(float, r[1:])] for r in rows] == info['boxes']
        assert all(packed['images'][r['image']]['box_count'] == 0 for r in packed['empty'])
        result['checks']['413_embedded_original_images_and_label_counts'] = 'passed'
        inventory = {p.relative_to(DATA).as_posix() for split in ('train', 'val')
                     for folder in ('images', 'labels') for p in (DATA / split / folder).iterdir() if p.is_file()}
        allowed = {p for p in full if p.split('/')[0] in ('train', 'val')}
        assert inventory == allowed and len(allowed) == 17668
        for rel in allowed:
            blob = (DATA / rel).read_bytes()
            assert len(blob) == full[rel]['bytes'] and sha(blob) == full[rel]['sha256'], rel
        result['checks']['all_17668_dataset_files_unchanged'] = 'passed'
        move = load(HERE / 'confirmed_move_pending.json')
        assert move['status'] == 'not_executed' and len(move['records']) == 16
        assert not move['labels_merged_or_selected']
        parser = ResourceCheck()
        parser.feed(text)
        assert not parser.external
        assert "connect-src 'none'" in text
        assert 'function showFragment()' in text
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for path in HERE.iterdir():
            if path.is_file() and path != OUTPUT and path.suffix in ('.py', '.json', '.md', '.html'):
                value = path.read_text(encoding='utf-8')
                assert '\ufffd' not in value and '\x00' not in value
                assert not re.search(r'^(?:<<<<<<< |=======$|>>>>>>> )', value, re.M)
                assert all(line.rstrip() == line for line in value.splitlines()), path
        readme = (HERE / 'README.md').read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)', readme):
            if not target.startswith(('http://', 'https://', '#')):
                dest = (HERE / target.split('#')[0]).resolve()
                assert dest == OUTPUT or dest.is_file(), target
        result['checks']['self_contained_csp_ast_utf8_links'] = 'passed'
        browser = load(HERE / 'browser_checks.json')
        assert browser['sample_id'] == meta['sample_id']
        assert all(p['errors'] == 0 for p in browser['page_loads'])
        result['browser_checks'] = browser
        result['artifacts'] = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                                'sha256': sha(p.read_bytes())}
                               for p in sorted(HERE.iterdir()) if p.is_file() and p != OUTPUT]
        result['status'] = 'passed'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        raise
    finally:
        result['ended'] = datetime.now().astimezone().isoformat()
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': result['checks'],
                      'remaining_browser_checks': result['browser_checks']['remaining_browser_checks'],
                      'output': str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
