"""Random audit sampling and self-contained HTML only; no dataset/model writes."""
from pathlib import Path
from datetime import datetime
import ast
import base64
import hashlib
import json
import random
import re
import secrets
import sys
from PIL import Image

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
ROOT = AUDIT.parent.parent
DATA = ROOT / 'multi-vehicle_kaggle_competition_starter_kit_files' / 'data'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def inputs():
    original = {x['relative'].replace('\\', '/'): x for x in load(ROOT / 'reports/data_manifest.json')}
    previous_path = AUDIT / 'manual-review-20261024/review_manifest.json'
    previous = load(previous_path)
    seen = set(previous['images'])
    known = load(AUDIT / 'cleaning_audit.json')
    excluded_val = {p['val']['rel'] for p in known['pairs']}
    assert len(seen) == 413 and len(excluded_val) == 16
    images = {s: sorted(p for p in original if p.startswith(s + '/images/')) for s in ('train', 'val')}
    pools = {s: [p for p in images[s] if p not in seen and p not in excluded_val] for s in images}
    assert len(pools['train']) == 7620 and len(pools['val']) == 787
    return original, previous_path, seen, excluded_val, images, pools


def build():
    original, previous_path, seen, excluded_val, all_images, pools = inputs()
    config_path = HERE / 'selection_config.json'
    if config_path.exists():
        config = load(config_path)
    else:
        config = {'seed': secrets.randbelow(2 ** 32), 'seed_source': 'AI-generated random seed; user explicitly delegated audit seed choice',
                  'per_split_count': 48, 'method': 'Python random.Random(seed); sample sorted eligible train paths then sorted eligible val paths; without replacement',
                  'created_at': datetime.now().astimezone().isoformat()}
        save('selection_config.json', config)
    assert config['per_split_count'] == 48
    rng = random.Random(config['seed'])
    selected = {s: rng.sample(pools[s], 48) for s in ('train', 'val')}
    sample_id = sha(('seed=' + str(config['seed']) + '\n' + '\n'.join(selected['train'] + selected['val'])).encode('utf-8'))
    rows = [{'id': 'row-' + str(i + 1).zfill(2), 'index': i + 1,
             'train': selected['train'][i], 'val': selected['val'][i]} for i in range(48)]
    cards = {}
    embedded = {}
    for split in ('train', 'val'):
        for i, rel in enumerate(selected[split], 1):
            path = DATA / rel
            blob = path.read_bytes()
            assert sha(blob) == original[rel]['sha256'] and len(blob) == original[rel]['bytes']
            label_rel = rel.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            lb = (DATA / label_rel).read_bytes()
            assert sha(lb) == original[label_rel]['sha256'] and len(lb) == original[label_rel]['bytes']
            boxes = []
            for line in lb.decode('utf-8-sig').splitlines():
                if line.strip():
                    tok = line.split()
                    assert len(tok) == 5
                    boxes.append([int(tok[0]), *map(float, tok[1:])])
            with Image.open(path) as image:
                image.load()
                assert image.size == (640, 640)
            card = {'id': split + '-' + str(i).zfill(2), 'relative': rel, 'split': split,
                    'name': path.stem.split('_jpg.rf.')[0], 'label_relative': label_rel,
                    'box_count': len(boxes), 'boxes': boxes, 'bytes': len(blob),
                    'sha256': sha(blob), 'label_sha256': sha(lb)}
            cards[rel] = card
            embedded[rel] = {**card, 'src': 'data:image/jpeg;base64,' + base64.b64encode(blob).decode('ascii')}
    meta = {'sample_id': sample_id, 'created_at': datetime.now().astimezone().isoformat(),
            'seed': config['seed'], 'per_split': 48, 'rows': 48, 'total_images': 96,
            'population_original': {'train': 7852, 'val': 982},
            'eligible_sampling_population': {s: len(pools[s]) for s in pools},
            'sample_fraction_of_eligible': {s: 48 / len(pools[s]) for s in pools},
            'previous_html_images_excluded': len(seen), 'confirmed_val_images_excluded': len(excluded_val),
            'previous_manifest_sha256': sha(previous_path.read_bytes()),
            'pairing': 'independent random train/val samples paired only for side-by-side display; no visual similarity condition',
            'scope': 'remaining unshown eligible population, not a simple random sample of all original 8834 images',
            'data_modified': False, 'labels_modified': False, 'configured_for_model_use': False}
    manifest = {'metadata': meta, 'selection_config': config, 'rows': rows, 'images': cards,
                'excluded_val_images': sorted(excluded_val), 'excluded_previous_images': sorted(seen)}
    payload = {**manifest, 'images': embedded}
    template = (HERE / 'review.template.html').read_text(encoding='utf-8')
    assert template.count('__REVIEW_DATA__') == 1
    packed = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    (HERE / 'review.html').write_text(template.replace('__REVIEW_DATA__', packed), encoding='utf-8')
    save('review_manifest.json', manifest)
    val_list = [p for p in all_images['val'] if p not in excluded_val]
    assert len(val_list) == 966
    (HERE / 'val_use_list_excluding_16.txt').write_text(''.join(str(DATA / p) + '\n' for p in val_list), encoding='utf-8')
    save('val_use_list_metadata.json', {'status': 'list_only_not_wired_to_training_or_evaluation',
         'original_split_unchanged': True, 'original_val_count': 982, 'listed_val_count': 966,
         'excluded_count': 16, 'excluded_images': sorted(excluded_val),
         'list_sha256': sha((HERE / 'val_use_list_excluding_16.txt').read_bytes())})
    print(json.dumps({**meta, 'html_bytes': (HERE / 'review.html').stat().st_size}, ensure_ascii=False, indent=2))


def verify():
    result = {'started': datetime.now().astimezone().isoformat(), 'status': 'running', 'checks': {}}
    try:
        original, previous_path, seen, excluded_val, all_images, pools = inputs()
        manifest = load(HERE / 'review_manifest.json')
        meta = manifest['metadata']
        config = load(HERE / 'selection_config.json')
        rng = random.Random(config['seed'])
        expected = {s: rng.sample(pools[s], 48) for s in ('train', 'val')}
        assert len(manifest['rows']) == 48 and len(manifest['images']) == 96
        for s in ('train', 'val'):
            got = [r[s] for r in manifest['rows']]
            assert got == expected[s] and len(set(got)) == 48
            assert not set(got).intersection(seen | excluded_val)
        result['checks']['sample_and_exclusions'] = 'passed'
        html = (HERE / 'review.html').read_text(encoding='utf-8')
        packed = re.search(r'<script id="review-data" type="application/json">(.*?)</script>', html, re.S)
        assert packed and '__REVIEW_DATA__' not in html
        payload = json.loads(packed[1])
        assert payload['metadata'] == meta and payload['rows'] == manifest['rows']
        for rel, info in manifest['images'].items():
            emb = payload['images'][rel]
            assert {k: v for k, v in emb.items() if k != 'src'} == info
            blob = base64.b64decode(emb['src'].split(',', 1)[1], validate=True)
            assert blob == (DATA / rel).read_bytes() and sha(blob) == original[rel]['sha256']
            lb = (DATA / info['label_relative']).read_bytes()
            assert sha(lb) == original[info['label_relative']]['sha256']
            boxes = [[int(t[0]), *map(float, t[1:])] for t in (line.split() for line in lb.decode('utf-8-sig').splitlines()) if t]
            assert boxes == info['boxes'] and len(boxes) == info['box_count']
        result['checks']['embedded_original_bytes_and_labels'] = 'passed'
        listed = (HERE / 'val_use_list_excluding_16.txt').read_text(encoding='utf-8').splitlines()
        expected_val = [str(DATA / p) for p in all_images['val'] if p not in excluded_val]
        assert listed == expected_val and len(listed) == 966
        result['checks']['val966_list_only_original_split_preserved'] = 'passed'
        actual = {p.relative_to(DATA).as_posix() for s in ('train', 'val') for folder in ('images', 'labels')
                  for p in (DATA / s / folder).iterdir() if p.is_file()}
        wanted = {p for p in original if p.startswith(('train/', 'val/'))}
        assert actual == wanted and len(actual) == 17668
        for rel in wanted:
            blob = (DATA / rel).read_bytes()
            assert len(blob) == original[rel]['bytes'] and sha(blob) == original[rel]['sha256']
        result['checks']['all_17668_data_files_unchanged'] = 'passed'
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'))
        for path in HERE.iterdir():
            if path.is_file() and path.suffix in ('.py', '.html', '.json', '.md', '.txt') and path.name != 'verification.json':
                value = path.read_text(encoding='utf-8')
                assert '\ufffd' not in value and '\x00' not in value
                assert all(line.rstrip() == line for line in value.splitlines())
        assert "connect-src 'none'" in html
        result['checks']['ast_utf8_whitespace_and_network_csp'] = 'passed'
        result['sample_id'] = meta['sample_id']
        result['artifacts'] = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
                               for p in sorted(HERE.iterdir()) if p.is_file() and p.name != 'verification.json']
        result['status'] = 'passed'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        raise
    finally:
        result['ended'] = datetime.now().astimezone().isoformat()
        save('verification.json', result)
    print(json.dumps({'status': result['status'], 'checks': result['checks']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    assert Path(sys.executable).resolve() == Path(r'E:\anaconda\envs\dl-3lc\python.exe').resolve()
    verify() if '--verify' in sys.argv[1:] else build()
