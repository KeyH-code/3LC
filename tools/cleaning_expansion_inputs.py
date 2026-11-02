"""Prepare small agent-first original-image manifests; no model calls or label writes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/cleaning-expansion-080-070-060-20261029'


def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def subset(manifest, keys, batch_id):
    if not keys or len(keys) != len(set(keys)) or not set(keys) <= set(manifest['candidate_keys']):
        raise ValueError('Subset must contain unique assigned uncompleted keys')
    rows = [r for r in manifest['candidates'] if r['key'] in keys]
    evidence = [dict(e, candidate_keys=[k for k in e['candidate_keys'] if k in keys])
                for e in manifest['evidence'] if set(e['candidate_keys']) & set(keys)]
    return dict(manifest, batch_id=batch_id, parent_chunk_id=manifest['batch_id'],
                candidate_keys=[r['key'] for r in rows], candidates=rows, evidence=evidence,
                subset_note='Only previously uncompleted keys; completed opinions must not be repeated')


def build(root, output, candidates, parents, max_items=12):
    if output.exists(): raise FileExistsError('Never overwrite existing review input')
    if max_items < 1: raise ValueError('Invalid chunk size')
    by_key = {r['key']: r for r in candidates}
    assigned = [k for parent in parents for k in parent['candidate_keys']]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(by_key):
        raise ValueError('Parent batch coverage mismatch')
    prepared = []
    for parent in parents:
        rows = [by_key[k] for k in parent['candidate_keys']]
        for start in range(0, len(rows), max_items):
            chunk = rows[start:start+max_items]
            evidence = []
            for image_path in dict.fromkeys(r['image_path'] for r in chunk):
                image = (root/image_path).resolve()
                if not image.is_relative_to(root.resolve()): raise ValueError('Image outside project')
                same = [r for r in chunk if r['image_path'] == image_path]
                digest = sha(image)
                if any(r['image_sha256'] != digest or r['band'] != parent['band'] for r in same):
                    raise ValueError('Source image or band drift')
                evidence.append(dict(id=f'original-{len(evidence)+1:02}', path=image_path, sha256=digest,
                                     candidate_keys=[r['key'] for r in same], kind='original',
                                     source_image_sha256=digest))
            batch = f"{parent['batch_id']}-part{start//max_items+1:02}"
            manifest = dict(schema='cleaning-expansion-input-v1', batch_id=batch,
                            parent_batch_id=parent['batch_id'], band=parent['band'],
                            candidate_keys=[r['key'] for r in chunk], candidates=chunk, evidence=evidence,
                            coordinate_space=dict(width=640, height=640, transform='identity'),
                            original_image_only=True)
            prepared.append((batch, manifest))
    for batch, manifest in prepared:
        save(output/batch/'input.json', manifest)
    return len(prepared)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-items', type=int, default=12)
    args = p.parse_args()
    count = build(ROOT, OUT/'review_v3/inputs', load(OUT/'candidates.json')['rows'],
                  load(OUT/'batches.json')['rows'], args.max_items)
    print(f'Prepared {count} original-image chunks; no visual decisions')
