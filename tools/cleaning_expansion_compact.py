"""Print deduplicated candidate text; reviewers MUST separately read actual images."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT/'reports/cleaning-expansion-080-070-060-20261029/review_v3'


def compact(manifest):
    by_key = {r['key']: r for r in manifest['candidates']}
    images = []
    fields = ('key', 'signal', 'xyxy', 'confidence', 'predicted_class', 'original_class',
              'gt_xyxy', 'stable_target_id', 'd085_target_status')
    for evidence in manifest['evidence']:
        rows = [by_key[key] for key in evidence['candidate_keys']]
        if any(r['image_path'] != evidence['path'] or r['image_sha256'] != evidence['sha256'] or
               r['gt_boxes'] != rows[0]['gt_boxes'] for r in rows):
            raise ValueError('Candidate image/evidence identity mismatch')
        images.append(dict(evidence_id=evidence['id'], path=evidence['path'], sha256=evidence['sha256'],
                           gt_boxes=rows[0]['gt_boxes'],
                           candidates=[{k: r[k] for k in fields if k in r} for r in rows]))
    if [key for image in images for key in [r['key'] for r in image['candidates']]] != manifest['candidate_keys']:
        raise ValueError('Compact context reordered candidate keys')
    return dict(batch_id=manifest['batch_id'], images=images, coordinate_space='original 640x640 identity',
                actual_images_must_be_read_separately=True,
                note='Text context only, not visual evidence or review completion')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch_id')
    args = parser.parse_args()
    path = (STORE/'inputs'/args.batch_id/'input.json').resolve()
    if not path.is_relative_to((STORE/'inputs').resolve()): raise ValueError('Input path outside store')
    data = compact(json.loads(path.read_text(encoding='utf-8')))
    data['input_manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(data, ensure_ascii=True, separators=(',', ':')))
