"""Assemble explicit reviewer decisions with manifest metadata, never visual inference.

seen_evidence_ids MUST be authored by the reviewer after actual read-image receipt.
This is self-reported vision, not independent host verification.
"""
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('feedback_review_receiver', ROOT/'tools/cleaning_expansion_review.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def assemble(manifest_path, decisions, attempt_id):
    manifest = review.load(manifest_path)
    seen = decisions['seen_evidence_ids']
    planned = {v['id']: v for v in manifest['evidence']}
    review.check(len(seen) == len(set(seen)) and set(seen) <= set(planned), 'Invalid explicit seen evidence IDs')
    rows = decisions['results']
    done = [r['key'] for r in rows]
    uncompleted = decisions['uncompleted_keys']
    review.check(len(done+uncompleted) == len(set(done+uncompleted)) and
                 set(done+uncompleted) == set(manifest['candidate_keys']), 'Opinions must partition assigned keys')
    for row in rows:
        for ident in row['evidence_ids']:
            review.check(ident in seen and row['key'] in planned[ident]['candidate_keys'],
                         'Opinion not bound to explicitly seen evidence')
        for field in ('reason', 'notes', 'object_key'):
            text = row[field]
            review.check(text is None or (text.isascii() and '???' not in text), 'Use readable ASCII handwritten text')
    inputs = [{k: planned[ident][k] for k in ('id', 'path', 'sha256', 'candidate_keys')} |
              dict(tool='read', actual_image_received=True) for ident in seen]
    data = dict(schema_version=review.SCHEMA, batch_id=manifest['batch_id'], attempt_id=attempt_id,
                requested_model=review.MODEL, reported_model=decisions.get('reported_model'),
                input_manifest_sha256=review.sha(manifest_path), started_at=decisions['started_at'],
                finished_at=decisions['finished_at'], vision_inputs=inputs, results=rows,
                uncompleted_keys=uncompleted)
    review.validate_schema(data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--decisions', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    store = review.OUT/'review_v3'
    manifest = review.local(ROOT, store/'inputs'/args.batch/'input.json')
    decisions = review.load(review.local(ROOT, args.decisions))
    data = assemble(manifest, decisions, args.attempt)
    output = review.local(ROOT, store/'raw'/(args.attempt+'.json'))
    review.save(output, data)
    review.check(review.load(output) == data, 'Saved feedback differs from authored opinion')
    print(dict(path=output.relative_to(ROOT).as_posix(), sha256=review.sha(output),
               results=len(data['results']), uncompleted=len(data['uncompleted_keys'])))
