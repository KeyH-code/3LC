"""Serially collect explicit completed-run receipts. No watching, model calls or apply."""
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('expansion_review_receiver', ROOT/'tools/cleaning_expansion_review.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def collect_receipt(root, store, candidates, receipt):
    review.check(receipt['requested_model'] == review.MODEL, 'Receipt requested model mismatch')
    review.check(receipt['reported_model'] in (None, review.MODEL), 'Receipt actual model mismatch')
    for name in ('batch_id', 'attempt_id'):
        review.check(re.fullmatch(r'[A-Za-z0-9_-]{1,120}', receipt[name]), 'Unsafe receipt identity')
    review.check(receipt['reviewer_session_id'].startswith('ses_'), 'Missing reviewer session identity')
    raw = review.local(root, receipt['raw_path'])
    review.check(review.sha(raw) == receipt['raw_sha256'], 'Receipt/raw hash mismatch')
    accepted = Path(store)/'accepted'/(receipt['attempt_id']+'.json')
    if accepted.exists():
        event = review.load(accepted)
        review.check(event['transport']['session_id'] == receipt['reviewer_session_id'], 'Receipt session differs')
        review.check(event['transport']['raw_sha256'] == receipt['raw_sha256'] == event['feedback_sha256'],
                     'Committed event hash differs')
        review.check(event['feedback']['batch_id'] == receipt['batch_id'] and
                     event['feedback']['attempt_id'] == receipt['attempt_id'], 'Committed event identity differs')
        return 'already_accepted'
    data = review.load(raw)
    review.check(data['batch_id'] == receipt['batch_id'] and data['attempt_id'] == receipt['attempt_id'],
                 'Receipt/batch identity mismatch')
    review.check(data['reported_model'] == receipt['reported_model'], 'Reported model differs from receipt')
    for row in data['results']:
        for name in ('reason', 'notes', 'object_key'):
            review.check(row[name] is None or row[name].isascii(), 'New receipt requires ASCII handwritten text')
    review.collect_run(root, store, candidates, Path(store)/'inputs'/receipt['batch_id']/'input.json',
                       raw, receipt['reviewer_session_id'])
    return 'accepted'


if __name__ == '__main__':
    store = review.OUT/'review_v3'
    candidates = review.load(review.OUT/'candidates.json')['rows']
    statuses = []
    for path in sorted((store/'full-review/receipts').glob('*.json')):
        receipt = review.load(path)
        status = collect_receipt(ROOT, store, candidates, receipt)
        statuses.append(dict(receipt=path.name, status=status))
    print(json.dumps(dict(receipts=statuses, progress=review.progress(store, candidates)), indent=2))
