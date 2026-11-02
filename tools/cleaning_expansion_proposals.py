"""Export three immutable proposals after FULL review only. No apply or training."""
import argparse
import collections
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('review_for_export', ROOT/'tools/cleaning_expansion_review.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def export(store, candidates, output):
    output = Path(output)
    if output.exists(): raise FileExistsError('Frozen proposals cannot be overwritten')
    data = review.proposals(store, candidates)
    state = data.pop('review_snapshot')
    output.mkdir(parents=True)
    review.save(output/'review_snapshot.json', state)
    snapshot_sha = review.sha(output/'review_snapshot.json')
    review.save(output/'conflicts.json', dict(conflicts=data['conflicts'], excluded_keys=data['excluded_keys']))
    files = ['review_snapshot.json', 'conflicts.json']
    summary = {}
    for band, proposal in data['bands'].items():
        proposal = dict(proposal, version=band, reference_dataset='D085',
                        review_snapshot_sha256=snapshot_sha, label_application_authorized=False,
                        training_authorized=False)
        proposal['incremental_action_types'] = dict(collections.Counter(r['category'] for r in proposal['incremental_actions']))
        proposal['cumulative_action_types'] = dict(collections.Counter(r['category'] for r in proposal['cumulative_actions']))
        review.save(output/(band+'.json'), proposal)
        files.append(band+'.json')
        summary[band] = {key: value for key, value in proposal.items() if key not in ('incremental_actions', 'cumulative_actions')}
    review.save(output/'summary.json', summary)
    files.append('summary.json')
    seal = dict(schema='cleaning-expansion-final-proposal-seal-v1', review_snapshot_sha256=snapshot_sha,
                candidate_count=len(candidates), completed=state['completed'],
                files={name: review.sha(output/name) for name in files},
                label_application_authorized=False, training_authorized=False,
                stop='A: exact proposal hashes require user approval before label application')
    review.save(output/'seal.json', seal)
    return seal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = review.local(ROOT, args.output)
    result = export(review.OUT/'review_v3', review.load(review.OUT/'candidates.json')['rows'], output)
    print(result)
