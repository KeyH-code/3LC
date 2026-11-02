"""Append-only agent review receiver. Deliberately has no label apply/train action."""
import argparse
import collections
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/cleaning-expansion-080-070-060-20261029'
MODEL = 'cpa/gpt-6-luna-high'
SCHEMA = 'cleaning-expansion-agent-review-v1'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(data)


def local(root, path):
    root = Path(root).resolve()
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path outside project: ' + str(path))
    return path


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_schema(value, schema=None, document=None):
    """Validate the explicitly supported subset of the frozen feedback schema."""
    if schema is None:
        schema = load(ROOT / 'docs/plans/cleaning-expansion-review.schema.json')
    document = document or schema
    if '$ref' in schema:
        target = document
        for part in schema['$ref'].split('/')[1:]:
            target = target[part]
        validate_schema(value, target, document)
    if 'const' in schema:
        check(type(value) is type(schema['const']) and value == schema['const'], 'Schema const mismatch')
    if 'enum' in schema:
        check(any(type(value) is type(option) and value == option for option in schema['enum']), 'Schema enum mismatch')
    if 'type' in schema:
        types = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        matches = dict(object=isinstance(value, dict), array=isinstance(value, list),
                       string=isinstance(value, str), boolean=type(value) is bool,
                       number=type(value) in (int, float) and math.isfinite(value), null=value is None)
        check(any(matches[t] for t in types), 'Schema type mismatch')
    if isinstance(value, dict):
        check(set(schema.get('required', [])) <= set(value), 'Missing schema field')
        properties = schema.get('properties', {})
        if schema.get('additionalProperties') is False:
            check(set(value) <= set(properties), 'Unknown schema field')
        for key in value.keys() & properties.keys():
            validate_schema(value[key], properties[key], document)
    if isinstance(value, str):
        check(schema.get('minLength', 0) <= len(value) <= schema.get('maxLength', math.inf), 'Schema text length')
        if 'pattern' in schema:
            check(re.fullmatch(schema['pattern'], value) is not None, 'Schema pattern')
        if schema.get('format') == 'date-time':
            try:
                check(datetime.fromisoformat(value).tzinfo is not None, 'Timestamp requires timezone')
            except (ValueError, TypeError):
                raise ValueError('Invalid timestamp') from None
    if isinstance(value, list):
        check(schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', math.inf), 'Schema array length')
        if schema.get('uniqueItems'):
            check(len({json.dumps(v, sort_keys=True) for v in value}) == len(value), 'Duplicate array item')
        for item in value:
            if 'items' in schema:
                validate_schema(item, schema['items'], document)
    if type(value) in (int, float):
        check(math.isfinite(value) and schema.get('minimum', -math.inf) <= value <= schema.get('maximum', math.inf),
              'Schema number range')
    if 'oneOf' in schema:
        passed = 0
        for option in schema['oneOf']:
            try:
                validate_schema(value, option, document)
                passed += 1
            except ValueError:
                pass
        check(passed == 1, 'Schema oneOf mismatch')
    for rule in schema.get('allOf', []):
        try:
            validate_schema(value, rule['if'], document)
        except ValueError:
            continue
        validate_schema(value, rule['then'], document)


def snapshot(store, candidates):
    by_key = {row['key']: row for row in candidates}
    rows = {}
    blocked = {}
    holds = {}
    held_rows = []
    releases = {}
    for path in sorted((Path(store)/'quality_releases').glob('*.json')):
        release = load(path)
        hold_path = local(Path(store)/'quality_holds', release['hold_file'])
        check(sha(hold_path) == release['hold_sha256'], 'Released hold hash drift')
        approval_path = local(store, release['approval_path'])
        check(sha(approval_path) == release['approval_sha256'], 'Release approval hash drift')
        approval = load(approval_path)
        check(approval.get('approved_by_user') is True and
              approval.get('restore_retrospective_visual_holds') is True, 'Missing scoped user visual approval')
        hold = load(hold_path)
        check(hold['event_sha256'] == release['event_sha256'], 'Released event identity differs')
        releases[hold_path.name] = release
    for path in sorted((Path(store)/'quality_holds').glob('*.json')):
        hold = load(path)
        name = hold['event_file']
        check(Path(name).name == name, 'Unsafe held event path')
        event_path = Path(store)/'accepted'/name
        check(sha(event_path) == hold['event_sha256'], 'Quality held event hash drift')
        if path.name not in releases:
            holds[name] = hold
    for path in sorted((Path(store) / 'accepted').glob('*.json')):
        event = load(path)
        if path.name in holds:
            held_rows.extend(event['rows'])
            continue
        for row in event['rows']:
            if row['category'] == 'blocked':
                blocked[row['key']] = row
                continue
            check(row['key'] in by_key and row['key'] not in rows, 'Ledger duplicate/unknown key')
            rows[row['key']] = row
    ordered = [rows[c['key']] for c in candidates if c['key'] in rows]
    outstanding_blocked = [r for k, r in blocked.items() if k not in rows]
    held_keys = sorted({r['key'] for r in held_rows if r['key'] not in rows})
    return dict(rows=ordered, completed=len(ordered), total=len(candidates),
                quality_held=len(held_keys), quality_held_keys=held_keys,
                blocked=len(outstanding_blocked), blocked_rows=outstanding_blocked,
                uncompleted_keys=[c['key'] for c in candidates if c['key'] not in rows])


def stable_target(candidate):
    parts = [candidate['id'], candidate['label_sha256'], candidate['d085_gt_index'],
             candidate['original_label_line']]
    return hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).hexdigest()


def target_binding(root, candidate):
    label = local(root, candidate['label_path'])
    check(sha(label) == candidate['label_sha256'] == candidate['reference_label_sha256'], 'D085 label hash drift')
    lines = label.read_text(encoding='utf-8').splitlines()
    index = candidate['d085_gt_index']
    check(type(index) is int and 0 <= index < len(lines), 'D085 target index invalid')
    check(lines[index] == candidate['original_label_line'], 'D085 original target line differs')
    check(int(lines[index].split()[0]) == candidate['original_class'], 'D085 original class mismatch')
    check(stable_target(candidate) == candidate['stable_target_id'], 'D085 stable target identity mismatch')
    check(candidate['d085_target_status'] == 'unchanged_original', 'Protected D085 approved action')
    return {field: candidate[field] for field in ('stable_target_id', 'reference_label_sha256',
            'd085_gt_index', 'original_label_line', 'd085_target_status')}


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    areas = [(box[2]-box[0])*(box[3]-box[1]) for box in (a, b)]
    return intersection / (sum(areas)-intersection), intersection / min(areas)


def proposals(store, candidates):
    state = snapshot(store, candidates)
    check(not state['uncompleted_keys'], 'Unfinished visual review; final proposals forbidden')
    by_key = {r['key']: r for r in candidates}
    actions = [r for r in state['rows'] if r['category'] in ('add_box', 'change_class')]
    conflicts = []
    excluded = set()
    for i, action in enumerate(actions):
        candidate = by_key[action['key']]
        if action['category'] == 'change_class':
            target = action['target_binding']['stable_target_id']
            keep_keys = [r['key'] for r in state['rows'] if r['category'] == 'keep' and
                         by_key[r['key']].get('stable_target_id') == target]
            if keep_keys:
                conflicts.append(dict(keys=[action['key']] + keep_keys, reason='same_GT_keep_vs_change'))
                excluded.add(action['key'])
        if action['category'] == 'add_box':
            for gt in candidate['gt_boxes']:
                iou, fraction = overlap(action['final_xyxy'], gt['xyxy'])
                if iou >= .5 or fraction >= .8:
                    conflicts.append(dict(keys=[action['key']], reason='overlaps_D085_existing_target'))
                    excluded.add(action['key'])
                    break
        for previous in actions[:i]:
            other = by_key[previous['key']]
            if candidate['id'] != other['id']:
                continue
            reason = None
            if action['category'] == previous['category'] == 'change_class':
                if action['target_binding']['stable_target_id'] == previous['target_binding']['stable_target_id']:
                    reason = 'same_GT_multiple_opinions'
            elif action['category'] == previous['category'] == 'add_box':
                iou, fraction = overlap(action['final_xyxy'], previous['final_xyxy'])
                if action['object_key'] == previous['object_key'] or iou >= .5 or fraction >= .8:
                    reason = 'same_object_or_overlapping_additions'
            elif {action['category'], previous['category']} == {'add_box', 'change_class'}:
                addition = action if action['category'] == 'add_box' else previous
                changed_candidate = candidate if action['category'] == 'change_class' else other
                iou, fraction = overlap(addition['final_xyxy'], changed_candidate['gt_xyxy'])
                if iou >= .5 or fraction >= .8:
                    reason = 'addition_vs_class_change_target'
            if reason:
                keys = [previous['key'], action['key']]
                conflicts.append(dict(keys=keys, reason=reason))
                excluded.update(keys)
    safe = [dict(r, band=by_key[r['key']]['band'], image_id=by_key[r['key']]['id'],
                 image_path=by_key[r['key']]['image_path'], label_path=by_key[r['key']].get('label_path'),
                 scene=by_key[r['key']]['scene']) for r in actions if r['key'] not in excluded]
    bands = {}
    cumulative = []
    for band in ('D080', 'D070', 'D060'):
        incremental = [r for r in safe if r['band'] == band]
        cumulative = cumulative + incremental
        reviewed = [r for r in state['rows'] if by_key[r['key']]['band'] == band]
        bands[band] = dict(incremental_actions=incremental, cumulative_actions=cumulative,
                           decisions=dict(collections.Counter(r['category'] for r in reviewed)),
                           incremental_files=len({r['label_path'] for r in incremental}),
                           cumulative_files=len({r['label_path'] for r in cumulative}),
                           proposed_classes=dict(collections.Counter(str(r['proposed_class']) for r in incremental)),
                           scenes=dict(collections.Counter(r['scene'] for r in incremental)))
    return dict(schema='cleaning-expansion-proposals-v1', bands=bands, conflicts=conflicts,
                excluded_keys=sorted(excluded), review_snapshot=state,
                label_application_authorized=False, training_authorized=False,
                stop='A: user must approve exact proposal hashes before any label write')


def progress(store, candidates):
    state = snapshot(store, candidates)
    by_key = {r['key']: r for r in candidates}
    bands = {}
    for band in ('D080', 'D070', 'D060'):
        assigned = [r for r in candidates if r['band'] == band]
        completed = [r for r in state['rows'] if by_key[r['key']]['band'] == band]
        blocked = [r for r in state['blocked_rows'] if by_key[r['key']]['band'] == band]
        bands[band] = dict(total=len(assigned), completed=len(completed), blocked=len(blocked),
                           unreviewed=len(assigned)-len(completed)-len(blocked),
                           decisions=dict(collections.Counter(r['category'] for r in completed)))
    return dict(total=len(candidates), completed=state['completed'], blocked=state['blocked'],
                quality_held=state['quality_held'],
                uncompleted=len(state['uncompleted_keys']), bands=bands,
                proposal_ready=not state['uncompleted_keys'],
                label_application_authorized=False, training_authorized=False)


@contextmanager
def serial_receiver(store):
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    path = store / 'receive.lock'
    try:
        lock = path.open('x', encoding='utf-8')
    except FileExistsError:
        raise ValueError('Receiver locked; inspect active process/stale lock before resuming') from None
    try:
        with lock:
            lock.write(str(os.getpid()))
            lock.flush()
            yield
    finally:
        path.unlink()


def receive(root, store, candidates, manifest_path, feedback_path, transport):
    with serial_receiver(store):
        return receive_locked(root, store, candidates, manifest_path, feedback_path, transport)


def collect_run(root, store, candidates, manifest_path, raw_path, session_id):
    """Collect a completed subagent's saved opinion; receipt basis stays self-report."""
    data = load(raw_path)
    validate_schema(data)
    transport = dict(session_id=session_id, requested_model=MODEL, reported_model=data['reported_model'],
                     raw_path=Path(raw_path).resolve().relative_to(Path(root).resolve()).as_posix(),
                     raw_sha256=sha(raw_path), vision_receipts=data['vision_inputs'],
                     receipt_basis='agent_tool_report_after_capability_check')
    path = Path(store)/'transports'/(data['attempt_id']+'.json')
    if path.exists():
        check(load(path) == transport, 'Existing transport differs; do not overwrite')
    else:
        save(path, transport)
    return receive(root, store, candidates, manifest_path, raw_path, transport)


def receive_locked(root, store, candidates, manifest_path, feedback_path, transport):
    root = Path(root)
    manifest_path = local(root, manifest_path)
    feedback_path = local(root, feedback_path)
    data = load(feedback_path)
    validate_schema(data)
    check(datetime.fromisoformat(data['started_at']) <= datetime.fromisoformat(data['finished_at']), 'Reversed timestamps')
    manifest = load(manifest_path)
    check(data['schema_version'] == SCHEMA, 'Unknown review schema')
    check(data['requested_model'] == MODEL == transport['requested_model'], 'Requested model mismatch')
    check(data['reported_model'] == transport['reported_model'], 'Reported model differs from transport')
    check(data['reported_model'] in (None, MODEL), 'Actual model mismatch; batch stopped')
    check(bool(transport['session_id']), 'Missing session provenance')
    raw_path = local(root, transport['raw_path'])
    check(sha(raw_path) == transport['raw_sha256'], 'Raw response drift')
    raw = raw_path.read_text(encoding='utf-8')
    try:
        raw_value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        check(match is not None, 'Raw response does not contain a JSON opinion')
        raw_value = json.loads(match[1])
    check(raw_value == data, 'Feedback differs from original agent opinion')
    check(data['input_manifest_sha256'] == sha(manifest_path), 'Input manifest drift')
    check(data['batch_id'] == manifest['batch_id'], 'Batch identity mismatch')
    by_key = {row['key']: row for row in candidates}
    keys = manifest['candidate_keys']
    check(len(keys) == len(set(keys)) and set(keys) <= set(by_key), 'Invalid assigned keys')
    if 'candidates' in manifest:
        check(manifest['candidates'] == [by_key[key] for key in keys], 'Input differs from frozen candidates')
    results = data['results']
    result_keys = [row['key'] for row in results]
    uncompleted = data['uncompleted_keys']
    check(len(result_keys) == len(set(result_keys)), 'Duplicate result key')
    check(len(uncompleted) == len(set(uncompleted)), 'Duplicate uncompleted key')
    check(not set(result_keys) & set(uncompleted) and set(result_keys + uncompleted) == set(keys),
          'Results/uncompleted must partition assigned batch')
    state = snapshot(store, candidates)
    check(not set(result_keys) & {r['key'] for r in state['rows']}, 'Repeated import of candidate opinion')
    inputs = {v['id']: v for v in data['vision_inputs']}
    check(len(inputs) == len(data['vision_inputs']), 'Duplicate evidence ID')
    receipts = {r['id']: r for r in transport.get('vision_receipts', [])}
    check(transport.get('receipt_basis') == 'agent_tool_report_after_capability_check',
          'Only agent tool report evidence available; no independent host receipt')
    for ident, evidence in inputs.items():
        check(ident in receipts and receipts[ident] == evidence, 'Image receipt differs from collected tool report')
    planned = {v['id']: v for v in manifest['evidence']}
    for ident, evidence in inputs.items():
        check(ident in planned, 'Unplanned evidence')
        bound = planned[ident]
        for field in ('path', 'sha256', 'candidate_keys'):
            check(evidence[field] == bound[field], 'Evidence manifest mismatch: ' + field)
        check(sha(local(root, evidence['path'])) == evidence['sha256'], 'Evidence bytes drift')
        for key in evidence['candidate_keys']:
            check(key in keys, 'Evidence outside assigned batch')
            candidate = by_key[key]
            check(bound['kind'] == 'original', 'Unsupported derived evidence')
            check(local(root, evidence['path']) == local(root, candidate['image_path']) and
                  evidence['sha256'] == candidate['image_sha256'], 'Wrong image for candidate')
    accepted = []
    for result in results:
        for field in ('reason', 'notes'):
            text = result[field]
            check('???' not in text and '\ufffd' not in text and
                  (not text or any(character.isalnum() for character in text)),
                  'Unreadable feedback text; preserve raw and recover original explanation')
        if result['category'] == 'add_box':
            x1, y1, x2, y2 = result['final_xyxy']
            check(x1 < x2 and y1 < y2, 'Invalid box extent')
        for ident in result['evidence_ids']:
            check(ident in inputs and result['key'] in inputs[ident]['candidate_keys'] and
                  inputs[ident]['actual_image_received'], 'Visual evidence not received/bound')
        candidate = by_key[result['key']]
        binding = None
        if result['category'] == 'change_class':
            check(candidate['signal'] == 'class_disagreement', 'Cannot change unbound GT')
            check(result['proposed_class'] != candidate['original_class'], 'Proposed class is unchanged')
            binding = target_binding(root, candidate)
        if result['category'] == 'add_box':
            check(candidate['signal'] == 'prediction_without_label_coverage', 'Addition not in missing-label scope')
        row = dict(result, source='agent_adopted', yolo_confidence=candidate['confidence'],
                   requested_model=MODEL, reported_model=data['reported_model'],
                   session_id=transport['session_id'], batch_id=data['batch_id'], attempt_id=data['attempt_id'],
                   input_manifest_sha256=data['input_manifest_sha256'])
        row['target_binding'] = binding
        accepted.append(row)
    event = dict(schema='cleaning-expansion-review-event-v1', rows=accepted, feedback=data, raw_response=raw,
                 feedback_sha256=sha(feedback_path), manifest=manifest, transport=transport)
    attempt = data['attempt_id']
    check(re.fullmatch(r'[A-Za-z0-9_-]{1,120}', attempt), 'Unsafe attempt identity')
    pending = Path(store) / 'pending' / (attempt + '.json')
    destination = Path(store) / 'accepted' / (attempt + '.json')
    check(not destination.exists(), 'Attempt already committed')
    if pending.exists():
        check(load(pending) == event, 'Interrupted event differs; retain and use new attempt identity')
    else:
        save(pending, event)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Windows rename fails rather than replacing an existing destination.
    os.rename(pending, destination)
    return snapshot(store, candidates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'collect'])
    parser.add_argument('--batch')
    parser.add_argument('--attempt')
    parser.add_argument('--session')
    args = parser.parse_args()
    if args.action == 'collect':
        check(all((args.batch, args.attempt, args.session)), 'collect requires batch/attempt/session')
        for name in (args.batch, args.attempt):
            check(re.fullmatch(r'[A-Za-z0-9_-]{1,120}', name), 'Unsafe collection identity')
        store = OUT/'review_v3'
        collect_run(ROOT, store, load(OUT/'candidates.json')['rows'], store/'inputs'/args.batch/'input.json',
                    store/'raw'/(args.attempt+'.json'), args.session)
    state = progress(OUT / 'review_v3', load(OUT / 'candidates.json')['rows'])
    print(json.dumps(state, indent=2))
