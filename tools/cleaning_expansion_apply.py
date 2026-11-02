"""Apply hash-approved expansion proposals to independent copies; no model calls."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


review = module('expansion_apply_review', ROOT/'tools/cleaning_expansion_review.py')
labels = module('expansion_apply_labels', ROOT/'tools/oof_label_cleaning.py')


def edit(raw, actions):
    changes = []
    for action in actions:
        change = dict(action=action['category'], new_class=action['proposed_class'])
        if action['category'] == 'change_class':
            target = action['target_binding']
            change.update(gt_index=target['d085_gt_index'],
                          original_label_line=target['original_label_line'],
                          old_class=int(target['original_label_line'].split()[0]))
        elif action['category'] == 'add_box':
            change['final_xyxy'] = action['final_xyxy']
        else:
            raise ValueError('Unsupported approved action')
        changes.append(change)
    return labels.edited_label(raw, changes)


def authorize(root, proposals, approval, reference):
    proposals, approval, reference = [review.local(root, p) for p in (proposals, approval, reference)]
    a = review.load(approval)
    seal = review.load(proposals/'seal.json')
    review.check(a.get('approved_by_user') is True and bool(a.get('user_message')) and
                 bool(a.get('source_reference')), 'Approval requires explicit user provenance')
    review.check(a.get('proposal_seal_sha256') == review.sha(proposals/'seal.json'), 'Approval seal mismatch')
    review.check(a.get('deferred_policy') == 'leave_unchanged' and
                 a.get('conflict_policy') == 'exclude' and a.get('training_authorized') is False,
                 'Approval scope must preserve deferred/conflict/training gates')
    review.check(a.get('reference_manifest_sha256') == review.sha(reference/'dataset_manifest.json'),
                 'Approval reference manifest mismatch')
    for name, digest in seal['files'].items():
        review.check(review.sha(review.local(proposals, name)) == digest, 'Frozen proposal hash drift')
    review.check(a.get('proposal_sha256') == {b:seal['files'][b+'.json'] for b in ('D080','D070','D060')},
                 'Approval proposal hash mismatch')
    review.check(seal['completed'] == seal['candidate_count'], 'Incomplete review')
    return a, seal


def apply(root, proposals, approval, reference, val_manifest, output):
    root = Path(root).resolve()
    proposals, approval, reference, val_manifest, output = [review.local(root, p) for p in
        (proposals, approval, reference, val_manifest, output)]
    a, seal = authorize(root, proposals, approval, reference)
    if output.exists():
        raise FileExistsError('Never overwrite a dataset application or failed attempt')
    source = review.load(reference/'dataset_manifest.json')
    train = sorted(source['rows'], key=lambda row:Path(row['image']).name)
    validation = review.load(val_manifest)
    review.check(bool(train) and bool(validation), 'Empty source split')
    ids = [r['id'] for r in train]
    review.check(len(set(ids)) == len(ids), 'Duplicate train ID')
    for rows in (train, validation):
        for field in ('image','label'):
            review.check(len({Path(r[field]).name for r in rows}) == len(rows), 'Duplicate source basename')
        review.check(all(Path(r['image']).stem == Path(r['label']).stem for r in rows), 'Unpaired source names')
    review.check(not {r['image_sha256'] for r in train} & {r['image_sha256'] for r in validation},
                 'Train/val image overlap')
    pinned = {str((reference/'dataset_manifest.json').relative_to(root)): review.sha(reference/'dataset_manifest.json'),
              str(val_manifest.relative_to(root)): review.sha(val_manifest)}
    if a.get('experiment_spec_sha256'):
        spec = root/'reports/cleaning-expansion-080-070-060-20261029/experiment_spec.json'
        review.check(review.sha(spec) == a['experiment_spec_sha256'], 'Frozen experiment specification drift')
        pinned[spec.relative_to(root).as_posix()] = a['experiment_spec_sha256']
    original = {}
    base_classes = collections.Counter()
    size = 0
    for split, rows in [('train',train), ('val',validation)]:
        for row in rows:
            for field in ('image','label'):
                path = review.local(root, reference/row[field] if split == 'train' else row[field])
                review.check(review.sha(path) == row[field+'_sha256'], 'Source hash drift')
                pinned[path.relative_to(root).as_posix()] = row[field+'_sha256']
                size += path.stat().st_size
            raw = review.local(root, reference/row['label'] if split == 'train' else row['label']).read_bytes()
            if split == 'train':
                original[row['id']] = raw
                base_classes.update(class_counts(raw))
    review.check(sum(base_classes.values()) == source['boxes'], 'Reference box count drift')
    prepared = {}
    previous = []
    excluded = set(review.load(proposals/'conflicts.json')['excluded_keys'])
    by_id = {r['id']:r for r in train}
    for band in ('D080','D070','D060'):
        proposal = review.load(proposals/(band+'.json'))
        actions = proposal['cumulative_actions']
        review.check(proposal['review_snapshot_sha256'] == seal['review_snapshot_sha256'], 'Snapshot mismatch')
        review.check(actions == previous + proposal['incremental_actions'], 'Noncumulative proposals')
        previous = actions
        review.check(len({r['key'] for r in actions}) == len(actions) and
                     not excluded & {r['key'] for r in actions}, 'Duplicate/excluded action')
        grouped = collections.defaultdict(list)
        wanted = base_classes.copy()
        targets = set()
        for action in actions:
            ident = action['image_id']
            review.check(ident in by_id, 'Unknown train target')
            row = by_id[ident]
            review.check(review.local(root, action['label_path']) == review.local(root, reference/row['label']),
                         'Action label path mismatch')
            if action['category'] == 'change_class':
                binding = action['target_binding']
                target = (ident, binding['d085_gt_index'])
                review.check(target not in targets, 'Repeated GT change')
                targets.add(target)
                review.check(binding['reference_label_sha256'] == row['label_sha256'], 'GT reference hash drift')
                old = int(binding['original_label_line'].split()[0])
                review.check(old != action['proposed_class'], 'Unchanged class action')
                wanted[old] -= 1
            review.check(action['category'] in ('add_box','change_class'), 'Unsupported proposal action')
            wanted[action['proposed_class']] += 1
            grouped[ident].append(action)
        labels_by_id = {r['id']:edit(original[r['id']], grouped[r['id']]) for r in train}
        actual = collections.Counter()
        for raw in labels_by_id.values(): actual.update(class_counts(raw))
        changed = sum(labels_by_id[r['id']] != original[r['id']] for r in train)
        review.check(actual == wanted and changed == proposal['cumulative_files'], 'Applied counts differ from proposal')
        prepared[band] = dict(labels=labels_by_id, actions=actions, classes=dict(actual),
                              boxes=sum(actual.values()), changed_label_files=changed)
    required = size*3 + sum(len(raw) for v in prepared.values() for raw in v['labels'].values()) + 100_000_000
    review.check(shutil.disk_usage(root).free > required, 'Insufficient copy space')
    output.mkdir(parents=True)
    review.save(output/'approval_record.json', a)
    review.save(output/'source_seal.json', dict(files=pinned, proposals=str(proposals.relative_to(root)),
        proposal_seal_sha256=review.sha(proposals/'seal.json'), reference=str(reference.relative_to(root)),
        val_manifest=str(val_manifest.relative_to(root)), storage='independent full copies, no hardlinks',
        required_bytes_with_reserve=required))
    summary = {}
    for band, plan in prepared.items():
        destination = output/band
        manifest = []
        for split, rows in [('train',train), ('val',validation)]:
            for row in rows:
                image = destination/'dataset/images'/split/Path(row['image']).name
                label = destination/'dataset/labels'/split/Path(row['label']).name
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                source_image = review.local(root, reference/row['image'] if split == 'train' else row['image'])
                source_label = review.local(root, reference/row['label'] if split == 'train' else row['label'])
                with source_image.open('rb') as src, image.open('xb') as dst: shutil.copyfileobj(src, dst)
                raw = plan['labels'][row['id']] if split == 'train' else source_label.read_bytes()
                with label.open('xb') as stream: stream.write(raw)
                review.check(review.sha(image) == row['image_sha256'] and review.sha(label) == digest(raw),
                             'Copy hash mismatch')
                manifest.append(dict(id=row.get('id', Path(row['image']).stem), split=split,
                    image=image.relative_to(output).as_posix(), label=label.relative_to(output).as_posix(),
                    image_sha256=row['image_sha256'], label_sha256=digest(raw),
                    source_label=source_label.relative_to(root).as_posix(), source_label_sha256=row['label_sha256']))
        config = dict(path=str(destination/'dataset'), train='images/train', val='images/val',
                      nc=4, names=['truck','car','van','bus'])
        review.save(destination/'data.yaml', config)  # JSON is a YAML 1.2 subset.
        review.save(destination/'manifest.json', dict(rows=manifest, boxes=plan['boxes'], classes=plan['classes']))
        review.save(destination/'applied_changes.json', dict(proposal_sha256=seal['files'][band+'.json'],
            review_snapshot_sha256=seal['review_snapshot_sha256'], actions=plan['actions']))
        summary[band] = dict(boxes=plan['boxes'], classes=plan['classes'], changed_label_files=plan['changed_label_files'],
            actions=len(plan['actions']), added_boxes=sum(r['category']=='add_box' for r in plan['actions']),
            changed_classes=sum(r['category']=='change_class' for r in plan['actions']),
            train_images=len(train), val_images=len(validation), data_yaml=str(destination/'data.yaml'))
    files = ['approval_record.json','source_seal.json'] + [f'{b}/{n}' for b in prepared for n in
            ('data.yaml','manifest.json','applied_changes.json')]
    code = [Path(__file__), ROOT/'tools/oof_label_cleaning.py', ROOT/'tools/cleaning_expansion_review.py',
            ROOT/'docs/plans/cleaning-expansion-review.schema.json']
    review.save(output/'application_seal.json', dict(files={n:review.sha(output/n) for n in files},
        code_files={str(p.relative_to(ROOT)):review.sha(p) for p in code},
        experiment_spec_sha256=a.get('experiment_spec_sha256'),
        versions=summary, training_authorized=False))
    result = verify(root, output)
    review.save(output/'verification.json', result)
    review.save(output/'SUCCESS.json', dict(verified=True, verification_sha256=review.sha(output/'verification.json'),
        application_seal_sha256=review.sha(output/'application_seal.json'), training_authorized=False))
    return result


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def class_counts(raw):
    counts = collections.Counter()
    for line in raw.splitlines():
        parts = line.split()
        review.check(len(parts) == 5 and parts[0] in (b'0',b'1',b'2',b'3'), 'Malformed YOLO label')
        x, y, w, h = map(float, parts[1:])
        review.check(w > 0 and h > 0 and min(x-w/2,y-h/2) >= -1e-9 and
                     max(x+w/2,y+h/2) <= 1+1e-9, 'Invalid normalized geometry')
        counts[int(parts[0])] += 1
    return counts


def verify(root, output):
    """Read-only file hashes and separate per-line reconstruction of approved edits."""
    root, output = Path(root).resolve(), review.local(root, output)
    seal = review.load(output/'application_seal.json')
    for name, expected in seal['files'].items():
        review.check(review.sha(review.local(output,name)) == expected, 'Application metadata hash drift')
    source = review.load(output/'source_seal.json')
    for name, expected in source['files'].items():
        review.check(review.sha(review.local(root,name)) == expected, 'Protected source changed')
    proposals = review.local(root, source['proposals'])
    review.check(review.sha(proposals/'seal.json') == source['proposal_seal_sha256'], 'Proposal seal changed')
    authorize(root, proposals, output/'approval_record.json', review.local(root,source['reference']))
    reference = review.local(root,source['reference'])
    train = sorted(review.load(reference/'dataset_manifest.json')['rows'], key=lambda row:Path(row['image']).name)
    validation = review.load(review.local(root,source['val_manifest']))
    versions = {}
    for band in ('D080','D070','D060'):
        meta = review.load(output/band/'manifest.json')
        rows = meta['rows']
        actions = review.load(output/band/'applied_changes.json')['actions']
        review.check(actions == review.load(proposals/(band+'.json'))['cumulative_actions'], 'Applied action drift')
        grouped = collections.defaultdict(list)
        for action in actions: grouped[action['image_id']].append(action)
        counts = collections.Counter()
        changed = 0
        for split, original in [('train',train),('val',validation)]:
            assigned = [r for r in rows if r['split'] == split]
            review.check([Path(r['image']).name for r in assigned] == [Path(r['image']).name for r in original],
                         'Split membership/order mismatch')
            for field, ext in [('image','*.jpg'),('label','*.txt')]:
                folder = output/band/'dataset'/('images' if field=='image' else 'labels')/split
                review.check({p.name for p in folder.iterdir()} == {Path(r[field]).name for r in assigned},
                             'Extra/missing split files')
            for row, base in zip(assigned, original):
                for field in ('image','label'):
                    review.check(review.sha(review.local(output,row[field])) == row[field+'_sha256'], 'Dataset byte drift')
                review.check(row['image_sha256'] == base['image_sha256'], 'Copied image identity differs')
                src = review.local(root, reference/base['label'] if split=='train' else base['label'])
                raw = src.read_bytes()
                actual = review.local(output,row['label']).read_bytes()
                if split == 'val':
                    review.check(actual == raw, 'Validation label changed')
                    continue
                lines = raw.splitlines(keepends=True)
                additions = []
                for action in grouped[base['id']]:
                    if action['category']=='change_class':
                        target = action['target_binding'];i=target['d085_gt_index']
                        review.check(lines[i].rstrip(b'\r\n').decode()==target['original_label_line'], 'GT line mismatch')
                        lines[i] = re.sub(rb'^(\s*)[0-3](?=\s)',
                            lambda match:match[1]+str(action['proposed_class']).encode(), lines[i], count=1)
                    else: additions.append(action)
                prefix = b''.join(lines)
                newline = b'\r\n' if b'\r\n' in raw else b'\n'
                if additions and prefix and not prefix.endswith((b'\n',b'\r')): prefix += newline
                review.check(actual.startswith(prefix), 'Original line/geometry/spacing changed')
                appended = actual[len(prefix):].splitlines()
                review.check(len(appended)==len(additions), 'Added label count mismatch')
                for line, action in zip(appended, additions):
                    fields=line.split();x,y,w,h=map(float,fields[1:]);box=action['final_xyxy']
                    reconstructed=[(x-w/2)*640,(y-h/2)*640,(x+w/2)*640,(y+h/2)*640]
                    review.check(int(fields[0])==action['proposed_class'] and
                                 all(abs(a-b)<1e-6 for a,b in zip(reconstructed,box)), 'Added geometry/class differs')
                changed += raw != actual
                counts.update(class_counts(actual))
        wanted = seal['versions'][band]
        review.check(sum(counts.values())==wanted['boxes'] and changed==wanted['changed_label_files'] and
                     {str(k):v for k,v in counts.items()}==wanted['classes'], 'Dataset totals mismatch')
        versions[band] = wanted
    return dict(verified=True, at=review.datetime.now(review.timezone.utc).isoformat(), versions=versions,
                protected_source_files=len(source['files']), training_authorized=False, model_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['apply','verify'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--approval')
    args = parser.parse_args()
    base = review.OUT
    if args.action == 'verify':
        result = verify(ROOT, args.output)
    else:
        review.check(bool(args.approval), 'Approval path required')
        result = apply(ROOT, base/'review_v3/final-proposals-20261101', args.approval,
            ROOT/'reports/oof-label-cleaning-conf085-20261027',
            ROOT/'reports/train-cleaning-ab-ss04-20261029/val_manifest.json', args.output)
    print(json.dumps(result, indent=2))
