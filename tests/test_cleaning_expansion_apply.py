import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LabelApplicationTests(unittest.TestCase):
    def tool(self):
        path = ROOT/'tools/cleaning_expansion_apply.py'
        self.assertTrue(path.exists(), 'Missing approved-proposal label application tool')
        spec = importlib.util.spec_from_file_location('expansion_apply_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_same_file_class_edit_and_addition_preserve_original_bytes(self):
        m = self.tool()
        raw = b'  1\t0.5  0.5 0.1 0.2\r\n2 0.25 0.25 0.1 0.1'
        actions = [dict(key='a', category='change_class', proposed_class=3,
                        target_binding=dict(d085_gt_index=1,
                            original_label_line='2 0.25 0.25 0.1 0.1')),
                   dict(key='b', category='add_box', proposed_class=1,
                        final_xyxy=[0, 0, 64, 128])]
        result = m.edit(raw, actions)
        self.assertTrue(result.startswith(b'  1\t0.5  0.5 0.1 0.2\r\n3 0.25 0.25 0.1 0.1\r\n'))
        self.assertEqual(result.splitlines()[-1],
                         b'1 0.050000000000 0.100000000000 0.100000000000 0.200000000000')
        self.assertEqual(m.edit(raw, []), raw)

    def fixture(self, root, m):
        reference = root/'reference'
        rows = []
        for name, raw in [('first', b'1 0.5 0.5 0.1 0.2'),
                          ('second', b'2 0.3 0.3 0.1 0.1\r\n')]:
            image = reference/f'dataset/images/train/{name}.jpg'
            label = reference/f'dataset/labels/train/{name}.txt'
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(name.encode())
            label.write_bytes(raw)
            rows.append(dict(id=name, image=image.relative_to(reference).as_posix(),
                             label=label.relative_to(reference).as_posix(),
                             image_sha256=m.review.sha(image), label_sha256=m.review.sha(label)))
        m.review.save(reference/'dataset_manifest.json', dict(images=2, boxes=2, rows=rows))
        vi = root/'val/images/val/v.jpg'
        vl = root/'val/labels/val/v.txt'
        vi.parent.mkdir(parents=True)
        vl.parent.mkdir(parents=True)
        vi.write_bytes(b'validation')
        vl.write_bytes(b'3 0.5 0.5 0.2 0.2\n')
        vm = root/'val_manifest.json'
        m.review.save(vm, [dict(image=str(vi), label=str(vl),
                               image_sha256=m.review.sha(vi), label_sha256=m.review.sha(vl))])
        proposals = root/'proposals'
        m.review.save(proposals/'review_snapshot.json', dict(completed=3, total=3))
        snapshot = m.review.sha(proposals/'review_snapshot.json')
        m.review.save(proposals/'conflicts.json', dict(conflicts=[], excluded_keys=[]))
        actions = [dict(key='a', category='change_class', proposed_class=3, image_id='first',
                        label_path=(reference/rows[0]['label']).relative_to(root).as_posix(),
                        target_binding=dict(d085_gt_index=0, original_label_line='1 0.5 0.5 0.1 0.2',
                                            reference_label_sha256=rows[0]['label_sha256'])),
                   dict(key='b', category='add_box', proposed_class=1, image_id='first',
                        label_path=(reference/rows[0]['label']).relative_to(root).as_posix(),
                        final_xyxy=[0, 0, 64, 128], object_key='first-added'),
                   dict(key='c', category='add_box', proposed_class=2, image_id='second',
                        label_path=(reference/rows[1]['label']).relative_to(root).as_posix(),
                        final_xyxy=[128, 128, 192, 192], object_key='second-added')]
        for i, band in enumerate(('D080', 'D070', 'D060')):
            m.review.save(proposals/(band+'.json'), dict(version=band, review_snapshot_sha256=snapshot,
                incremental_actions=actions[i:i+1], cumulative_actions=actions[:i+1],
                cumulative_files=len({a['label_path'] for a in actions[:i+1]})))
        files = {p.name:m.review.sha(p) for p in proposals.iterdir()}
        m.review.save(proposals/'seal.json', dict(files=files, completed=3, candidate_count=3,
                                               review_snapshot_sha256=snapshot))
        approval = root/'approval.json'
        m.review.save(approval, dict(approved_by_user=True, user_message='Apply these labels',
            source_reference='Test explicit approval', proposal_seal_sha256=m.review.sha(proposals/'seal.json'),
            proposal_sha256={b:files[b+'.json'] for b in ('D080','D070','D060')},
            reference_manifest_sha256=m.review.sha(reference/'dataset_manifest.json'),
            deferred_policy='leave_unchanged', conflict_policy='exclude', training_authorized=False))
        return reference, vm, proposals, approval

    def test_application_requires_hash_bound_approval_before_any_output(self):
        m = self.tool()
        self.assertTrue(hasattr(m, 'apply'), 'Missing guarded dataset application')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            ref, vm, proposals, approval = self.fixture(root, m)
            bad = json.loads(approval.read_text())
            bad['proposal_sha256']['D070'] = '0'*64
            approval.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'Approval'):
                m.apply(root, proposals, approval, ref, vm, root/'output')
            self.assertFalse((root/'output').exists())

    def test_three_independent_cumulative_versions_and_readonly_verification(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            ref, vm, proposals, approval = self.fixture(root, m)
            original = (ref/'dataset/labels/train/first.txt').read_bytes()
            output = root/'output'
            result = m.apply(root, proposals, approval, ref, vm, output)
            self.assertEqual([result['versions'][b]['boxes'] for b in ('D080','D070','D060')], [2,3,4])
            self.assertEqual([result['versions'][b]['changed_label_files'] for b in ('D080','D070','D060')], [1,1,2])
            self.assertEqual((ref/'dataset/labels/train/first.txt').read_bytes(), original)
            for band in ('D080','D070','D060'):
                self.assertTrue((output/band/'data.yaml').exists())
                self.assertEqual((output/band/'dataset/images/train/first.jpg').read_bytes(), b'first')
                self.assertEqual((output/band/'dataset/labels/val/v.txt').read_bytes(), b'3 0.5 0.5 0.2 0.2\n')
            self.assertTrue(m.verify(root, output)['verified'])
            with self.assertRaises(FileExistsError):
                m.apply(root, proposals, approval, ref, vm, output)
            target = output/'D060/dataset/labels/train/first.txt'
            target.write_bytes(target.read_bytes()+b'1 0.5 0.5 0.2 0.2\n')
            with self.assertRaises(ValueError):
                m.verify(root, output)

    def test_approval_rejects_training_authorization_even_with_matching_hashes(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            ref, vm, proposals, approval = self.fixture(root, m)
            data = m.review.load(approval)
            data['training_authorized'] = True
            approval.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'training'):
                m.apply(root, proposals, approval, ref, vm, root/'output')
            self.assertFalse((root/'output').exists())

    def test_source_drift_is_rejected_before_copying(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            ref, vm, proposals, approval = self.fixture(root, m)
            (ref/'dataset/images/train/first.jpg').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Source hash'):
                m.apply(root, proposals, approval, ref, vm, root/'output')
            self.assertFalse((root/'output').exists())

    def test_duplicate_gt_edits_and_wrong_original_lines_are_rejected(self):
        m = self.tool()
        raw = b'1 0.5 0.5 0.1 0.2\n'
        action = dict(category='change_class', proposed_class=2,
                      target_binding=dict(d085_gt_index=0, original_label_line='1 0.5 0.5 0.1 0.2'))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            m.edit(raw, [action, action])
        action['target_binding']['original_label_line'] = '1 0.4 0.5 0.1 0.2'
        with self.assertRaisesRegex(ValueError, 'Original label line'):
            m.edit(raw, [action])

    def test_lf_and_empty_labels_have_exact_append_behavior(self):
        m = self.tool()
        add = dict(category='add_box', proposed_class=0, final_xyxy=[0,0,640,640])
        line = b'0 0.500000000000 0.500000000000 1.000000000000 1.000000000000\n'
        self.assertEqual(m.edit(b'', [add]), line)
        raw = b'1 0.5 0.5 0.1 0.1\n'
        self.assertEqual(m.edit(raw, [add]), raw+line)

    def test_verifier_rejects_source_drift_after_application(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            ref, vm, proposals, approval = self.fixture(root, m)
            m.apply(root, proposals, approval, ref, vm, root/'output')
            (ref/'dataset/labels/train/first.txt').write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                m.verify(root, root/'output')

    def test_output_train_manifest_uses_same_lexical_order_as_ab_loader(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            ref, vm, proposals, approval = self.fixture(root, m)
            path = ref/'dataset_manifest.json'
            data = m.review.load(path)
            data['rows'].reverse()
            path.write_text(json.dumps(data))
            consent = m.review.load(approval)
            consent['reference_manifest_sha256'] = m.review.sha(path)
            approval.write_text(json.dumps(consent))
            m.apply(root, proposals, approval, ref, vm, root/'output')
            rows = m.review.load(root/'output/D080/manifest.json')['rows']
            self.assertEqual([r['id'] for r in rows if r['split']=='train'], ['first','second'])

    def test_unchanged_validation_edge_boxes_are_copied_without_repair(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            ref, vm, proposals, approval = self.fixture(root, m)
            val = m.review.load(vm)
            raw = b'3 0.935938 0.450000 0.128125 0.390625\n1 0.99 0.5 0.1 0.2\n'
            Path(val[0]['label']).write_bytes(raw)
            val[0]['label_sha256'] = m.digest(raw)
            vm.write_text(json.dumps(val))
            m.apply(root, proposals, approval, ref, vm, root/'output')
            for band in ('D080','D070','D060'):
                self.assertEqual((root/'output'/band/'dataset/labels/val/v.txt').read_bytes(),raw)


if __name__ == '__main__':
    unittest.main()
