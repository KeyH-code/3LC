"""Review receiver uses real fixture files; never touches frozen experiment data."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReviewTests(unittest.TestCase):
    def tool(self):
        path = ROOT / 'tools/cleaning_expansion_review.py'
        self.assertTrue(path.exists(), 'review receiver is not implemented')
        spec = importlib.util.spec_from_file_location('expansion_review_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self, callback):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            image = root / 'image.jpg'
            image.write_bytes(b'fixture-image-bytes')
            candidate = dict(key='a'*64, id='image', band='D080', scene='scene',
                             image_path='image.jpg', image_sha256=m.sha(image),
                             confidence=.82, signal='prediction_without_label_coverage', gt_boxes=[])
            manifest = dict(batch_id='D080-001-part01', candidate_keys=[candidate['key']],
                            evidence=[dict(id='original', path='image.jpg', sha256=m.sha(image),
                                           candidate_keys=[candidate['key']], kind='original',
                                           source_image_sha256=m.sha(image))])
            manifest_path = root / 'input.json'
            m.save(manifest_path, manifest)
            feedback = dict(schema_version=m.SCHEMA, batch_id=manifest['batch_id'], attempt_id='attempt-1',
                            requested_model=m.MODEL, reported_model=None,
                            input_manifest_sha256=m.sha(manifest_path),
                            started_at='2026-10-29T14:00:00+00:00', finished_at='2026-10-29T14:01:00+00:00',
                            vision_inputs=[dict(id='original', path='image.jpg', sha256=m.sha(image), tool='read',
                                                actual_image_received=True, candidate_keys=[candidate['key']])],
                            results=[dict(key=candidate['key'], category='add_box', confidence='high',
                                          proposed_class=1, final_xyxy=[10,20,30,40], object_key='image:10:20',
                                          reason_code='missing_label', reason='Visible unlabelled car.', notes='',
                                          visual_completed=True, evidence_ids=['original'])], uncompleted_keys=[])
            feedback_path = root / 'feedback.json'
            m.save(feedback_path, feedback)
            raw = root / 'raw.txt'
            raw.write_text(json.dumps(feedback), encoding='utf-8')
            transport = dict(session_id='ses_fixture', requested_model=m.MODEL, reported_model=None,
                             raw_path='raw.txt', raw_sha256=m.sha(raw),
                             vision_receipts=copy.deepcopy(feedback['vision_inputs']),
                             receipt_basis='agent_tool_report_after_capability_check')
            store = root / 'ledger'
            callback(m, root, store, candidate, manifest_path, feedback_path, transport, feedback)

    def test_accept_preserves_score_and_agent_provenance(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
            state = m.snapshot(store, [candidate])
            row = state['rows'][0]
            self.assertEqual(row['source'], 'agent_adopted')
            self.assertEqual(row['yolo_confidence'], .82)
            self.assertEqual(row['confidence'], 'high')
            self.assertEqual(row['session_id'], 'ses_fixture')
            self.assertIsNone(row['reported_model'])
            event = m.load(store/'accepted/attempt-1.json')
            self.assertEqual(event['raw_response'], (root/'raw.txt').read_text())
            self.assertEqual(state['completed'], 1)
            with self.assertRaises(ValueError):
                m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
            self.assertEqual(state, m.snapshot(store, [candidate]))
        self.fixture(run)

    def test_evidence_and_transport_tampering_rejected(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            wrong = root / 'wrong.jpg'
            wrong.write_bytes(b'wrong image')
            for case in ['model', 'raw', 'manifest', 'image', 'foreign_evidence', 'traversal']:
                with self.subTest(case=case):
                    data, t = copy.deepcopy(feedback), copy.deepcopy(transport)
                    man = m.load(manifest_path)
                    if case == 'model':
                        data['reported_model'] = t['reported_model'] = 'cpa/other'
                    elif case == 'raw':
                        t['raw_sha256'] = '0'*64
                    elif case == 'manifest':
                        data['input_manifest_sha256'] = '0'*64
                    elif case == 'image':
                        data['vision_inputs'][0]['path'] = 'wrong.jpg'
                        data['vision_inputs'][0]['sha256'] = m.sha(wrong)
                        man['evidence'][0].update(path='wrong.jpg', sha256=m.sha(wrong))
                    elif case == 'foreign_evidence':
                        data['results'][0]['evidence_ids'] = ['unbound']
                    else:
                        data['vision_inputs'][0]['path'] = '../not-permitted.jpg'
                        man['evidence'][0]['path'] = '../not-permitted.jpg'
                    manifest_path.write_text(json.dumps(man), encoding='utf-8')
                    if case in ['image', 'traversal']:
                        data['input_manifest_sha256'] = m.sha(manifest_path)
                    feedback_path.write_text(json.dumps(data), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        m.receive(root, store, [candidate], manifest_path, feedback_path, t)
                    self.assertEqual(m.snapshot(store, [candidate])['completed'], 0)
        self.fixture(run)

    def test_class_change_binds_stable_d085_line_and_preserves_geometry(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            label = root / 'label.txt'
            label.write_text('0  .25 .50 .10 .20\n', encoding='utf-8')
            candidate.update(signal='class_disagreement', label_path='label.txt', label_sha256=m.sha(label),
                             reference_label_sha256=m.sha(label), d085_gt_index=0, original_class=0,
                             original_label_line='0  .25 .50 .10 .20', d085_target_status='unchanged_original')
            candidate['stable_target_id'] = m.stable_target(candidate)
            result = feedback['results'][0]
            result.update(category='change_class', proposed_class=2, final_xyxy=None, object_key=None,
                          reason_code='wrong_class')
            feedback_path.write_text(json.dumps(feedback), encoding='utf-8')
            (root/'raw.txt').write_text(json.dumps(feedback), encoding='utf-8')
            transport['raw_sha256'] = m.sha(root/'raw.txt')
            m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
            row = m.snapshot(store, [candidate])['rows'][0]
            self.assertEqual(row['target_binding']['stable_target_id'], candidate['stable_target_id'])
            self.assertEqual(label.read_text(), '0  .25 .50 .10 .20\n')
            for mutation in ['line', 'hash', 'identity', 'old_action', 'same_class']:
                c = copy.deepcopy(candidate)
                f = copy.deepcopy(feedback)
                f['attempt_id'] = 'attempt-' + mutation
                if mutation == 'line': c['original_label_line'] = '0 .1 .1 .1 .1'
                if mutation == 'hash': c['reference_label_sha256'] = 'f'*64
                if mutation == 'identity': c['stable_target_id'] = 'f'*64
                if mutation == 'old_action': c['d085_target_status'] = 'approved_class_correction'
                if mutation == 'same_class': f['results'][0]['proposed_class'] = 0
                feedback_path.write_text(json.dumps(f), encoding='utf-8')
                (root/'raw.txt').write_text(json.dumps(f), encoding='utf-8')
                transport['raw_sha256'] = m.sha(root/'raw.txt')
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    m.receive(root, root/('store-'+mutation), [c], manifest_path, feedback_path, transport)
        self.fixture(run)

    def test_cross_band_duplicate_and_overlap_excluded_from_all_proposals(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            second = dict(candidate, key='b'*64, band='D060')
            man = m.load(manifest_path)
            man['candidate_keys'].append(second['key'])
            man['evidence'][0]['candidate_keys'].append(second['key'])
            manifest_path.write_text(json.dumps(man), encoding='utf-8')
            feedback['input_manifest_sha256'] = m.sha(manifest_path)
            feedback['vision_inputs'][0]['candidate_keys'].append(second['key'])
            feedback['results'].append(dict(feedback['results'][0], key=second['key'], object_key='different-id'))
            transport['vision_receipts'] = copy.deepcopy(feedback['vision_inputs'])
            feedback_path.write_text(json.dumps(feedback), encoding='utf-8')
            (root/'raw.txt').write_text(json.dumps(feedback), encoding='utf-8')
            transport['raw_sha256'] = m.sha(root/'raw.txt')
            m.receive(root, store, [candidate, second], manifest_path, feedback_path, transport)
            proposals = m.proposals(store, [candidate, second])
            self.assertEqual(len(proposals['conflicts']), 1)
            self.assertEqual(proposals['bands']['D060']['cumulative_actions'], [])
            self.assertEqual(proposals['bands']['D080']['cumulative_actions'], [])
            self.assertFalse(proposals['label_application_authorized'])
        self.fixture(run)

    def test_unfinished_and_blocked_do_not_become_final_proposals(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            with self.assertRaises(ValueError): m.proposals(store, [candidate])
            feedback['results'][0].update(category='blocked', confidence='low', proposed_class=None,
                final_xyxy=None, object_key=None, reason_code='tool_failure', visual_completed=False, evidence_ids=[])
            feedback_path.write_text(json.dumps(feedback), encoding='utf-8')
            (root/'raw.txt').write_text(json.dumps(feedback), encoding='utf-8')
            transport['raw_sha256'] = m.sha(root/'raw.txt')
            m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
            self.assertEqual(m.snapshot(store, [candidate])['completed'], 0)
            self.assertEqual(m.snapshot(store, [candidate])['blocked'], 1)
            with self.assertRaises(ValueError): m.proposals(store, [candidate])
        self.fixture(run)

    def test_transport_receipt_required_even_when_feedback_claims_image_received(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            transport['vision_receipts'] = []
            with self.assertRaises(ValueError):
                m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
        self.fixture(run)

    def test_receipt_basis_cannot_claim_unavailable_host_verification(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            transport['receipt_basis'] = 'host_independently_verified'
            with self.assertRaises(ValueError):
                m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
        self.fixture(run)

    def test_store_lock_prevents_parallel_import_before_writing(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            store.mkdir()
            (store/'receive.lock').write_text('other process')
            with self.assertRaises(ValueError):
                m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
            self.assertFalse((store/'accepted').exists())
        self.fixture(run)

    def test_complete_pending_event_can_be_resumed_without_overwriting(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
            committed = store/'accepted/attempt-1.json'
            original = committed.read_bytes()
            (store/'pending').mkdir(exist_ok=True)
            committed.rename(store/'pending/attempt-1.json')
            m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
            self.assertEqual(committed.read_bytes(), original)
            self.assertEqual(m.snapshot(store, [candidate])['completed'], 1)
        self.fixture(run)

    def test_recovery_ignores_uncommitted_event_and_raw_feedback_cannot_diverge(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            m.save(store/'pending'/'interrupted.json', {'not': 'committed'})
            self.assertEqual(m.snapshot(store, [candidate])['completed'], 0)
            feedback['results'][0]['reason'] = 'Not present in raw opinion'
            feedback_path.write_text(json.dumps(feedback), encoding='utf-8')
            with self.assertRaises(ValueError):
                m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
        self.fixture(run)

    def test_manifest_candidate_content_cannot_override_frozen_identity(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            man = m.load(manifest_path)
            man['candidates'] = [dict(candidate, confidence=.99)]
            manifest_path.write_text(json.dumps(man), encoding='utf-8')
            feedback['input_manifest_sha256'] = m.sha(manifest_path)
            feedback_path.write_text(json.dumps(feedback), encoding='utf-8')
            (root/'raw.txt').write_text(json.dumps(feedback), encoding='utf-8')
            transport['raw_sha256'] = m.sha(root/'raw.txt')
            with self.assertRaises(ValueError):
                m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
        self.fixture(run)

    def test_same_target_class_conflict_is_excluded_across_bands(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            label = root/'label.txt'
            label.write_text('0 .5 .5 .2 .2\n', encoding='utf-8')
            candidate.update(signal='class_disagreement', label_path='label.txt', label_sha256=m.sha(label),
                reference_label_sha256=m.sha(label), d085_gt_index=0, original_class=0,
                original_label_line='0 .5 .5 .2 .2', d085_target_status='unchanged_original')
            candidate['stable_target_id'] = m.stable_target(candidate)
            second = dict(candidate, key='b'*64, band='D070')
            man = m.load(manifest_path)
            man['candidate_keys'].append(second['key'])
            man['evidence'][0]['candidate_keys'].append(second['key'])
            manifest_path.write_text(json.dumps(man), encoding='utf-8')
            feedback['input_manifest_sha256'] = m.sha(manifest_path)
            feedback['vision_inputs'][0]['candidate_keys'].append(second['key'])
            feedback['results'][0].update(category='change_class', proposed_class=2, final_xyxy=None,
                object_key=None, reason_code='wrong_class')
            feedback['results'].append(dict(feedback['results'][0], key=second['key'], proposed_class=3))
            feedback_path.write_text(json.dumps(feedback), encoding='utf-8')
            (root/'raw.txt').write_text(json.dumps(feedback), encoding='utf-8')
            transport['raw_sha256'] = m.sha(root/'raw.txt')
            transport['vision_receipts'] = copy.deepcopy(feedback['vision_inputs'])
            m.receive(root, store, [candidate, second], manifest_path, feedback_path, transport)
            proposal = m.proposals(store, [candidate, second])
            self.assertEqual(proposal['excluded_keys'], [candidate['key'], second['key']])
        self.fixture(run)

    def test_progress_reports_unreviewed_and_conflicts_without_finalizing(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            report = m.progress(store, [candidate])
            self.assertFalse(report['proposal_ready'])
            self.assertEqual(report['bands']['D080']['unreviewed'], 1)
            m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
            report = m.progress(store, [candidate])
            self.assertTrue(report['proposal_ready'])
            self.assertEqual(report['bands']['D080']['decisions'], {'add_box': 1})
            self.assertFalse(report['label_application_authorized'])
        self.fixture(run)

    def test_collect_run_creates_transport_and_imports_saved_agent_opinion(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            state = m.collect_run(root, store, [candidate], manifest_path, feedback_path, 'ses_fixture')
            self.assertEqual(state['completed'], 1)
            record = m.load(store/'transports/attempt-1.json')
            self.assertEqual(record['receipt_basis'], 'agent_tool_report_after_capability_check')
            self.assertEqual(record['raw_sha256'], m.sha(feedback_path))
        self.fixture(run)

    def test_keep_opinion_for_same_gt_blocks_cross_band_class_change(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            store = Path(temp)
            target = 'c'*64
            candidates = [dict(key='a'*64, id='image', band='D080', scene='s', label_path='label.txt',
                               image_path='image.jpg', stable_target_id=target, gt_boxes=[]),
                          dict(key='b'*64, id='image', band='D060', scene='s', label_path='label.txt',
                               image_path='image.jpg', stable_target_id=target, gt_boxes=[])]
            rows = [dict(key='a'*64, category='change_class', proposed_class=2,
                         target_binding=dict(stable_target_id=target)),
                    dict(key='b'*64, category='keep')]
            m.save(store/'accepted/opinions.json', dict(rows=rows))
            proposal = m.proposals(store, candidates)
            self.assertEqual(proposal['excluded_keys'], ['a'*64])
            self.assertEqual(proposal['bands']['D080']['cumulative_actions'], [])
            self.assertEqual(proposal['conflicts'][0]['reason'], 'same_GT_keep_vs_change')

    def test_addition_conflicts_with_reclassified_same_image_target(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            store = Path(temp)
            target = 'c'*64
            gt = dict(**{'class':0}, xyxy=[10,20,30,40])
            candidates = [dict(key='a'*64, id='image', band='D080', scene='s', label_path='l',
                image_path='i', stable_target_id=target, gt_xyxy=gt['xyxy'], gt_boxes=[gt]),
                dict(key='b'*64, id='image', band='D070', scene='s', label_path='l', image_path='i', gt_boxes=[gt])]
            rows = [dict(key='a'*64, category='change_class', proposed_class=2,
                         target_binding=dict(stable_target_id=target)),
                    dict(key='b'*64, category='add_box', proposed_class=1,
                         final_xyxy=[10,20,30,40], object_key='same-target')]
            m.save(store/'accepted/event.json', dict(rows=rows))
            proposal = m.proposals(store, candidates)
            self.assertEqual(proposal['excluded_keys'], ['a'*64, 'b'*64])
            self.assertTrue(any(r['reason']=='addition_vs_class_change_target' for r in proposal['conflicts']))

    def test_unreadable_reason_is_rejected_before_acceptance(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            feedback['results'][0]['reason'] = '???????? van ?????? car ?????'
            feedback_path.write_text(json.dumps(feedback), encoding='utf-8')
            (root/'raw.txt').write_text(json.dumps(feedback), encoding='utf-8')
            transport['raw_sha256'] = m.sha(root/'raw.txt')
            with self.assertRaises(ValueError):
                m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
            self.assertEqual(m.snapshot(store, [candidate])['completed'], 0)
        self.fixture(run)

    def test_append_only_quality_hold_excludes_invalid_event_without_erasing_it(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
            event = store/'accepted/attempt-1.json'
            original = event.read_bytes()
            m.save(store/'quality_holds/hold-1.json', dict(event_file='attempt-1.json',
                event_sha256=m.sha(event), reason='Encoding failure; original retained'))
            state = m.snapshot(store, [candidate])
            self.assertEqual(state['completed'], 0)
            self.assertEqual(state['quality_held'], 1)
            self.assertEqual(event.read_bytes(), original)
            with self.assertRaises(ValueError): m.proposals(store, [candidate])
        self.fixture(run)

    def test_user_visual_approval_releases_specific_hold_without_rewriting_history(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
            event = store/'accepted/attempt-1.json'
            hold = store/'quality_holds/visual.json'
            m.save(hold, dict(event_file=event.name, event_sha256=m.sha(event), reason='Retrospective uncertainty'))
            original = event.read_bytes()
            self.assertEqual(m.snapshot(store,[candidate])['completed'],0)
            approval = store/'user-approval.json'
            m.save(approval,dict(approved_by_user=True, restore_retrospective_visual_holds=True))
            m.save(store/'quality_releases/release.json',dict(hold_file=hold.name,
                hold_sha256=m.sha(hold), event_sha256=m.sha(event),
                approval_path='user-approval.json', approval_sha256=m.sha(approval)))
            self.assertEqual(m.snapshot(store,[candidate])['completed'],1)
            self.assertEqual(event.read_bytes(),original)
            self.assertTrue(hold.exists())
        self.fixture(run)

    def test_schema_invalid_payloads_never_create_accepted_events(self):
        def run(m, root, store, candidate, manifest_path, feedback_path, transport, feedback):
            bad_values = [
                lambda d: d.pop('finished_at'),
                lambda d: d.update(unknown='not allowed'),
                lambda d: d.update(started_at='not a timestamp'),
                lambda d: d.update(finished_at='2026-10-29T13:00:00+00:00'),
                lambda d: d['results'][0].update(reason='x'*161),
                lambda d: d['results'][0].update(notes='x'*161),
                lambda d: d['results'][0].update(category='delete_box'),
                lambda d: d['results'][0].update(proposed_class=True),
                lambda d: d['results'][0].update(final_xyxy=[30,20,10,40]),
                lambda d: d['results'][0].update(final_xyxy=[0,0,641,10]),
                lambda d: d['results'][0].update(confidence='medium'),
                lambda d: d['results'][0].update(visual_completed=False),
                lambda d: d['results'][0].update(evidence_ids=[]),
                lambda d: d['results'][0].update(yolo_confidence=.99),
                lambda d: d['vision_inputs'][0].update(actual_image_received=False),
                lambda d: d['results'][0].update(category='keep', proposed_class=None,
                    final_xyxy=None, object_key=None, reason_code='model_error', visual_completed=False),
                lambda d: d.update(results=[]),
                lambda d: d['results'].append(copy.deepcopy(d['results'][0])),
                lambda d: d['results'][0].update(key='b'*64),
            ]
            for mutate in bad_values:
                with self.subTest(mutation=bad_values.index(mutate)):
                    bad = copy.deepcopy(feedback)
                    mutate(bad)
                    feedback_path.write_text(json.dumps(bad), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        m.receive(root, store, [candidate], manifest_path, feedback_path, transport)
                    self.assertEqual(m.snapshot(store, [candidate])['completed'], 0)
        self.fixture(run)


if __name__ == '__main__':
    unittest.main()
