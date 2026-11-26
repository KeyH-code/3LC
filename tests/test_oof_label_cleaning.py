"""Synthetic fixtures only; never run model inference or edit project datasets."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('label_cleaning', ROOT/'tools/oof_label_cleaning.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def prediction(ident='p0', cls=2, conf=.85, box=None):
    return {'id': ident, 'cls': cls, 'class_name': c.NAMES[cls], 'confidence': conf,
            'xyxy': box or [256, 256, 384, 384]}


def sample(labels=None):
    return dict(id='image1', scene='MVI_FAKE', frame=1, image_path='train/images/a.jpg',
                label_path='train/labels/a.txt', image_sha256='imagehash', label_sha256='labelhash',
                labels=labels if labels is not None else [[1, .5, .5, .2, .2]], split='train')


def record(predictions):
    return dict(fold=1, model_sha256='modelhash', candidates=predictions)


def candidate(key='k1', ident='image1', signal=c.CONFLICT, gt=0):
    r = {k: v for k, v in sample().items() if k not in ['labels', 'split']}
    r.update(key=key, id=ident, number=1, signal=signal, prediction_id='p0', confidence=.85,
             predicted_class=2, predicted_name='van', xyxy=[256, 256, 384, 384], touches_image_edge=False,
             gt_boxes=[{'class': 1, 'xyxy': [256, 256, 384, 384]}])
    if signal == c.CONFLICT:
        r.update(gt_index=gt, original_class=1, original_name='car', gt_xyxy=r['xyxy'],
                 original_label_line='1 0.5 0.5 0.2 0.2')
    return r


def review(key='k1', decision='change_class', **kwargs):
    r = dict(key=key, decision=decision, source='agent_adopted', proposed_class=2,
             geometry_ready=True, final_xyxy=None, object_key=None, reason='fixture reason',
             observation='fixture visible target', uncertainty='', evidence=[], visual_completed=True,
             review_model='gpt-6-luna', evidence_sufficiency='clear')
    r.update(kwargs)
    return r


class SignalTests(unittest.TestCase):
    def test_exact_confidence_threshold_included(self):
        rows = c.signals(sample(), record([prediction()]), .85)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['signal'], c.CONFLICT)
        self.assertEqual(rows[0]['gt_index'], 0)

    def test_below_confidence_excluded(self):
        self.assertEqual(c.signals(sample(), record([prediction(conf=.849999)]), .85), [])

    def test_empty_gt_generates_uncovered(self):
        rows = c.signals(sample([]), record([prediction()]), .85)
        self.assertEqual(rows[0]['signal'], c.EXTRA)
        self.assertEqual(rows[0]['max_iou_any_gt'], 0)

    def test_no_prediction(self):
        self.assertEqual(c.signals(sample(), record([]), .85), [])

    def test_same_class_is_not_conflict(self):
        self.assertEqual(c.signals(sample(), record([prediction(cls=1)]), .85), [])

    def test_highest_conflict_per_gt(self):
        rows = c.signals(sample(), record([prediction('p0', conf=.86), prediction('p1', conf=.91)]), .85)
        self.assertEqual([r['prediction_id'] for r in rows], ['p1'])

    def test_same_class_overlap_suppresses_conflict(self):
        labels = [[1, .5, .5, .2, .2], [2, .5, .5, .2, .2]]
        self.assertEqual(c.signals(sample(labels), record([prediction()]), .85), [])

    def test_independent_uncovered_targets_not_sampled_away(self):
        rows = c.signals(sample([]), record([prediction('p0'), prediction('p1', box=[0, 0, 20, 20])]), .85)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({r['key'] for r in rows}), 2)

    def test_prediction_fraction_boundary_is_strict(self):
        labels = [[1, 5/640, 50/640, 10/640, 100/640]]
        rows = c.signals(sample(labels), record([prediction(box=[0, 0, 50, 10])]), .85)
        self.assertEqual(rows, [])

    def test_nonfinite_or_outside_box_rejected(self):
        for box in [[0, 0, float('nan'), 20], [0, 0, 641, 20], [0, 0, 0, 20]]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                c.checked_box(box)

    def test_candidate_key_tracks_gt_identity(self):
        a = candidate(); b = copy.deepcopy(a); b['gt_index'] = 1
        self.assertNotEqual(c.candidate_key(a), c.candidate_key(b))


class ProposalTests(unittest.TestCase):
    def test_missing_geometry_is_deferred(self):
        ca = candidate(signal=c.EXTRA); ca['gt_boxes'] = []
        changes, deferred = c.proposal([ca], [review(decision='add_box', geometry_ready=False)])
        self.assertEqual(changes, [])
        self.assertEqual(deferred[0]['status'], 'pending_geometry')

    def test_duplicate_same_object_consolidated(self):
        a = candidate(); b = candidate(key='k2')
        changes, deferred = c.proposal([a, b], [review(), review(key='k2')])
        self.assertEqual(len(changes), 1)
        self.assertEqual(len(changes[0]['candidate_keys']), 2)
        self.assertEqual(deferred, [])

    def test_conflicting_same_gt_is_deferred(self):
        changes, deferred = c.proposal([candidate(), candidate(key='k2')],
                                      [review(), review(key='k2', proposed_class=3)])
        self.assertEqual(changes, [])
        self.assertEqual(len(deferred), 2)

    def test_additions_overlap_flags_not_merged(self):
        a = candidate(signal=c.EXTRA); a['gt_boxes'] = []
        b = candidate(key='k2', signal=c.EXTRA); b['gt_boxes'] = []
        ra = review(decision='add_box', final_xyxy=[0, 0, 50, 50], object_key='left')
        rb = review(key='k2', decision='add_box', final_xyxy=[5, 5, 55, 55], object_key='right')
        changes, deferred = c.proposal([a, b], [ra, rb])
        self.assertEqual(changes, [])
        self.assertEqual(len(deferred), 2)

    def test_addition_over_existing_gt_deferred(self):
        changes, deferred = c.proposal([candidate(signal=c.EXTRA)],
            [review(decision='add_box', final_xyxy=[256, 256, 384, 384], object_key='car')])
        self.assertEqual(changes, [])
        self.assertEqual(deferred[0]['status'], 'overlap_conflict')

    def test_different_images_not_deduplicated(self):
        changes, deferred = c.proposal([candidate(), candidate(key='k2', ident='image2')], [review(), review(key='k2')])
        self.assertEqual((len(changes), len(deferred)), (2, 0))


class LabelEditTests(unittest.TestCase):
    def test_class_only_preserves_spacing_coordinates_crlf(self):
        raw = b'1  0.5\t0.5 0.2 0.2\r\n3 0.3 0.3 0.1 0.1\r\n'
        change = dict(action='change_class', gt_index=0, original_label_line='1  0.5\t0.5 0.2 0.2', old_class=1, new_class=2)
        self.assertEqual(c.edited_label(raw, [change]), b'2'+raw[1:])

    def test_unchanged_bytes_empty_and_no_newline(self):
        for raw in [b'', b'1 0.5 0.5 0.2 0.2', b'1 0.5 0.5 0.2 0.2\r\n']:
            self.assertEqual(c.edited_label(raw, []), raw)

    def test_add_box_appends_and_normalizes(self):
        raw = b'1 0.5 0.5 0.2 0.2'
        result = c.edited_label(raw, [dict(action='add_box', new_class=3, final_xyxy=[0, 0, 640, 640])])
        self.assertTrue(result.startswith(raw+b'\n'))
        self.assertEqual(list(map(float, result.splitlines()[1].split())), [3, .5, .5, 1, 1])

    def test_changed_original_line_rejected(self):
        change = dict(action='change_class', gt_index=0, original_label_line='wrong', old_class=1, new_class=2)
        with self.assertRaises(ValueError):
            c.edited_label(b'1 0.5 0.5 0.2 0.2\n', [change])

    def test_duplicate_gt_changes_rejected(self):
        change = dict(action='change_class', gt_index=0, original_label_line='1 0.5 0.5 0.2 0.2', old_class=1, new_class=2)
        with self.assertRaises(ValueError):
            c.edited_label((change['original_label_line']+'\n').encode(), [change, change])


class WorkflowFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='oof085-test-', dir=os.environ.get('PI_SCRATCH_DIR'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.out = self.root/'reports/fixture'; self.out.mkdir(parents=True)
        for name in ['tools/oof_label_cleaning.py', 'tests/test_oof_label_cleaning.py', 'tools/oof_label_cleaning_review.template.html']:
            p = self.root/name; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('__DATA__' if name.endswith('.html') else '# fixture', encoding='utf-8')
        image = self.root/'train/images/a.jpg'; image.parent.mkdir(parents=True); image.write_bytes(b'synthetic-image')
        label = self.root/'train/labels/a.txt'; label.parent.mkdir(parents=True); label.write_bytes(b'1 0.5 0.5 0.2 0.2\r\n')
        self.ca = candidate(); self.ca.update(image_sha256=c.sha(image), label_sha256=c.sha(label))
        source = sample(); source.update(image_sha256=c.sha(image), label_sha256=c.sha(label))
        manifest = self.root/c.OOF/'source_manifest.json'; manifest.parent.mkdir(parents=True)
        c.save(manifest, [source])
        evidence = self.out/'evidence.jpg'; evidence.write_bytes(b'synthetic-test-evidence-not-real-review')
        self.r = review(evidence=[{'root': 'report', 'path': 'evidence.jpg', 'sha256': c.sha(evidence)}])
        c.save(self.out/'candidates.json', {'rows': [self.ca]})
        c.save(self.out/'inherited.json', {'rows': []})
        c.save(self.out/'selection_config.json', {'phase_b_authorized': False})
        c.save(self.out/'review.json', {'rows': [self.r]})
        c.save(self.out/'input_integrity.json', {'protected': {
            image.relative_to(self.root).as_posix(): c.sha(image),
            label.relative_to(self.root).as_posix(): c.sha(label)}})
        self.freeze()
        self.confirm()

    def freeze(self):
        c.save(self.out/'freeze.json', {'files': {n: c.sha(self.out/n) for n in
               ['input_integrity.json', 'candidates.json', 'inherited.json', 'selection_config.json']}})

    def confirm(self):
        c.save(self.out/'model_confirmation.json', {'confirmed_by_user': True, 'image_input_capable': True,
               'actual_model': 'gpt-6-luna', 'evidence_reference': 'SYNTHETIC UNIT TEST ONLY'})

    def write_review(self):
        c.save(self.out/'review.json', {'rows': [self.r]})

    def approval(self):
        p = self.out/'approval.json'
        c.save(p, dict(phase='B', approved_by_user=True, proposal_sha256=c.sha(self.out/'proposed_changes.json'),
            user_message='SYNTHETIC TEST AUTHORIZATION', source_reference='unit-test-only',
            approved_at=(datetime.now(timezone.utc)+timedelta(seconds=2)).isoformat(), deferred_policy='leave_unchanged',
            change_ids=[x['change_id'] for x in c.load(self.out/'proposed_changes.json')['rows']]))
        return p

    def test_new_decision_requires_confirmed_model(self):
        (self.out/'model_confirmation.json').unlink()
        with self.assertRaisesRegex(ValueError, 'not confirmed'):
            c.validate_reviews(self.root, self.out)

    def inherited_model(self, with_authorization=True, image_capable=True):
        c.save(self.out/'model_confirmation.json', {'confirmed_by_user': True,
               'image_input_capable': image_capable, 'actual_model': 'current-session-inherited',
               'selection_mode': 'inherit_current_session_subagent',
               'evidence_reference': 'SYNTHETIC VISION CAPABILITY TEST ONLY'})
        if with_authorization:
            c.save(self.out/'review_model_authorization.json', {'approved_by_user': True,
                   'mode': 'inherit_current_session_subagent', 'message': 'SYNTHETIC USER MODEL AMENDMENT',
                   'source_reference': 'synthetic-unit-test'})
        self.r['review_model'] = 'current-session-inherited'
        self.write_review()

    def test_explicit_current_model_subagent_authorization(self):
        self.inherited_model()
        self.assertEqual(c.validate_reviews(self.root, self.out)[1][0]['review_model'], 'current-session-inherited')

    def test_current_model_not_silently_substituted(self):
        self.inherited_model(with_authorization=False)
        with self.assertRaisesRegex(ValueError, 'not explicitly authorized'):
            c.validate_reviews(self.root, self.out)

    def test_current_model_still_needs_image_capability(self):
        self.inherited_model(image_capable=False)
        with self.assertRaisesRegex(ValueError, 'confirmation incomplete'):
            c.validate_reviews(self.root, self.out)

    def test_current_model_record_cannot_claim_pinned_model(self):
        self.inherited_model()
        self.r['review_model'] = 'gpt-6-luna'; self.write_review()
        with self.assertRaisesRegex(ValueError, 'does not match'):
            c.validate_reviews(self.root, self.out)

    def parent_model(self):
        c.save(self.out/'parent_review_authorization.json', {
            'approved_by_user': True, 'selection_mode': 'any_actual_image_capable_agent_for_frozen_scope',
            'user_message': 'SYNTHETIC EXPLICIT ONE-TASK REVIEW AUTHORIZATION',
            'source_reference': 'synthetic-unit-test', 'image_input_verified': True,
            'evidence_reference': 'synthetic-image-only', 'candidate_keys': [self.r['key']]})
        self.r.update(review_model='current-session-parent', reviewer_role='parent',
                      vision_inputs=[{'tool': 'synthetic-image-only', 'actual_image_received': True}])
        self.write_review()

    def test_parent_model_preserves_actual_reviewer_identity(self):
        self.parent_model()
        self.assertEqual(c.validate_reviews(self.root, self.out)[1][0]['review_model'], 'current-session-parent')

    def test_parent_model_requires_explicit_authorization(self):
        self.parent_model(); (self.out/'parent_review_authorization.json').unlink()
        with self.assertRaisesRegex(ValueError, 'parent reviewer not explicitly authorized'):
            c.validate_reviews(self.root, self.out)

    def test_parent_model_rejects_out_of_scope_key(self):
        self.parent_model(); p = self.out/'parent_review_authorization.json'; a = c.load(p)
        a['candidate_keys'] = ['not-authorized']; c.save(p, a)
        with self.assertRaisesRegex(ValueError, 'outside authorized frozen scope'):
            c.validate_reviews(self.root, self.out)

    def test_parent_model_requires_image_input_not_only_path(self):
        self.parent_model(); self.r['vision_inputs'][0]['actual_image_received'] = False; self.write_review()
        with self.assertRaisesRegex(ValueError, 'actual image input record'):
            c.validate_reviews(self.root, self.out)

    def test_parent_model_still_rejects_uncertain_substantive_change(self):
        self.parent_model(); self.r['uncertainty'] = 'visible geometry unclear'; self.write_review()
        with self.assertRaisesRegex(ValueError, 'Uncertain'):
            c.validate_reviews(self.root, self.out)

    def session_subagent(self):
        (self.out/'chrome-mcp').mkdir(exist_ok=True)
        c.save(self.out/'chrome-mcp/static-plates-manifest.json', {'plates': [
            {'path': 'evidence.jpg', 'sha256': c.sha(self.out/'evidence.jpg'), 'keys': [self.r['key']]}]})
        c.save(self.out/'session_review_authorization.json', {
            'approved_by_user': True, 'phase': 'A',
            'selection_mode': 'actual_image_capable_session_agents',
            'user_message': 'SYNTHETIC CONTINUE FROZEN VISUAL REVIEW',
            'source_reference': 'synthetic-unit-test',
            'candidate_keys': [self.r['key']],
            'allowed_models': ['cpa/gpt-6.1-sol-high'],
            'label_application_authorized': False})
        self.r.update(review_model='current-session-subagent', reviewer_role='subagent',
                      actual_model='cpa/gpt-6.1-sol-high',
                      vision_inputs=[{'tool': 'functions.read', 'path': 'evidence.jpg',
                                      'actual_image_received': True}])
        self.write_review()

    def test_session_subagent_preserves_real_model_and_scoped_authorization(self):
        self.session_subagent()
        self.assertEqual(c.validate_reviews(self.root, self.out)[1][0]['actual_model'],
                         'cpa/gpt-6.1-sol-high')

    def test_session_subagent_requires_authorization_file(self):
        self.session_subagent(); (self.out/'session_review_authorization.json').unlink()
        with self.assertRaisesRegex(ValueError, 'not explicitly authorized'):
            c.validate_reviews(self.root, self.out)

    def test_session_subagent_rejects_outside_scope(self):
        self.session_subagent()
        path = self.out/'session_review_authorization.json'; a = c.load(path)
        a['candidate_keys'] = ['another-key']; c.save(path, a)
        with self.assertRaisesRegex(ValueError, 'outside authorized frozen scope'):
            c.validate_reviews(self.root, self.out)

    def test_session_subagent_rejects_unapproved_model(self):
        self.session_subagent(); self.r['actual_model'] = 'pretend-model'; self.write_review()
        with self.assertRaisesRegex(ValueError, 'model not authorized'):
            c.validate_reviews(self.root, self.out)

    def test_session_subagent_requires_received_image_and_matching_evidence(self):
        self.session_subagent(); self.r['vision_inputs'][0]['path'] = 'different.jpg'; self.write_review()
        with self.assertRaisesRegex(ValueError, 'actual image input linked'):
            c.validate_reviews(self.root, self.out)
        self.r['vision_inputs'][0].update(path='evidence.jpg', actual_image_received=False); self.write_review()
        with self.assertRaisesRegex(ValueError, 'actual image input linked'):
            c.validate_reviews(self.root, self.out)

    def test_session_subagent_cannot_turn_review_permission_into_apply_permission(self):
        self.session_subagent()
        path = self.out/'session_review_authorization.json'; a = c.load(path)
        a['label_application_authorized'] = True; c.save(path, a)
        with self.assertRaisesRegex(ValueError, 'authorization incomplete'):
            c.validate_reviews(self.root, self.out)

    def test_summary_reports_actual_mixed_reviewer_identity(self):
        self.session_subagent()
        summary = c.summarize(self.root, self.out)
        self.assertEqual(summary['reviewer_counts'], {'current-session-subagent': 1})
        self.assertEqual(summary['actual_session_models'], {'cpa/gpt-6.1-sol-high': 1})

    def test_session_subagent_rejects_valid_evidence_for_another_candidate(self):
        self.session_subagent()
        path = self.out/'chrome-mcp/static-plates-manifest.json'; manifest = c.load(path)
        manifest['plates'][0]['keys'] = ['a-different-candidate']; c.save(path, manifest)
        with self.assertRaisesRegex(ValueError, 'candidate-bound image evidence'):
            c.validate_reviews(self.root, self.out)

    def test_session_parent_uses_real_identity_and_same_strict_image_binding(self):
        self.session_subagent()
        self.r.update(review_model='current-session-parent', reviewer_role='parent')
        self.write_review()
        self.assertEqual(c.validate_reviews(self.root, self.out)[1][0]['reviewer_role'], 'parent')

    def test_visual_observation_and_evidence_required(self):
        self.r['observation'] = ''; self.write_review()
        with self.assertRaises(ValueError):c.validate_reviews(self.root, self.out)

    def test_evidence_tamper_rejected(self):
        (self.out/'evidence.jpg').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'evidence hash'):c.validate_reviews(self.root, self.out)

    def test_model_mismatch_rejected(self):
        self.r['review_model'] = 'other'; self.write_review()
        with self.assertRaisesRegex(ValueError, 'model'):c.validate_reviews(self.root, self.out)

    def test_uncertain_clear_decision_rejected(self):
        self.r['uncertainty'] = 'cannot see'; self.write_review()
        with self.assertRaisesRegex(ValueError, 'Uncertain'):c.validate_reviews(self.root, self.out)

    def test_defer_allows_uncertainty(self):
        self.r.update(decision='defer', uncertainty='class boundary', evidence_sufficiency='uncertain');self.write_review()
        self.assertEqual(c.validate_reviews(self.root, self.out)[1][0]['decision'], 'defer')

    def test_human_adjudication_protected(self):
        seed = copy.deepcopy(self.r); seed.update(source='human_verified', historical_status='wrong_class', historical_case=1, historical_batch='fixture')
        c.save(self.out/'inherited.json', {'rows': [seed]}); self.freeze()
        self.r = copy.deepcopy(seed); self.r['proposed_class'] = 3; self.write_review()
        with self.assertRaisesRegex(ValueError, 'Human adjudication override'):c.validate_reviews(self.root, self.out)

    def pending_human_addition(self):
        self.ca['signal'] = c.EXTRA
        c.save(self.out/'candidates.json', {'rows': [self.ca]})
        seed = copy.deepcopy(self.r)
        seed.update(source='human_verified', decision='add_box', geometry_ready=False,
                    historical_status='missing_label', historical_case=1, historical_batch='fixture',
                    uncertainty='human-confirmed object; visible box boundary still uncertain',
                    evidence_sufficiency='uncertain')
        c.save(self.out/'inherited.json', {'rows': [seed]}); self.freeze()
        self.r = copy.deepcopy(seed)
        self.r['observation'] = 'new actual visual evidence; boundary remains unresolved'
        self.write_review()

    def test_pending_human_geometry_can_retain_uncertainty(self):
        self.pending_human_addition()
        candidates, rows = c.validate_reviews(self.root, self.out)
        changes, deferred = c.proposal(candidates, rows)
        self.assertEqual(changes, [])
        self.assertEqual(deferred[0]['status'], 'pending_geometry')
        self.assertEqual(rows[0]['source'], 'human_verified')
        self.assertTrue(rows[0]['uncertainty'])

    def test_finalized_human_geometry_still_rejects_uncertainty(self):
        self.pending_human_addition()
        self.r.update(geometry_ready=True, final_xyxy=[256, 256, 384, 384], object_key='visible-object')
        self.write_review()
        with self.assertRaisesRegex(ValueError, 'Uncertain'):
            c.validate_reviews(self.root, self.out)

    def test_new_agent_pending_addition_still_rejects_uncertainty(self):
        self.ca['signal'] = c.EXTRA
        c.save(self.out/'candidates.json', {'rows': [self.ca]}); self.freeze()
        self.r.update(decision='add_box', geometry_ready=False, uncertainty='class unclear',
                      evidence_sufficiency='uncertain')
        self.write_review()
        with self.assertRaisesRegex(ValueError, 'Uncertain'):
            c.validate_reviews(self.root, self.out)

    def test_final_add_box_cannot_target_unrelated_location(self):
        self.ca['signal'] = c.EXTRA
        c.save(self.out/'candidates.json', {'rows': [self.ca]}); self.freeze()
        self.r.update(decision='add_box', final_xyxy=[0, 0, 20, 20], object_key='unrelated')
        self.write_review()
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            c.validate_reviews(self.root, self.out)

    def test_render_urls_default_and_nested_output(self):
        from urllib.parse import unquote
        for out in [self.out, self.out/'v2']:
            out.mkdir(exist_ok=True)
            c.render(self.root, out, [self.ca], [self.r])
            data = json.loads((out/'index.html').read_text(encoding='utf-8'))
            actual = (out/unquote(data['candidates'][0]['image_url'])).resolve()
            self.assertEqual(actual, (self.root/self.ca['image_path']).resolve())

    def test_unsupported_label_operation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            c.edited_label(b'', [dict(action='delete', new_class=1)])

    def test_apply_no_approval_writes_nothing(self):
        before = set(self.out.rglob('*'))
        with self.assertRaisesRegex(ValueError, 'phase-B approval'):c.apply(self.root, self.out, None)
        self.assertEqual(set(self.out.rglob('*')), before)

    def test_incomplete_review_cannot_apply(self):
        self.r.update(decision='unreviewed', visual_completed=False, source=None);self.write_review()
        c.summarize(self.root, self.out); approval = self.approval()
        with self.assertRaisesRegex(ValueError, 'not complete'):c.apply(self.root, self.out, approval)
        self.assertFalse((self.out/'dataset').exists())

    def test_approval_wrong_hash_rejected(self):
        c.summarize(self.root, self.out);p = self.approval();a = c.load(p);a['proposal_sha256'] = 'wrong';c.save(p, a)
        with self.assertRaisesRegex(ValueError, 'another proposal'):c.apply(self.root, self.out, p)
        self.assertFalse((self.out/'dataset').exists())

    def test_approval_must_follow_summary(self):
        c.summarize(self.root, self.out);p = self.approval();a = c.load(p);a['approved_at'] = '2000-01-01T00:00:00+00:00';c.save(p, a)
        with self.assertRaisesRegex(ValueError, 'must follow'):c.apply(self.root, self.out, p)

    def test_frozen_proposal_tampering_rejected(self):
        c.summarize(self.root, self.out);p = self.approval()
        pp = c.load(self.out/'proposed_changes.json');pp['rows'][0]['new_class'] = 3;c.save(self.out/'proposed_changes.json', pp)
        with self.assertRaisesRegex(ValueError, 'changed after summary'):c.apply(self.root, self.out, p)

    def test_protected_input_tampering_rejected(self):
        c.summarize(self.root, self.out);(self.root/'train/labels/a.txt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Protected input changed'):c.verify_review(self.root, self.out)

    def test_invalid_import_rolls_back(self):
        c.summarize(self.root, self.out);before = (self.out/'review.json').read_bytes()
        bad = copy.deepcopy(self.r);bad['proposed_class'] = 7;p = self.out/'batch.json';c.save(p, {'rows': [bad]})
        with self.assertRaises(ValueError):c.import_review(self.root, self.out, p)
        self.assertEqual((self.out/'review.json').read_bytes(), before)

    def test_synthetic_apply_preserves_source(self):
        c.summarize(self.root, self.out);source = self.root/'train/labels/a.txt';before = source.read_bytes()
        result = c.apply(self.root, self.out, self.approval())
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual((self.out/'dataset/labels/train/a.txt').read_bytes(), b'2'+before[1:])
        self.assertEqual(result['changed_files'], 1)
        with self.assertRaises(ValueError):c.apply(self.root, self.out, self.approval())


if __name__ == '__main__':
    unittest.main()
