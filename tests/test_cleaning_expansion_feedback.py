"""Feedback assembly copies metadata, never generates visual judgments or receipt claims."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FeedbackAssemblyTests(unittest.TestCase):
    def tool(self):
        path = ROOT/'tools/cleaning_expansion_feedback.py'
        self.assertTrue(path.exists(), 'feedback assembler not implemented')
        spec = importlib.util.spec_from_file_location('feedback_assembly', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_explicit_seen_ids_and_original_opinions_are_required(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            key = 'a'*64
            manifest = dict(batch_id='D080-001-part01', candidate_keys=[key],
                            evidence=[dict(id='original-01', path='i.jpg', sha256='b'*64,
                                           candidate_keys=[key], kind='original')])
            p = root/'input.json'
            m.review.save(p, manifest)
            row = dict(key=key, category='keep', confidence='high', proposed_class=None,
                       final_xyxy=None, object_key=None, reason_code='model_error',
                       reason='Visible passenger car matches current label.', notes='',
                       visual_completed=True, evidence_ids=['original-01'])
            decisions = dict(results=[row], uncompleted_keys=[], seen_evidence_ids=['original-01'],
                started_at='2026-11-01T00:00:00+00:00', finished_at='2026-11-01T00:01:00+00:00')
            data = m.assemble(p, decisions, 'D080-001-part01-a01')
            self.assertEqual(data['results'], [row])
            self.assertEqual(data['vision_inputs'][0]['id'], 'original-01')
            self.assertTrue(data['vision_inputs'][0]['actual_image_received'])
            self.assertEqual(data['input_manifest_sha256'], m.review.sha(p))
            self.assertIsNone(data['reported_model'])
            with self.assertRaises(ValueError):
                m.assemble(p, dict(decisions, seen_evidence_ids=[]), 'a')


if __name__ == '__main__': unittest.main()
