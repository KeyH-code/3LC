"""Compact text context does not replace real images or change frozen fields."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CompactTests(unittest.TestCase):
    def test_same_image_gt_is_shown_once_and_keys_stay_bound(self):
        path = ROOT/'tools/cleaning_expansion_compact.py'
        self.assertTrue(path.exists(), 'compact renderer not implemented')
        spec = importlib.util.spec_from_file_location('compact_review', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        shared = dict(id='image', image_path='image.jpg', image_sha256='a'*64,
                      gt_boxes=[dict(**{'class':1}, xyxy=[1,2,3,4])])
        rows = [dict(shared, key='b'*64, signal='class_disagreement', xyxy=[1,2,3,4],
                     original_class=1, predicted_class=2, gt_xyxy=[1,2,3,4], stable_target_id='d'*64),
                dict(shared, key='c'*64, signal='prediction_without_label_coverage', xyxy=[5,6,7,8], predicted_class=1)]
        manifest = dict(batch_id='D080-001-part01', candidate_keys=[r['key'] for r in rows], candidates=rows,
                        evidence=[dict(id='original-01', path='image.jpg', sha256='a'*64,
                                       candidate_keys=[r['key'] for r in rows])])
        compact = m.compact(manifest)
        self.assertEqual(len(compact['images']), 1)
        image = compact['images'][0]
        self.assertEqual(image['evidence_id'], 'original-01')
        self.assertEqual(image['gt_boxes'], shared['gt_boxes'])
        self.assertEqual([r['key'] for r in image['candidates']], manifest['candidate_keys'])
        self.assertNotIn('gt_boxes', image['candidates'][0])
        self.assertTrue(compact['actual_images_must_be_read_separately'])


if __name__ == '__main__': unittest.main()
