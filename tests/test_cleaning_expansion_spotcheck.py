"""Evidence plates use unmodified source images and original saved opinions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SpotcheckTests(unittest.TestCase):
    def tool(self):
        path = ROOT/'tools/cleaning_expansion_spotcheck.py'
        self.assertTrue(path.exists(), 'evidence plate builder not implemented')
        spec = importlib.util.spec_from_file_location('spotcheck', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_crop_transform_preserves_target_and_stays_on_original_canvas(self):
        m = self.tool()
        roi = m.crop_roi([0,500,80,640])
        self.assertTrue(0 <= roi[0] < roi[2] <= 640)
        self.assertTrue(0 <= roi[1] < roi[3] <= 640)
        self.assertLessEqual(roi[0], 0)
        self.assertLessEqual(roi[1], 500)
        self.assertGreaterEqual(roi[2], 80)
        self.assertGreaterEqual(roi[3], 640)

    def test_plate_retains_actual_opinion_and_writes_exclusive_manifest(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            image = root/'image.png'
            m.Image.new('RGB', (640,640), 'gray').save(image)
            candidate = dict(key='a'*64, image_path='image.png', image_sha256=m.sha(image),
                id='i', scene='scene', frame=1, band='D080', xyxy=[10,20,40,60],
                predicted_class=1, gt_boxes=[], signal='prediction_without_label_coverage')
            opinion = dict(key=candidate['key'], category='add_box', proposed_class=1,
                final_xyxy=[11,21,41,61], confidence='high', reason='Visible car.', notes='')
            item = dict(candidate=candidate, opinion=opinion, source_event='original.json',
                        source_event_sha256='b'*64, group='disputed addition')
            output = root/'plate'
            record = m.render(root, [item], output, columns=1)
            self.assertEqual(record['items'][0]['opinion'], opinion)
            self.assertEqual(record['items'][0]['image_sha256'], m.sha(image))
            self.assertTrue((output/'overview.jpg').exists())
            self.assertTrue((output/'cards/01.png').exists())
            with self.assertRaises(FileExistsError): m.render(root,[item],output,columns=1)


if __name__ == '__main__': unittest.main()
