"""Final proposal snapshots are immutable and impossible for incomplete review."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProposalExportTests(unittest.TestCase):
    def tool(self):
        path = ROOT/'tools/cleaning_expansion_proposals.py'
        self.assertTrue(path.exists(), 'proposal exporter not implemented')
        spec = importlib.util.spec_from_file_location('proposal_export', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_incomplete_review_creates_no_proposal_directory(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            candidate = dict(key='a'*64, band='D080')
            with self.assertRaises(ValueError):
                m.export(root/'ledger', [candidate], root/'proposals')
            self.assertFalse((root/'proposals').exists())

    def test_all_three_versions_bind_one_snapshot_and_are_never_apply_approval(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            store = root/'ledger'
            candidate = dict(key='a'*64, id='image', band='D080', image_path='image.jpg',
                             label_path='label.txt', scene='s', gt_boxes=[])
            row = dict(key=candidate['key'], category='add_box', proposed_class=1,
                       final_xyxy=[10,20,30,40], object_key='car-left')
            m.review.save(store/'accepted/event.json', dict(rows=[row]))
            out = root/'proposals'
            m.export(store, [candidate], out)
            seal = m.review.load(out/'seal.json')
            self.assertFalse(seal['label_application_authorized'])
            self.assertFalse(seal['training_authorized'])
            for band in ('D080', 'D070', 'D060'):
                data = m.review.load(out/(band+'.json'))
                self.assertEqual(data['review_snapshot_sha256'], seal['review_snapshot_sha256'])
                self.assertEqual(len(data['cumulative_actions']), 1)
                self.assertEqual(m.review.sha(out/(band+'.json')), seal['files'][band+'.json'])
            with self.assertRaises(FileExistsError): m.export(store, [candidate], out)


if __name__ == '__main__': unittest.main()
