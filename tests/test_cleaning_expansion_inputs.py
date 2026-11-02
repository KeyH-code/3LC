"""Mechanical review input manifests preserve frozen ownership and source hashes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InputTests(unittest.TestCase):
    def test_subset_contains_only_uncompleted_keys_and_trims_evidence_binding(self):
        path = ROOT/'tools/cleaning_expansion_inputs.py'
        spec = importlib.util.spec_from_file_location('review_inputs_subset', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        rows = [dict(key='a'*64, image_path='i'), dict(key='b'*64, image_path='i')]
        manifest = dict(batch_id='D080-001-part01', candidate_keys=[r['key'] for r in rows], candidates=rows,
                        evidence=[dict(id='original-01', path='i', sha256='c'*64,
                                       candidate_keys=[r['key'] for r in rows])])
        subset = m.subset(manifest, ['b'*64], 'D080-001-part01-remaining01')
        self.assertEqual(subset['candidate_keys'], ['b'*64])
        self.assertEqual(subset['candidates'], rows[1:])
        self.assertEqual(subset['evidence'][0]['candidate_keys'], ['b'*64])
        self.assertEqual(subset['parent_chunk_id'], manifest['batch_id'])
        with self.assertRaises(ValueError): m.subset(manifest, ['d'*64], 'other')

    def test_input_chunks_group_images_without_changing_parent_band(self):
        path = ROOT/'tools/cleaning_expansion_inputs.py'
        self.assertTrue(path.exists(), 'input builder not implemented')
        spec = importlib.util.spec_from_file_location('review_inputs', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            image = root/'image.jpg'
            image.write_bytes(b'image')
            rows = [dict(key=str(i)*64, image_path='image.jpg', image_sha256=m.sha(image), band='D080')
                    for i in range(3)]
            parents = [dict(batch_id='D080-001', band='D080', candidate_keys=[r['key'] for r in rows])]
            m.build(root, root/'inputs', rows, parents, max_items=2)
            inputs = sorted((root/'inputs').glob('*/input.json'))
            self.assertEqual(len(inputs), 2)
            manifests = [m.load(p) for p in inputs]
            self.assertEqual([len(r['candidate_keys']) for r in manifests], [2, 1])
            self.assertEqual({r['parent_batch_id'] for r in manifests}, {'D080-001'})
            self.assertEqual(manifests[0]['evidence'][0]['candidate_keys'], ['0'*64, '1'*64])
            self.assertEqual(manifests[0]['candidates'], rows[:2])
            with self.assertRaises(FileExistsError):
                m.build(root, root/'inputs', rows, parents, max_items=2)


if __name__ == '__main__': unittest.main()
