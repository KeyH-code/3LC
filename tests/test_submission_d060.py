"""Synthetic submission contracts: no model imports, test images or GPU."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class SubmissionContracts(unittest.TestCase):
    def tool(self):
        path=ROOT/'tools/submission_d060.py'
        self.assertTrue(path.is_file(),'Missing D060 submission tool')
        s=importlib.util.spec_from_file_location('submission_d060_test',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        return m

    def test_prediction_encoding_six_fields_no_box_finite_and_range(self):
        m=self.tool()
        self.assertEqual(m.encode([]),'no box')
        text=m.encode([(1,.92,.55,.4,.12,.18),(3,.71,.2,.3,.08,.1)])
        self.assertEqual(len(text.split()),12)
        self.assertEqual(text.split()[:2],['1','0.920000'])
        for wrong in [[(4,.9,.5,.5,.1,.1)],[(0,float('nan'),.5,.5,.1,.1)],[(0,.9,1.1,.5,.1,.1)],[(1.5,.9,.5,.5,.1,.1)]]:
            with self.assertRaises(ValueError):m.encode(wrong)

    def test_csv_identity_order_headers_and_invalid_tokens(self):
        m=self.tool();self.assertTrue(hasattr(m,'verify_csv'),'Missing independent CSV validation')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            p=Path(temp)/'submission.csv';sample=[{'id':'8','image_id':'a','prediction_string':'no box'},{'id':'2','image_id':'b','prediction_string':'no box'}]
            def write(rows,fields=m.FIELDS):
                with p.open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
            rows=[dict(sample[0],prediction_string='1 0.9 0.5 0.5 0.1 0.2'),sample[1]];write(rows)
            self.assertEqual(m.verify_csv(p,sample)['boxes'],1)
            for wrong in [rows[::-1],rows[:1],[dict(rows[0],prediction_string='0 nan 0 0 0 0'),rows[1]],
                          [dict(rows[0],prediction_string='0 0.9 0.5'),rows[1]],
                          [dict(rows[0],prediction_string='0 0.9 0.5\n0.5 0.1 0.2'),rows[1]]]:
                write(wrong)
                with self.assertRaises(ValueError):m.verify_csv(p,sample)

    def test_test_membership_rejects_missing_and_duplicate_stems(self):
        m=self.tool();self.assertTrue(hasattr(m,'match_images'),'Missing exact test membership validation')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            d=Path(temp);(d/'a.jpg').write_bytes(b'fixture');rows=[{'id':'0','image_id':'a'}]
            self.assertEqual(m.match_images(rows,d),[d/'a.jpg'])
            (d/'a.png').write_bytes(b'other')
            with self.assertRaises(ValueError):m.match_images(rows,d)
            with self.assertRaises(ValueError):m.match_images([{'id':'0','image_id':'../a'}],d)


if __name__=='__main__':unittest.main()
