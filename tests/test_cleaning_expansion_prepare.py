"""Synthetic candidate-freeze tests; never constructs models or edits data labels."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ExpansionPrepareTests(unittest.TestCase):
    def tool(self):
        path=ROOT/'tools/cleaning_expansion_prepare.py'
        spec=importlib.util.spec_from_file_location('expansion_prepare_test',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

    def test_band_boundaries_exclude_prior_high_confidence_scope(self):
        m=self.tool()
        self.assertEqual([m.band(x) for x in [.85,.849999,.8,.799999,.7,.699999,.6,.599999]],
                         [None,'D080','D080','D070','D070','D060','D060',None])
        with self.assertRaises(ValueError):m.band(float('nan'))

    def test_partition_is_unique_and_deterministic(self):
        m=self.tool()
        rows=[dict(key='b',confidence=.7,scene='s',frame=2,id='i2',signal='extra',prediction_id='p2'),
              dict(key='a',confidence=.8,scene='s',frame=1,id='i1',signal='extra',prediction_id='p1')]
        result=m.partition(rows)
        self.assertEqual(result,m.partition(list(reversed(rows))))
        self.assertEqual([r['key'] for r in result['D080']],['a'])
        self.assertEqual([r['key'] for r in result['D070']],['b'])
        with self.assertRaises(ValueError):m.partition(rows+[copy.deepcopy(rows[0])])

    def test_stable_target_retains_line_identity_and_prior_change_flag(self):
        m=self.tool()
        source=dict(id='image',label_sha256='clean-label')
        lines=['1 .5 .5 .2 .2','2 .2 .2 .1 .1']
        original=[['0','.5','.5','.2','.2']]
        r=m.target_reference(source,lines,original,0)
        self.assertEqual(r['d085_target_status'],'approved_class_correction')
        self.assertEqual(r['original_label_line'],lines[0])
        self.assertEqual(r['stable_target_id'],m.target_reference(source,lines,original,0)['stable_target_id'])
        self.assertEqual(m.target_reference(source,lines,original,1)['d085_target_status'],'approved_added_box')
        with self.assertRaises(ValueError):m.target_reference(source,lines,original,2)

    def test_evidence_free_review_placeholders_do_not_claim_visual_completion(self):
        m=self.tool();r=m.unreviewed('candidate')
        self.assertEqual(r['decision'],'unreviewed')
        self.assertFalse(r['visual_completed']);self.assertEqual(r['evidence'],[])
        self.assertIsNone(r['actual_model']);self.assertFalse(r['geometry_ready'])

    def test_batches_do_not_mix_bands_and_cover_every_candidate_once(self):
        m=self.tool()
        grouped={'D080':[{'key':str(i)} for i in range(5)],'D070':[{'key':'x'}],'D060':[]}
        batches=m.batches(grouped,size=3)
        self.assertEqual([len(r['candidate_keys']) for r in batches],[3,2,1])
        self.assertEqual(len({k for r in batches for k in r['candidate_keys']}),6)

    def test_never_overwrites_frozen_artifact(self):
        m=self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path=Path(temp)/'freeze.json';m.save(path,{'a':1})
            with self.assertRaises(FileExistsError):m.save(path,{'a':2})
            self.assertEqual(m.load(path),{'a':1})


if __name__=='__main__':unittest.main()
