"""Saved-record counting only; no model imports or label changes."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class LadderTests(unittest.TestCase):
    def tool(self):
        spec=importlib.util.spec_from_file_location('ladder_test',ROOT/'tools/cleaning_threshold_ladder.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

    def row(self,key,image,pred,confidence,signal='class_disagreement'):
        return dict(key=key,id=image,prediction_id=pred,confidence=confidence,signal=signal,predicted_class=1)

    def test_counts_signals_unique_prediction_boxes_and_images_separately(self):
        m=self.tool()
        rows=[self.row('a','im1','p1',.85),self.row('b','im1','p1',.85),
              self.row('c','im2','p2',.8,'prediction_without_label_coverage')]
        r=m.count(rows)
        self.assertEqual((r['signals'],r['prediction_boxes'],r['images']),(3,2,2))
        self.assertEqual(r['class_conflict_signals'],2)
        self.assertEqual(r['uncovered_signals'],1)

    def test_boundary_increment_and_reviewed_overlap(self):
        m=self.tool()
        rows=[self.row('old','im1','p1',.85),self.row('new1','im1','p2',.8),
              self.row('new2','im2','p3',.75)]
        ladder=m.ladder(rows,[.85,.8,.75],{'old':'defer'},{'im1'})
        self.assertEqual(ladder[0]['total']['signals'],1)
        self.assertEqual(ladder[1]['total']['signals'],2)
        self.assertEqual(ladder[2]['band']['signals'],1)
        self.assertEqual(ladder[2]['below085']['signals'],2)
        self.assertEqual(ladder[2]['unreviewed']['signals'],2)
        self.assertEqual(ladder[2]['unreviewed']['images'],2)
        self.assertEqual(ladder[2]['never_reviewed_images'],1)
        self.assertEqual(ladder[2]['reviewed_decisions'],{'defer':1})
        self.assertEqual(ladder[2]['new_images_vs085'],1)

    def test_nonfinite_confidence_rejected(self):
        m=self.tool()
        with self.assertRaises(ValueError):m.count([self.row('a','i','p',float('nan'))])

    def test_lower_threshold_does_not_reselect_existing_conflict_prediction(self):
        import types
        m=self.tool();path=ROOT/'tools/oof_label_cleaning.py'
        helper=types.ModuleType('signals_test');helper.__file__=str(path)
        exec(compile(path.read_bytes(),str(path),'exec'),helper.__dict__)
        source=dict(id='i',scene='s',frame=1,image_path='i.jpg',label_path='i.txt',
                    image_sha256='image',label_sha256='label',labels=[[1,.5,.5,.2,.2]])
        rec=dict(fold=1,model_sha256='model',candidates=[
            dict(id='high',confidence=.85,cls=2,class_name='van',xyxy=[256,256,384,384]),
            dict(id='lower',confidence=.6,cls=2,class_name='van',xyxy=[256,256,384,384]),
            dict(id='separate',confidence=.8,cls=1,class_name='car',xyxy=[0,0,20,20])])
        minimum=helper.signals(source,rec,.25)
        for threshold in [.25,.6,.8,.85,.9]:
            direct={r['key'] for r in helper.signals(source,rec,threshold)}
            filtered={r['key'] for r in minimum if r['confidence']>=threshold}
            self.assertEqual(direct,filtered)


if __name__=='__main__':unittest.main()
