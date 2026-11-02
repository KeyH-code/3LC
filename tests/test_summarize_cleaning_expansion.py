import importlib.util
from pathlib import Path
import unittest
import tempfile
import json

ROOT=Path(__file__).resolve().parents[1]


class Comparisons(unittest.TestCase):
    def test_hypotheses_use_fixed_references_not_self_scores(self):
        path=ROOT/'tools/summarize_cleaning_expansion.py'
        self.assertTrue(path.exists(),'Missing frozen hypothesis reporting')
        s=importlib.util.spec_from_file_location('summary_expansion_test',path)
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        rows={b:{'val966':{'native':{m.AP:v}},'D0':{'native':{m.AP:t}},
                 'D085':{'native':{m.AP:t+.01}},'self':{'native':{m.AP:.99}}}
              for b,v,t in [('A',.6,.8),('B',.65,.79),('D080',.66,.78),('D070',.67,.80),('D060',.665,.81)]}
        result=m.compare(rows)
        self.assertFalse(result['H1_strict_monotonic_val'])
        self.assertTrue(result['H1_all_new_above_B'])
        self.assertTrue(result['H2_any_adjacent_decline'])
        self.assertTrue(result['H3_fixed_train_down_then_up']['D0'])
        self.assertAlmostEqual(result['gaps']['D060']['D0'],.145)
        self.assertAlmostEqual(result['delta_previous']['D060'],-.005)
        self.assertEqual(result['hypothesis_status']['H1_strict_monotonic'],'not_supported')
        self.assertEqual(result['hypothesis_status']['H2_observed_diminishing_or_decline'],'supported_descriptively')
        self.assertTrue(result['H2_diminishing_adjacent_gain'])

    def test_full_report_reads_18_evaluations_without_model_calls(self):
        path=ROOT/'tools/summarize_cleaning_expansion.py'
        s=importlib.util.spec_from_file_location('summary_fixture',path)
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp);m.ROOT=root;m.OUT=root/'new';m.OLD=root/'old';m.RECOVERY=root/'recovery'
            write=lambda p,v:(p.parent.mkdir(parents=True,exist_ok=True),p.write_text(json.dumps(v)))
            for i,b in enumerate(m.ORDER):
                t=dict(epochs=28,best_epoch=28,initial_state_sha256='same',best_sha256=b,best='weight',seconds=1)
                write((m.OLD if b in ('A','B') else m.OUT)/b/'train/result.json',t)
                for ref in (('A','B','val966') if b in ('A','B') else ('D0','D085','self','val966')):
                    row=dict(state='completed',weight_sha256=b,images=966 if ref=='val966' else 7852,seconds=1,
                        native={m.AP:.6+i*.01,m.AP50:.8,m.P:.9,m.R:.7},
                        classes=[dict(cls=0,name='truck',AP50=.8,AP50_95=.6,precision_best_f1=.9,recall_best_f1=.7)],
                        fixed=dict(classes=[dict(cls=0,TP=1,FP=1,FN=1,precision=.5,recall=.5)],precision_macro=.5,recall_macro=.5))
                    write((m.RECOVERY if b in ('A','B') else m.OUT)/b/('eval_'+ref)/'result.json',row)
            scale={b:dict(actions=1,added_boxes=1,changed_classes=0,changed_label_files=1,boxes=10) for b in m.ORDER[2:]}
            write(root/'reports/cleaning-expansion-080-070-060-20261029/datasets-20261101/application_seal.json',dict(versions=scale))
            m.main()
            self.assertTrue((m.OUT/'RESULT.md').is_file())
            self.assertEqual(len((m.OUT/'metrics.csv').read_text(encoding='utf-8-sig').splitlines()),19)
            self.assertEqual(len((m.OUT/'per_class.csv').read_text(encoding='utf-8-sig').splitlines()),19)
            self.assertIn('F1_at_native_point',(m.OUT/'metrics.csv').read_text(encoding='utf-8-sig'))
            self.assertIn('delta_previous_pp',(m.OUT/'change_scale.csv').read_text(encoding='utf-8-sig'))
            self.assertTrue((m.OUT/'class_deltas.csv').is_file())
            self.assertIn('supported_descriptively',(m.OUT/'RESULT.md').read_text())
            with self.assertRaises(FileExistsError):m.main()


if __name__=='__main__':unittest.main()
