import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
import hashlib

ROOT=Path(__file__).resolve().parents[1]


class SummaryTests(unittest.TestCase):
    def tool(self):
        spec=importlib.util.spec_from_file_location('ab_summary', ROOT/'tools/summarize_cleaning_ab.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_same_reference_gaps_are_train_minus_val(self):
        m=self.tool()
        rows={arm:{ref:{'native':{'metrics/mAP50(B)':v, 'metrics/mAP50-95(B)':v-.1}}
                   for ref,v in [('A',a),('B',b),('val966',val)]}
              for arm,a,b,val in [('A',.9,.88,.8),('B',.87,.91,.82)]}
        result=m.comparisons(rows)
        self.assertAlmostEqual(result['val_B_minus_A']['metrics/mAP50(B)'], .02)
        self.assertAlmostEqual(result['gaps']['A']['A']['metrics/mAP50(B)'], .1)
        self.assertAlmostEqual(result['gaps']['B']['A']['metrics/mAP50(B)'], .05)
        self.assertAlmostEqual(result['gap_B_minus_A']['A']['metrics/mAP50(B)'], -.05)
        self.assertAlmostEqual(result['gaps']['B']['B']['metrics/mAP50(B)'], .09)

    def test_summary_refuses_to_publish_incomplete_run(self):
        m=self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            m.OUT=Path(temp)
            with self.assertRaises(FileNotFoundError): m.main()
            self.assertFalse((m.OUT/'RESULT.md').exists())

    def test_summary_writes_six_reference_rows_without_model_loading(self):
        m=self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            m.OUT=Path(temp)
            def write(path, data):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data),encoding='utf-8')
            write(m.OUT/'completed.json', {'protected_originals_unchanged':True})
            for arm in ['A','B']:
                weight=m.OUT/arm/'best.pt';weight.parent.mkdir();weight.write_bytes(b'fixture-not-a-model')
                h=hashlib.sha256(weight.read_bytes()).hexdigest()
                write(m.OUT/arm/'train/result.json',dict(epochs=28,best_epoch=28,best=str(weight),
                      best_sha256=h,initial_state_sha256='paired-fixture',seconds=1.))
                for ref in ['A','B','val966']:
                    write(m.OUT/arm/('eval_'+ref)/'result.json',dict(state='completed',weight_sha256=h,
                          images=966 if ref=='val966' else 7852, seconds=1.,
                          native={k:.5 for k in m.KEYS},fixed={'precision_macro':.6,'recall_macro':.7},
                          classes=[dict(cls=0,name='truck',AP50=.5,AP50_95=.4)]))
            m.main()
            self.assertEqual(len((m.OUT/'metrics.csv').read_text(encoding='utf-8-sig').splitlines()),7)
            self.assertIn('相同标签参照', (m.OUT/'RESULT.md').read_text(encoding='utf-8'))
            self.assertEqual(json.loads((m.OUT/'comparison.json').read_text())['comparison']['val_B_minus_A'][m.KEYS[3]],0.)
            with self.assertRaises(FileExistsError): m.main()


if __name__=='__main__':unittest.main()
