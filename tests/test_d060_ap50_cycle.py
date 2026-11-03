"""Synthetic contracts for the approved study; no training or model inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def tool():
    path = ROOT/'tools/train_d060_ap50_cycle.py'
    assert path.is_file(), 'Missing approved AP50 study executor'
    spec = importlib.util.spec_from_file_location('cycle_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class CycleContracts(unittest.TestCase):
    def test_exact_matrix_and_effective_optimizer(self):
        m = tool()
        self.assertEqual(m.RUNS, {'R00':28, 'R01':80})
        for run, epochs in m.RUNS.items():
            opts = m.training_options(run)
            self.assertEqual(opts['epochs'], epochs)
            self.assertEqual((opts['optimizer'],opts['lr0'],opts['momentum'],opts['warmup_bias_lr']),
                             ('AdamW',.00125,.9,0.))
            self.assertFalse(opts['resume']);self.assertFalse(opts['pretrained'])
            self.assertEqual(opts['patience'],100)
        with self.assertRaises(RuntimeError):m.training_options('R02')

    def test_raw_ap50_drives_fitness_even_below_prior_ap95(self):
        m = tool()
        class Native:
            def validate(self):
                self.best_fitness = .99  # Native AP95 side effect must not survive.
                return {'metrics/mAP50(B)':.85,'metrics/mAP50-95(B)':.99}, .99
        t = m.selection_factory(Native)()
        t.best_fitness = .850000001
        t.validator = SimpleNamespace(metrics=SimpleNamespace(box=SimpleNamespace(map50=.850000002)))
        metrics, fitness = t.validate()
        self.assertEqual(fitness,.850000002)
        self.assertEqual(t.best_fitness,fitness)
        self.assertTrue(t.ap50_improved)
        self.assertEqual(metrics['metrics/mAP50(B)'],fitness)
        t.validator.metrics.box.map50 = float('nan')
        with self.assertRaises(RuntimeError):t.validate()

    def test_tie_keeps_earlier_checkpoint_without_corrupting_serialized_best(self):
        m = tool()
        class Native:
            def validate(self):
                return {}, .99
            def save_model(self):
                blob=json.dumps({'best_fitness':self.best_fitness,'epoch':self.epoch}).encode()
                self.last.write_bytes(blob)
                if self.best_fitness==self.fitness:self.best.write_bytes(blob)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            t=m.selection_factory(Native)();t.best_fitness=None
            t.best=Path(temp)/'best.pt';t.last=Path(temp)/'last.pt'
            t.validator=SimpleNamespace(metrics=SimpleNamespace(box=SimpleNamespace(map50=.8)))
            for epoch,score in enumerate([.8,.8,.799,.80000000001,.80000000001],1):
                t.epoch=epoch;t.validator.metrics.box.map50=score
                _,t.fitness=t.validate();t.save_model()
            self.assertEqual(json.loads(t.best.read_bytes())['epoch'],4)
            self.assertEqual(json.loads(t.last.read_bytes()),{'best_fitness':.80000000001,'epoch':5})
            self.assertEqual(t.best.name,'best.pt')

    def test_trace_coverage_and_first_maximum_are_required(self):
        m=tool();rows=[dict(epoch=i+1,fitness=.5+i/1000,best_updated=True) for i in range(28)]
        self.assertEqual(m.validate_trace('R00',rows),28)
        with self.assertRaises(RuntimeError):m.validate_trace('R01',rows)
        rows[-1]['best_updated']=False
        with self.assertRaises(RuntimeError):m.validate_trace('R00',rows)

    def test_worker_read_write_boundaries(self):
        m=tool();c=m.configured()
        self.assertTrue(c.allowed_file(m.OUT/'runtime/x',True))
        self.assertFalse(c.allowed_file(m.SOURCE/'D060/train/result.json',False))
        self.assertFalse(c.allowed_file(ROOT.parent/'other.txt',False))
        self.assertFalse(c.allowed_file(m.SOURCE/'data/D060/labels/train/x.txt',True))
        self.assertTrue(c.allowed_file(m.SELF,False))

    def test_single_use_and_actual_80_epoch_weight_path(self):
        m=tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            m.OUT=Path(temp);c=m.configured()
            with self.assertRaises(RuntimeError):m.ready(c)
            c.save(m.OUT/'run_started.json',{})
            with self.assertRaisesRegex(RuntimeError,'repeat'):m.ready(c)
            p=m.OUT/'R01/yolov8n/weights/best.pt';p.parent.mkdir(parents=True);p.write_bytes(b'best')
            c.save(m.OUT/'R01/train/result.json',dict(state='completed',epochs=80,best=str(p),best_sha256=c.sha(p)))
            self.assertEqual(c.weight_path('R01'),p)
            p.write_bytes(b'drift')
            with self.assertRaises(RuntimeError):c.weight_path('R01')

    def test_authorization_rejects_extra_calls_or_old_deadline(self):
        m=tool();a=m.authorization();m.validate_authorization(a)
        for key,value in [('training_runs',3),('evaluations',3),('total_timeout_seconds',21600),('resume',True)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):m.validate_authorization(dict(a,**{key:value}))

    def test_dependency_drift_rejected_before_stage_directory(self):
        m=tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            m.OUT=Path(temp);c=m.configured();dep=m.OUT/'dependency.py';dep.write_text('old')
            c.save(m.OUT/'seal.json',dict(hashes={},dependencies={str(dep):c.sha(dep)}))
            dep.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'Dependency drift'):m.launch(c,'train','R00')
            self.assertFalse((m.OUT/'R00').exists())

    def test_strip_result_requires_checkpoint_trace_binding(self):
        m=tool()
        expected=dict(epoch=3,fitness=.80000000001,metrics={'metrics/mAP50(B)':.80000000001})
        with self.assertRaisesRegex(RuntimeError,'strip'):m.validate_stripped({},expected)
        payload=dict(model=object(),epoch=-1,optimizer=None,ema=None,scaler=None,updates=None,best_fitness=None,
            train_metrics=dict(expected['metrics'],fitness=expected['fitness']),
            train_results={'epoch':[1,2,3],'metrics/mAP50(B)':[.7,.75,.8]})
        m.validate_stripped(payload,expected)
        payload['train_metrics']['fitness']=.81
        with self.assertRaisesRegex(RuntimeError,'Checkpoint.*trace'):m.validate_stripped(payload,expected)

    def test_summary_uses_historical_cost_and_per_class_f1(self):
        m=tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            m.OUT=Path(temp);c=m.configured()
            v=dict(native={'metrics/mAP50(B)':.8,'metrics/mAP50-95(B)':.6},
                classes=[dict(cls=i,name=str(i),AP50=.8,AP50_95=.6) for i in range(4)],
                fixed=dict(precision_macro=.8,recall_macro=.5,classes=[dict(TP=2,FP=1,FN=3,precision=2/3,recall=.4) for i in range(4)]))
            c.save(m.OUT/'baseline_evaluation.json',v)
            c.save(m.OUT/'baseline_training.json',dict(best_epoch=28,seconds=2337.7508433))
            for run in m.RUNS:
                c.save(m.OUT/run/'train/result.json',dict(best_epoch=23,seconds=100))
                c.save(m.OUT/run/'eval_val966/result.json',v)
            m.summarize(c)
            self.assertIn('2337.75',(m.OUT/'RESULT.md').read_text(encoding='utf-8'))
            import csv
            with (m.OUT/'per_class.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
            self.assertEqual(float(rows[0]['fixed_F1']),.5)


if __name__=='__main__':unittest.main()
