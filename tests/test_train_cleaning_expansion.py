"""Synthetic contracts; never starts a training/evaluation process."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class ExpansionContracts(unittest.TestCase):
    def tool(self):
        path = ROOT/'tools/train_cleaning_expansion.py'
        self.assertTrue(path.is_file(), 'Missing bounded expansion executor')
        spec = importlib.util.spec_from_file_location('expansion_test', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_exact_approved_matrix_and_budgets(self):
        m = self.tool()
        self.assertEqual(m.BANDS, ('D080','D070','D060'))
        self.assertEqual(len(m.evaluation_matrix()),12)
        self.assertEqual(m.evaluation_matrix()[0], ('D080','D0'))
        self.assertEqual(m.evaluation_matrix()[-1], ('D060','val966'))
        self.assertEqual(m.timeout('train',10000,0),5400)
        self.assertEqual(m.timeout('evaluate',10000,0),3600)
        self.assertEqual(m.timeout('train',100,99),1)
        with self.assertRaises(TimeoutError): m.timeout('train',100,100)
        self.assertEqual(m.training_options()['epochs'],28)
        self.assertEqual(m.evaluation_options()['quantize'],None)

    def test_worker_guard_and_actual_weight_paths(self):
        m=self.tool()
        self.assertTrue(hasattr(m,'allowed_file'),'Missing offline read/write policy')
        self.assertTrue(m.allowed_file(m.OUT/'runtime/cache',True))
        self.assertFalse(m.allowed_file(m.DATA/'D080/dataset/labels/train/x.txt',True))
        self.assertTrue(m.allowed_file(m.PYTHON.parent/'Lib/site-packages/a.py',False))
        self.assertFalse(m.allowed_file(ROOT.parent/'other.txt',False))
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            m.OUT=Path(temp)
            weight=m.OUT/'D080/yolov8n/weights/best.pt'
            weight.parent.mkdir(parents=True);weight.write_bytes(b'local')
            m.save(m.OUT/'D080/train/result.json',dict(state='completed',epochs=28,best=str(weight),best_sha256=m.sha(weight)))
            self.assertEqual(m.weight_path('D080'),weight)
            weight.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):m.weight_path('D080')

    def test_auth_contract_requires_exact_once_counts_and_budget(self):
        m=self.tool()
        self.assertTrue(hasattr(m,'validate_authorization'),'Missing per-run authorization gate')
        a=dict(approved_by_user=True,training_runs=3,evaluations=12,
               models=list(m.BANDS),references=list(m.REFERENCES),
               training_timeout_seconds=5400,evaluation_timeout_seconds=3600,total_timeout_seconds=21600,
               no_retry=True,no_label_or_parameter_changes=True,no_test_evaluation=True,
               no_network_or_3lc=True,user_message='Approved this executor and budget')
        m.validate_authorization(a)
        for key,value in [('training_runs',4),('total_timeout_seconds',18000),('no_retry',False)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):m.validate_authorization(dict(a,**{key:value}))

    def test_never_restart_started_or_unsealed_run(self):
        m=self.tool()
        self.assertTrue(hasattr(m,'ready_to_run'),'Missing once-only run gate')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            m.OUT=Path(temp)
            with self.assertRaises(RuntimeError):m.ready_to_run()
            m.save(m.OUT/'supervision_started.json',dict(started=True))
            with self.assertRaisesRegex(RuntimeError,'repeat'):m.ready_to_run()

    def test_stage_directories_and_evaluation_options(self):
        m=self.tool()
        self.assertEqual(m.stage_path('train','D080'),m.OUT/'D080/train')
        self.assertEqual(m.stage_path('evaluate','D060','self'),m.OUT/'D060/eval_self')
        opts=m.evaluation_options()
        self.assertEqual((opts['conf'],opts['iou'],opts['max_det'],opts['batch']),(.001,.7,300,32))
        self.assertFalse(opts['rect']);self.assertFalse(opts['augment'])

    def test_every_worker_requires_two_guarded_spawn_children(self):
        m=self.tool()
        self.assertTrue(hasattr(m,'verify_child_guards'),'Missing actual child guard verification')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            work=Path(temp)
            m.save(work/'guard_ready_1.json',dict(module='__main__'))
            m.save(work/'guard_ready_2.json',dict(module='__mp_main__'))
            with self.assertRaises(RuntimeError):m.verify_child_guards(work)
            m.save(work/'guard_ready_3.json',dict(module='__mp_main__'))
            self.assertEqual(m.verify_child_guards(work),2)
            (work/'guard_events_3.jsonl').write_text('{}\n')
            with self.assertRaises(RuntimeError):m.verify_child_guards(work)

    def test_final_native_validation_loads_only_just_saved_checkpoint(self):
        m=self.tool()
        self.assertTrue(hasattr(m,'approved_asset'),'Missing current-run native-final-eval checkpoint binding')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            m.OUT=Path(temp);p=m.OUT/'D080/yolov8n/weights/best.pt';p.parent.mkdir(parents=True);p.write_bytes(b'saved')
            with self.assertRaises(RuntimeError):m.approved_asset(p)
            m.CHECKPOINTS[str(p.resolve())]=m.sha(p)
            self.assertEqual(m.approved_asset(p),str(p.resolve()))
            p.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):m.approved_asset(p)

    def test_native_optimizer_strip_refreshes_only_active_checkpoint_hashes(self):
        m=self.tool()
        self.assertTrue(hasattr(m,'native_final_eval'),'Missing optimizer-strip lifecycle binding')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            m.OUT=Path(temp);weights=m.OUT/'D080/model/weights';weights.mkdir(parents=True)
            best=weights/'best.pt';last=weights/'last.pt'
            best.write_bytes(b'best');last.write_bytes(b'last')
            def strip(path,**kwargs):
                path.write_bytes(path.read_bytes()+b'-stripped')
                return {'train_results':[]}
            engine=SimpleNamespace(strip_optimizer=strip)
            trainer=SimpleNamespace(best=best,last=last)
            def final():
                engine.strip_optimizer(last)
                engine.strip_optimizer(best,updates={'train_results':[]})
                self.assertEqual(m.approved_asset(best),str(best.resolve()))
                return 'native-metrics'
            self.assertEqual(m.native_final_eval(trainer,final,engine),'native-metrics')
            self.assertIs(engine.strip_optimizer,strip)
            self.assertEqual(m.approved_asset(best),str(best.resolve()))
            best.write_bytes(b'unexpected')
            with self.assertRaises(RuntimeError):m.approved_asset(best)


if __name__=='__main__': unittest.main()
