import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]


def tool():
    p=ROOT/'tools/resume_d060_ap50_cycle.py'
    assert p.is_file(),'Missing authorized recovery executor'
    s=importlib.util.spec_from_file_location('resume_cycle_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


class RecoveryContracts(unittest.TestCase):
    def test_checkpoint_requires_four_epochs_and_all_restorable_state(self):
        m=tool();c=m.study().configured()
        rows=[dict(epoch=i,fitness=i/10,metrics={'metrics/mAP50(B)':i/10},best_updated=True) for i in range(1,5)]
        args=m.study().training_options('R00')
        ck=dict(epoch=3,optimizer={'state':{0:{}},'param_groups':[{}]},ema=object(),scaler={'scale':65536},updates=4,
            best_fitness=.4,train_args=args,train_metrics={'fitness':.4,'metrics/mAP50(B)':.4})
        m.validate_checkpoint(c,ck,rows)
        for key,value in [('epoch',4),('optimizer',None),('scaler',{}),('updates',0),('best_fitness',.41)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):m.validate_checkpoint(c,dict(ck,**{key:value}),rows)

    def test_prior_trace_coverage_is_exact(self):
        m=tool();rows=[dict(epoch=i,fitness=i/10,best_updated=True) for i in range(1,5)]
        m.validate_prior(rows)
        with self.assertRaises(RuntimeError):m.validate_prior(rows[:3])
        rows[2]['best_updated']=False
        with self.assertRaises(RuntimeError):m.validate_prior(rows)

    def test_resume_relocates_paths_but_not_parameters(self):
        m=tool();args=SimpleNamespace(batch=32,epochs=28)
        m.relocate(args)
        self.assertEqual(args.batch,32);self.assertEqual(args.epochs,28)
        self.assertEqual(args.resume,str(m.OUT/'checkpoint/last.pt'))
        self.assertEqual(args.save_dir,str(m.OUT/'R00/yolov8n'))
        self.assertNotEqual(m.OUT,m.FAILED)


if __name__=='__main__':unittest.main()
