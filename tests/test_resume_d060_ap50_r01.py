"""No models or inference: bounded epoch63 recovery contracts."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def tool():
    p=ROOT/'tools/resume_d060_ap50_r01.py'
    assert p.is_file(),'Missing R01 epoch63 recovery executor'
    spec=importlib.util.spec_from_file_location('r01_recovery_test',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


class RecoveryContracts(unittest.TestCase):
    def test_only_remaining_training_and_original_evaluations(self):
        m=tool()
        self.assertEqual(m.stages(),[('train','R01'),('evaluate','R00'),('evaluate','R01')])
        a=m.authorization();self.assertEqual(a['resume_from_epoch'],63)
        self.assertEqual(a['remaining_epochs'],17);self.assertEqual(a['evaluations'],2)

    def test_checkpoint_binds_63_epochs_and_incumbent_best(self):
        m=tool();c=m.base().configured()
        rows=[dict(epoch=i,fitness=i/100,best_updated=True,metrics={'metrics/mAP50(B)':i/100}) for i in range(1,64)]
        ck=dict(epoch=62,optimizer={'state':{0:{}},'param_groups':[{}]},ema=object(),scaler={'scale':65536},
            updates=100,best_fitness=.63,train_metrics={'fitness':.63,'metrics/mAP50(B)':.63},
            train_args=m.base().training_options('R01'))
        m.validate_checkpoint(c,ck,rows)
        for key,value in [('epoch',61),('optimizer',None),('scaler',{}),('best_fitness',.62)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):m.validate_checkpoint(c,dict(ck,**{key:value}),rows)
        with self.assertRaises(RuntimeError):m.validate_checkpoint(c,ck,rows[:-1])

    def test_global_wrapper_keeps_old_paths_read_only(self):
        m=tool();b,c=m.configured()
        self.assertEqual(b.OUT,m.OUT);self.assertEqual(b.SELF,m.SELF)
        self.assertTrue(c.allowed_file(m.BASE_TOOL,False))
        self.assertFalse(c.allowed_file(m.PREVIOUS/'data/x',False))
        self.assertFalse(c.allowed_file(m.PREVIOUS/'R00/train/result.json',True))
        self.assertNotEqual(m.OUT,m.PREVIOUS)

    def test_epoch59_best_is_distinct_from_epoch63_last(self):
        m=tool();c=m.base().configured()
        rows=[dict(epoch=i,fitness=i/100,best_updated=True,metrics={'metrics/mAP50(B)':i/100}) for i in range(1,60)]
        rows += [dict(epoch=i,fitness=.58,best_updated=False,metrics={'metrics/mAP50(B)':.58}) for i in range(60,64)]
        ck=dict(epoch=62,optimizer={'state':{0:{}},'param_groups':[{}]},ema=object(),scaler={'scale':65536},
            updates=100,best_fitness=.59,train_metrics={'fitness':.58,'metrics/mAP50(B)':.58},
            train_args=m.base().training_options('R01'))
        m.validate_checkpoint(c,ck,rows)
        with self.assertRaises(RuntimeError):m.validate_checkpoint(c,dict(ck,best_fitness=.58),rows)


if __name__=='__main__':unittest.main()
