import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SCRATCH=Path(r'E:\pi-desktop\data\data\scratch\112b9029-dd22-4fc5-9132-2dab3956e6cc')
spec=importlib.util.spec_from_file_location('confidence_review_tests',ROOT/'tools/bus_confidence_review.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def fixture():
    data=dict(experiment_id='batch',groups=[{'id':'A'},{'id':'B'}],samples=[]);records=[]
    for sid,scene,bus,scores in [('a','A',0,[[.25,.7],[.7]]),('b','A',0,[[.9]]),('c','B',1,[[.05,.99]])]:
        sample=dict(id=sid,scene=scene,split='train',bus_boxes=bus,views=[])
        for i,values in enumerate(scores):
            v=dict(id=f'{sid}:T{i+1}',sample_id=sid,scene=scene,frame=1,tile=f'T{i+1}',roi=[0,0,320,320],scale=2,input_file=f'inputs/{sid}_T{i+1}.png',input_file_sha256='filehash',input_rgb_sha256='rgbhash')
            candidates=[dict(id=f'{v["id"]}:p{j:03d}',confidence=c,cls=3,class_name='bus',touches_crop_edge=c in (.7,.99) and j==1) for j,c in enumerate(values)]
            record=dict(v,candidates=candidates);records.append(record)
            sample['views'].append(dict(v,input_url=v['input_file'],bus_candidates=[dict(c,max_original_bus_iou=0.) for c in candidates]))
        data['samples'].append(sample)
    return data,dict(experiment_id='batch',records=records)

class SummaryTests(unittest.TestCase):
    def setUp(self):self.data,self.pred=fixture()
    def test_levels(self):self.assertEqual(m.LEVELS,[.25,.4,.5,.6,.7,.8,.9,.95,.99])
    def test_inclusive_cutoff(self):
        self.assertEqual(m.summarize(self.data,.7)['total']['boxes'],4)
        self.assertEqual(m.summarize(self.data,.9)['total']['boxes'],2)
        self.assertEqual(m.summarize(self.data,.99)['total']['boxes'],1)
    def test_baseline_excludes_weak(self):self.assertEqual(m.summarize(self.data,.25)['total']['boxes'],5)
    def test_counts_images_not_duplicate_targets(self):
        total=m.summarize(self.data,.25)['total']
        self.assertEqual(total['images_with_candidates'],3);self.assertEqual(total['lead_images'],2);self.assertEqual(total['boxes'],5)
    def test_original_bus_label_excludes_lead_not_candidate(self):
        total=m.summarize(self.data,.99)['total']
        self.assertEqual(total['boxes'],1);self.assertEqual(total['images_with_candidates'],1);self.assertEqual(total['lead_images'],0)
    def test_edges_and_groups(self):
        result=m.summarize(self.data,.25)
        self.assertEqual(result['total']['edge_boxes'],2);self.assertEqual(result['groups'][0]['boxes'],4);self.assertEqual(result['groups'][1]['boxes'],1)
    def test_empty_at_one(self):self.assertEqual(m.summarize(self.data,1)['total']['boxes'],0)
    def test_monotonic_all_statistics(self):
        rows=[m.summarize(self.data,t)['total'] for t in m.LEVELS]
        for a,b in zip(rows,rows[1:]):self.assertTrue(all(b[k]<=a[k] for k in a))
    def test_no_mutation(self):
        old=copy.deepcopy(self.data);m.summarize(self.data,.9);self.assertEqual(self.data,old)
    def test_invalid_thresholds(self):
        for value in [None,'0.9',float('nan'),float('inf'),.05,-1,1.1,True]:
            with self.subTest(value=value),self.assertRaises(ValueError):m.summarize(self.data,value)

class IdentityTests(unittest.TestCase):
    def setUp(self):self.data,self.pred=fixture()
    def validate(self):return m.validate_data(self.data,self.pred,3,4)
    def test_accept(self):self.assertEqual(self.validate(),6)
    def test_batch(self):
        self.pred['experiment_id']='other'
        with self.assertRaises(ValueError):self.validate()
    def test_train_only(self):
        self.data['samples'][0]['split']='val966'
        with self.assertRaises(ValueError):self.validate()
    def test_missing_view(self):
        self.pred['records'].pop()
        with self.assertRaises(ValueError):self.validate()
    def test_duplicate_view(self):
        self.data['samples'][0]['views'][1]['id']=self.data['samples'][0]['views'][0]['id']
        with self.assertRaises(ValueError):self.validate()
    def test_duplicate_sample(self):
        self.data['samples'][1]['id']='a'
        with self.assertRaises(ValueError):self.validate()
    def test_duplicate_prediction_view(self):
        self.pred['records'][1]['id']=self.pred['records'][0]['id']
        with self.assertRaises(ValueError):self.validate()
    def test_changed_confidence(self):
        self.data['samples'][0]['views'][0]['bus_candidates'][0]['confidence']=.8
        with self.assertRaises(ValueError):self.validate()
    def test_changed_roi(self):
        self.data['samples'][0]['views'][0]['roi']=[1,1,321,321]
        with self.assertRaises(ValueError):self.validate()
    def test_missing_candidate(self):
        self.data['samples'][0]['views'][0]['bus_candidates'].pop()
        with self.assertRaises(ValueError):self.validate()
    def test_invalid_iou(self):
        self.data['samples'][0]['views'][0]['bus_candidates'][0]['max_original_bus_iou']=float('nan')
        with self.assertRaises(ValueError):self.validate()
    def test_url_rewrite_only(self):
        old=copy.deepcopy(self.data);new=m.page_data(self.data)
        self.assertEqual(self.data,old);self.assertEqual(new['confidence_levels'],m.LEVELS)
        for before,after in zip(old['samples'],new['samples']):
            for a,b in zip(before['views'],after['views']):
                self.assertEqual(b['input_url'],'../'+m.SOURCE.name+'/'+a['input_file'])
                self.assertEqual(b['bus_candidates'],a['bus_candidates'])
    def test_url_escape_rejected(self):
        self.data['samples'][0]['views'][0]['input_file']='inputs/../../private.png'
        self.data['samples'][0]['views'][0]['input_url']='inputs/../../private.png'
        with self.assertRaises(ValueError):m.page_data(self.data)

class SafetyTests(unittest.TestCase):
    def test_extract_data(self):
        text='<script id="expansion-data" type="application/json">{"s":"<\\/script>"}</script>'
        self.assertEqual(m.extract_data(text),{'s':'</script>'})
    def test_extract_rejects_missing_duplicate_bad_json(self):
        valid='<script id="expansion-data" type="application/json">{}</script>'
        for text in ['',valid+valid,valid.replace('{}','broken')]:
            with self.assertRaises(ValueError):m.extract_data(text)
    def test_build_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp,patch.object(m,'OUT',Path(tmp)),patch.object(m,'check_source',side_effect=AssertionError('too late')):
            with self.assertRaises(FileExistsError):m.build()
    def test_exclusive_json(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            p=Path(tmp)/'test.json';m.write(p,{'keep':True})
            with self.assertRaises(FileExistsError):m.write(p,{})
            self.assertEqual(m.read(p),{'keep':True})
    def test_cli_dispatch_mocked(self):
        for command in ['build','verify']:
            with patch.object(sys,'argv',['review.py',command]),patch.object(m,'build') as build,patch.object(m,'verify') as verify:
                m.main();{'build':build,'verify':verify}[command].assert_called_once_with()
                {'build':verify,'verify':build}[command].assert_not_called()
    def test_no_model_or_subprocess_imports(self):
        tree=ast.parse((ROOT/'tools/bus_confidence_review.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):self.assertTrue(all(x.name.split('.')[0] not in ('torch','ultralytics','tlc','tlc_ultralytics','subprocess') for x in node.names))
            if isinstance(node,ast.ImportFrom):self.assertNotIn((node.module or '').split('.')[0],('torch','ultralytics','tlc','tlc_ultralytics','subprocess'))
    def test_no_browser_storage_or_network_or_training(self):
        text=m.JS.read_text(encoding='utf-8')
        for forbidden in ['localStorage','fetch(','XMLHttpRequest','WebSocket','model.train','model.predict']:
            self.assertNotIn(forbidden,text)
    def test_single_marker_and_scope_labels(self):
        text=m.TEMPLATE.read_text(encoding='utf-8')
        self.assertEqual(text.count('__CONFIDENCE_DATA__'),1);self.assertIn('筛掉的候选数 ≠ 减少的误检数',text)
        self.assertIn('id="alltiles" type="checkbox" checked',text)
    def test_shared_original_renderer_and_threshold(self):
        text=m.JS.read_text(encoding='utf-8')
        self.assertIn('c.confidence>=t',text);self.assertEqual(text.count('paintOriginal(ctx,s,st,orig,heat)'),2)
        self.assertIn('paintOriginal(ca,s,st,orig,heat)',text)
    def test_clear_before_load_and_token_checks(self):
        text=m.JS.read_text(encoding='utf-8').split('async function draw(){',1)[1].split('const fmt=',1)[0]
        self.assertLess(text.index('ca.clearRect'),text.index('await Promise.all'))
        self.assertLess(text.index('cb.clearRect'),text.index('await Promise.all'))
        self.assertIn('if(token!==drawToken)return',text)

if __name__=='__main__':unittest.main()
