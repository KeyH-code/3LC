from pathlib import Path
from collections import Counter
from decimal import Decimal
import json
OUT=Path(__file__).resolve().parent
DATA=Path(r'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516\multi-vehicle_kaggle_competition_starter_kit_files\data')
source=(OUT/'cleaning_audit.py').read_text(encoding='utf-8')
exec(source[source.index('def assignment(cost):'):source.index('def label_diff(a,b):')])
r=json.loads((OUT/'cleaning_audit.json').read_text(encoding='utf-8'))
def rows(rel):
 out=[]
 for ln,line in enumerate((DATA/rel).read_text(encoding='utf-8').splitlines(),1):
  if line.strip():
   tok=line.split();out.append({'line':ln,'class':int(tok[0]),'tokens':tok[1:],'coords':list(map(Decimal,tok[1:]))})
 return out
def rectangular(aa,bb):
 n=max(len(aa),len(bb))
 cost=[[float(max(abs(x-y) for x,y in zip(aa[i]['coords'],bb[j]['coords']))) if i<len(aa) and j<len(bb) else 0. for j in range(n)] for i in range(n)]
 return [(i,j) for i,j in assignment(cost) if i<len(aa) and j<len(bb)]
def compare(aa,bb,byclass):
 matched=[];useda=set();usedb=set()
 groups=sorted(set(x['class'] for x in aa+bb)) if byclass else [None]
 for c in groups:
  a=[x for x in aa if c is None or x['class']==c];b=[x for x in bb if c is None or x['class']==c]
  for i,j in rectangular(a,b):
   ar,br=a[i],b[j];delta=[abs(x-y) for x,y in zip(ar['coords'],br['coords'])]
   # All validation tokens checked separately below to be six decimals. A half-ULP of that precision is a stricter criterion than summing input precision.
   match={'train_line':ar['line'],'val_line':br['line'],'train_class':ar['class'],'val_class':br['class'],'train':ar['tokens'],'val':br['tokens'],'abs_delta':list(map(str,delta)),'max_delta':str(max(delta)),'rounding_6dp_compatible':max(delta)<=Decimal('0.0000005')}
   useda.add(ar['line']);usedb.add(br['line']);matched.append(match)
 return {'matches':matched,'unmatched_train':[{'line':x['line'],'class':x['class'],'coords':x['tokens']} for x in aa if x['line'] not in useda],'unmatched_val':[{'line':x['line'],'class':x['class'],'coords':x['tokens']} for x in bb if x['line'] not in usedb],'max_abs_delta':str(max((Decimal(m['max_delta']) for m in matched),default=Decimal(0))),'beyond_6dp_rounding_matched_boxes':sum(not m['rounding_6dp_compatible'] for m in matched),'changed_classes':[m for m in matched if m['train_class']!=m['val_class']]}
for i,p in enumerate(r['pairs'],1):
 a,b=rows(p['train']['label_rel']),rows(p['val']['label_rel'])
 p['labels']['val_all_tokens_six_decimals']=all(len(tok.split('.')[-1])==6 for x in b for tok in x['tokens'])
 p['labels']['class_preserving_rectangular']=compare(a,b,True)
 p['labels']['geometry_only_rectangular']=compare(a,b,False)
 cp=p['labels']['class_preserving_rectangular'];go=p['labels']['geometry_only_rectangular']
 print(i,p['train']['original_id'],'CLASS_PRESERVING max',cp['max_abs_delta'],'beyond',cp['beyond_6dp_rounding_matched_boxes'],'unmatched_T',json.dumps(cp['unmatched_train']),'unmatched_V',json.dumps(cp['unmatched_val']))
 for m in cp['matches']:
  if not m['rounding_6dp_compatible']: print(' GEOMETRY_DIFF',json.dumps(m))
 if go['changed_classes']:print(' GEOMETRY_ONLY_CLASS_DIFF',json.dumps(go['changed_classes']))
 print(' GEOMETRY_ONLY max',go['max_abs_delta'],'beyond',go['beyond_6dp_rounding_matched_boxes'],'unmatched_T',json.dumps(go['unmatched_train']),'unmatched_V',json.dumps(go['unmatched_val']))
(OUT/'cleaning_audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
vs=r['video_summary']; cross=[x for x in vs if x['train'] and x['val']]
print('VIDEOS',len(vs),'train',sum(x['train']>0 for x in vs),'val',sum(x['val']>0 for x in vs),'shared',len(cross),'shared_images',sum(x['train'] for x in cross),sum(x['val'] for x in cross))
print('TYPES',Counter(('count/class_different' if not p['labels']['class_counts_equal'] else 'rounding_only') for p in r['pairs']))
print('ALL_VAL_6DP',all(p['labels']['val_all_tokens_six_decimals'] for p in r['pairs']))