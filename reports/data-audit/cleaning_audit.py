from pathlib import Path
from collections import defaultdict, Counter
from decimal import Decimal
import hashlib, json, re, itertools, sys
from PIL import Image, __version__ as pillow_version
import numpy as np
ROOT=Path(r'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516')
DATA=ROOT/'multi-vehicle_kaggle_competition_starter_kit_files'/'data'
OUT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'reports'/'data_manifest.json').read_text(encoding='utf-8'))
manifest={x['relative'].replace('\\','/'):x for x in manifest if x['relative'].replace('\\','/').split('/')[0] in ('train','val')}
errors=[]; verified=0; records=[]; all_disk=[]
def sha(b): return hashlib.sha256(b).hexdigest()
def check(rel,blob):
 global verified
 all_disk.append(rel)
 item=manifest.get(rel)
 if item is None: errors.append({'missing_manifest':rel}); return
 if len(blob)!=item['bytes'] or sha(blob)!=item['sha256']: errors.append({'manifest_mismatch':rel})
 else: verified+=1
for split in ('train','val'):
 images=sorted((DATA/split/'images').iterdir())
 labels=sorted((DATA/split/'labels').iterdir())
 label_names={x.name for x in labels if x.is_file()}
 for image in images:
  if not image.is_file(): continue
  rel=image.relative_to(DATA).as_posix(); blob=image.read_bytes(); check(rel,blob)
  m=re.fullmatch(r'(MVI_\d+)_img(\d+)_jpg\.rf\.([0-9a-f]+)\.[^.]+',image.name)
  if m is None: errors.append({'unparsed_filename':rel})
  label=DATA/split/'labels'/(image.stem+'.txt'); lb=label.read_bytes(); check(label.relative_to(DATA).as_posix(),lb)
  rows=[]
  for line_no,line in enumerate(lb.decode('utf-8-sig').splitlines(),1):
   if not line.strip(): continue
   p=line.split()
   if len(p)!=5: errors.append({'bad_label':str(label),'line':line_no}); continue
   rows.append({'line':line_no,'class':int(p[0]),'tokens':p[1:],'coords':[Decimal(x) for x in p[1:]]})
  records.append({'split':split,'name':image.name,'rel':rel,'label_rel':label.relative_to(DATA).as_posix(),'sha256':sha(blob),'bytes':len(blob),'video':m[1] if m else None,'frame':int(m[2]) if m else None,'original_id':m[1]+'_img'+m[2] if m else None,'rf':m[3] if m else None,'boxes':len(rows),'classes':dict(sorted(Counter(r['class'] for r in rows).items())),'rows':rows,'label_bytes':len(lb),'label_sha256':sha(lb),'label_text':lb.decode('utf-8-sig')})
  label_names.discard(label.name)
 if label_names: errors.append({'orphan_labels':sorted(label_names),'split':split})
missing_disk=sorted(set(manifest)-set(all_disk))
if missing_disk: errors.append({'manifest_missing_on_disk':missing_disk})
byhash=defaultdict(list); byvideo=defaultdict(list); byoriginal=defaultdict(list)
for r in records:
 byhash[r['sha256']].append(r); byvideo[r['video']].append(r); byoriginal[r['original_id']].append(r)
def digest_record(r):
 return {k:v for k,v in r.items() if k not in ('rows','label_text')}
def decode(r):
 with Image.open(DATA/r['rel']) as im:
  im.load(); arr=np.array(im); rgb=np.array(im.convert('RGB'))
  return {'size':list(im.size),'mode':im.mode,'format':im.format,'pixel_sha256':sha(arr.tobytes()),'rgb_pixel_sha256':sha(rgb.tobytes())}, arr, rgb
# Hungarian assignment, square cost matrix; minimizes total L-inf coordinate difference within class.
def assignment(cost):
 n=len(cost); u=[0.]*(n+1);v=[0.]*(n+1);p=[0]*(n+1);way=[0]*(n+1)
 for i in range(1,n+1):
  p[0]=i;j0=0;minv=[float('inf')]*(n+1);used=[False]*(n+1)
  while True:
   used[j0]=True;i0=p[j0];delta=float('inf');j1=0
   for j in range(1,n+1):
    if not used[j]:
     cur=cost[i0-1][j-1]-u[i0]-v[j]
     if cur<minv[j]:minv[j]=cur;way[j]=j0
     if minv[j]<delta:delta=minv[j];j1=j
   for j in range(n+1):
    if used[j]:u[p[j]]+=delta;v[j]-=delta
    else:minv[j]-=delta
   j0=j1
   if p[j0]==0:break
  while True:
   j1=way[j0];p[j0]=p[j1];j0=j1
   if j0==0:break
 return sorted((p[j]-1,j-1) for j in range(1,n+1))
def label_diff(a,b):
 ra=a['rows'];rb=b['rows']
 key=lambda r:(r['class'],*r['coords'])
 result={'exact_bytes_equal':a['label_sha256']==b['label_sha256'] and a['label_bytes']==b['label_bytes'],'exact_text_equal':a['label_text']==b['label_text'],'line_numeric_equal':[key(x) for x in ra]==[key(x) for x in rb],'unordered_numeric_equal':Counter(map(key,ra))==Counter(map(key,rb)),'class_counts_equal':a['classes']==b['classes'],'box_counts_equal':len(ra)==len(rb),'matches':[]}
 if not result['class_counts_equal']:return result
 diffs=[];beyond=[]
 for c in a['classes']:
  aa=[r for r in ra if r['class']==c];bb=[r for r in rb if r['class']==c]
  cost=[[float(max(abs(x-y) for x,y in zip(ar['coords'],br['coords']))) for br in bb] for ar in aa]
  for i,j in assignment(cost):
   ar,br=aa[i],bb[j];delta=[abs(x-y) for x,y in zip(ar['coords'],br['coords'])]; diffs.extend(delta)
   limits=[(Decimal(10)**Decimal(x.as_tuple().exponent)+Decimal(10)**Decimal(y.as_tuple().exponent))/2 for x,y in zip(ar['coords'],br['coords'])]
   bad=[k for k,(d,lim) in enumerate(zip(delta,limits)) if d>lim]
   entry={'train_line':ar['line'],'val_line':br['line'],'class':c,'train':ar['tokens'],'val':br['tokens'],'abs_delta':list(map(str,delta)),'rounding_limits':list(map(str,limits)),'beyond_rounding_coordinates':bad}
   result['matches'].append(entry)
   if bad:beyond.append(entry)
 result['max_abs_delta']=str(max(diffs,default=Decimal(0)));result['max_abs_delta_times640']=str(max(diffs,default=Decimal(0))*640)
 result['changed_matched_boxes']=sum(any(Decimal(d)!=0 for d in x['abs_delta']) for x in result['matches'])
 result['rounding_only_compatible']=not beyond
 result['beyond_rounding_boxes']=len(beyond)
 result['line_order_preserved']=all(x['train_line']==x['val_line'] for x in result['matches'])
 return result
pairs=[]
for h,group in sorted(byhash.items()):
 tr=[r for r in group if r['split']=='train'];va=[r for r in group if r['split']=='val']
 for a,b in itertools.product(tr,va):
  ai,ap,ar=decode(a);bi,bp,br=decode(b)
  pairs.append({'train':digest_record(a),'val':digest_record(b),'train_decode':ai,'val_decode':bi,'pixels_equal':np.array_equal(ap,bp),'rgb_pixels_equal':np.array_equal(ar,br),'labels':label_diff(a,b)})
pairs.sort(key=lambda p:(p['train']['video'],p['train']['frame'],p['val']['frame']))
video_summary=[]
for v,group in sorted(byvideo.items()):
 counts=Counter(r['split'] for r in group)
 video_summary.append({'video':v,'train':counts['train'],'val':counts['val'],'train_frame_range':[min([r['frame'] for r in group if r['split']=='train'],default=None),max([r['frame'] for r in group if r['split']=='train'],default=None)],'val_frame_range':[min([r['frame'] for r in group if r['split']=='val'],default=None),max([r['frame'] for r in group if r['split']=='val'],default=None)]})
original_cross=[]
for original,group in sorted(byoriginal.items()):
 tr=[r for r in group if r['split']=='train'];va=[r for r in group if r['split']=='val']
 if tr and va:
  unequal=[(a,b) for a,b in itertools.product(tr,va) if a['sha256']!=b['sha256']]
  original_cross.append({'original_id':original,'train':[digest_record(x) for x in tr],'val':[digest_record(x) for x in va],'equal_hash_pairs':len(tr)*len(va)-len(unequal),'different_hash_pairs':len(unequal)})
empty=[]
for r in records:
 if r['boxes']!=0:continue
 info,_,_=decode(r)
 train_same_hash=[digest_record(x) for x in byhash[r['sha256']] if x['split']=='train']
 train_same_original=[digest_record(x) for x in byoriginal[r['original_id']] if x['split']=='train']
 nearest={}
 for split in ('train','val','both'):
  candidates=[x for x in byvideo[r['video']] if x['boxes']>0 and (split=='both' or x['split']==split)]
  dist=min((abs(x['frame']-r['frame']) for x in candidates),default=None)
  nearest[split]={'distance_frame_id':dist,'files':[digest_record(x) for x in candidates if abs(x['frame']-r['frame'])==dist]}
  for side,pred in [('before',lambda x:x['frame']<r['frame']),('after',lambda x:x['frame']>r['frame'])]:
   sidec=[x for x in candidates if pred(x)];sd=min((abs(x['frame']-r['frame']) for x in sidec),default=None)
   nearest[split][side]={'distance_frame_id':sd,'files':[digest_record(x) for x in sidec if abs(x['frame']-r['frame'])==sd]}
 empty.append({'image':digest_record(r),'decode':info,'train_same_hash':train_same_hash,'train_same_original_id':train_same_original,'nearest_labeled':nearest})
result={'python':sys.version,'pillow':pillow_version,'numpy':np.__version__,'verified_files':verified,'manifest_files':len(manifest),'actual_files':len(all_disk),'errors':errors,'images':dict(Counter(x['split'] for x in records)),'boxes':{s:sum(x['boxes'] for x in records if x['split']==s) for s in ('train','val')},'cross_hash_groups':len({p['train']['sha256'] for p in pairs}),'pairs':pairs,'video_summary':video_summary,'cross_original_id_groups':original_cross,'empty_labels':empty}
(OUT/'cleaning_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','cross_original_id_groups','empty_labels')},ensure_ascii=False,indent=2))
print('PAIRS')
for i,p in enumerate(pairs,1):
 a,b,l=p['train'],p['val'],p['labels']
 print(i,a['name'],b['name'],'bytes',a['bytes'],'SHA',a['sha256'],'sizes',p['train_decode']['size'],p['val_decode']['size'],'pixels',p['pixels_equal'],'boxes',a['boxes'],b['boxes'],'classes',a['classes'],b['classes'],'labels',{k:v for k,v in l.items() if k!='matches'})
 for match in l['matches']:
  if any(Decimal(d)!=0 for d in match['abs_delta']):print(' DIFF',json.dumps(match))
print('ORIGINAL_CROSS',len(original_cross),'diff_hash_groups',sum(x['different_hash_pairs']>0 for x in original_cross),'diff_hash_pairs',sum(x['different_hash_pairs'] for x in original_cross),'equal_hash_pairs',sum(x['equal_hash_pairs'] for x in original_cross))
print('EMPTY',json.dumps(empty,ensure_ascii=False,indent=2))