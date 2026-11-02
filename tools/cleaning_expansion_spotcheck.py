"""Compose real-image evidence plates from frozen candidates and saved opinions.

No model calls, synthetic imagery, label writes, or opinion changes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT/'reports/cleaning-expansion-080-070-060-20261029/review_v3'
CLASSES = {0:'truck', 1:'car', 2:'van', 3:'bus'}


def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2)


def crop_roi(box):
    x1,y1,x2,y2 = box
    side = min(640, max(144, (x2-x1)*1.8, (y2-y1)*1.8))
    left = max(0, min(640-side, (x1+x2-side)/2))
    top = max(0, min(640-side, (y1+y2-side)/2))
    return [math.floor(left), math.floor(top), math.ceil(left+side), math.ceil(top+side)]


def select(root=ROOT, store=STORE):
    candidates = {r['key']:r for r in load(store.parent/'candidates.json')['rows']}
    held_names = {load(p)['event_file'] for p in (store/'quality_holds').glob('*visual-provenance.json')}
    held = []
    other = []
    for path in sorted((store/'accepted').glob('*.json')):
        event = load(path)
        for row in event['rows']:
            item = dict(candidate=candidates[row['key']], opinion=row,
                        source_event=path.relative_to(root).as_posix(), source_event_sha256=sha(path),
                        quality_held=path.name in held_names)
            (held if path.name in held_names else other).append(item)
    chosen = []
    images = set()
    # A deliberate coverage sample, not a random accuracy or failure-rate estimate.
    requests = [('disputed addition',held,'add_box',4),('comparison addition',other,'add_box',2),
                ('disputed keep',held,'keep',2),('class decision',other,'change_class',2),
                ('disputed defer',held,'defer',10)]
    used_events = set()
    for group, pool, category, count in requests:
        available = [r for r in pool if r['opinion']['category']==category]
        available.sort(key=lambda r:hashlib.sha256((group+r['opinion']['key']).encode()).hexdigest())
        added = 0
        for prefer_new_event in (True,False):
            for item in available:
                if item['candidate']['image_path'] in images: continue
                if prefer_new_event and item['source_event'] in used_events: continue
                chosen.append(dict(item,group=group))
                images.add(item['candidate']['image_path'])
                used_events.add(item['source_event'])
                added += 1
                if added==count: break
            if added==count: break
        if added!=count: raise ValueError('Insufficient unique sample images for '+group)
    if len(chosen)!=20: raise ValueError('Expected exactly20 real images')
    return chosen


def render(root, items, output, columns=4):
    output = Path(output)
    if output.exists(): raise FileExistsError('Evidence plate cannot be overwritten')
    prepared = []
    for item in items:
        candidate = item['candidate']
        image_path = (Path(root)/candidate['image_path']).resolve()
        if not image_path.is_relative_to(Path(root).resolve()): raise ValueError('Outside project')
        if sha(image_path)!=candidate['image_sha256']: raise ValueError('Source image hash drift')
        with Image.open(image_path) as source:
            if source.size!=(640,640): raise ValueError('Expected frozen640 image')
            prepared.append((item,source.convert('RGB')))
    output.mkdir(parents=True)
    (output/'cards').mkdir()
    font = ImageFont.load_default(size=17)
    heading = ImageFont.load_default(size=22)
    card_w,card_h,panel = 656,526,320
    overview = Image.new('RGB',(columns*card_w,104+math.ceil(len(items)/columns)*card_h),'#e5e7eb')
    draw = ImageDraw.Draw(overview)
    draw.text((16,10),'20 REAL IMAGE SPOT CHECKS - original saved Agent opinions, NOT new decisions',font=heading,fill='#111827')
    draw.text((16,44),'GREEN = existing GT   YELLOW = candidate clue / current target   CYAN = proposed added box',font=font,fill='#111827')
    draw.text((16,70),'Each card: full scene LEFT, same-image target zoom RIGHT. DEFER = leave label unchanged for now.',font=font,fill='#111827')
    records = []
    for index,(item,image) in enumerate(prepared,1):
        c,r = item['candidate'],item['opinion']
        target = c.get('gt_xyxy',c['xyxy'])
        roi = crop_roi(r['final_xyxy'] if r['category']=='add_box' else target)
        annotated = image.copy()
        a = ImageDraw.Draw(annotated)
        for gt in c['gt_boxes']:
            a.rectangle(gt['xyxy'],outline='#22c55e',width=2)
        a.rectangle(target,outline='#ffd400',width=4)
        if r['category']=='add_box':
            a.rectangle(r['final_xyxy'],outline='#00ddff',width=4)
        full = annotated.resize((panel,panel),Image.Resampling.LANCZOS)
        zoom = annotated.crop(roi).resize((panel,panel),Image.Resampling.LANCZOS)
        card = Image.new('RGB',(card_w,card_h),'white')
        d = ImageDraw.Draw(card)
        decision = r['category'].upper()
        if r['category']=='add_box': decision += ' '+CLASSES[r['proposed_class']]
        if r['category']=='change_class': decision += ' '+CLASSES[c['original_class']]+' -> '+CLASSES[r['proposed_class']]
        if r['category']=='keep' and 'original_class' in c: decision += ' '+CLASSES[c['original_class']]
        d.text((8,5),f'{index:02}  {decision}',font=heading,fill='#111827')
        d.text((8,33),c['band']+' | '+c['scene']+' frame '+str(c['frame'])+' | '+r['confidence'],font=font,fill='#374151')
        card.paste(full,(6,61));card.paste(zoom,(330,61))
        reason = r['reason'].encode('ascii','backslashreplace').decode('ascii')
        for line_no,line in enumerate(textwrap.wrap('Agent: '+reason, width=76)[:4]):
            d.text((8,389+line_no*21),line,font=font,fill='#111827')
        d.text((8,478),'Source: '+Path(item['source_event']).name,font=font,fill='#64748b')
        d.text((8,502),'Existing label: '+(CLASSES[c['original_class']] if 'original_class' in c else 'missing-label candidate'),font=font,fill='#64748b')
        card.save(output/'cards'/f'{index:02}.png')
        overview.paste(card,(((index-1)%columns)*card_w,104+((index-1)//columns)*card_h))
        records.append(dict(number=index,key=c['key'],image_path=c['image_path'],image_sha256=c['image_sha256'],
                            group=item['group'],source_event=item['source_event'],source_event_sha256=item['source_event_sha256'],
                            quality_held=item.get('quality_held',False),candidate=c,opinion=r,
                            crop_roi_original_xyxy=roi,zoom_size=[panel,panel],
                            transform='zoom_x=(original_x-left)*320/(right-left); same for y'))
    overview.save(output/'overview.jpg',quality=95,subsampling=0)
    for start in range(0,len(prepared),4):
        page = Image.new('RGB',(card_w*2,card_h*2),'#e5e7eb')
        for pos in range(min(4,len(prepared)-start)):
            with Image.open(output/'cards'/f'{start+pos+1:02}.png') as card:
                page.paste(card,((pos%2)*card_w,(pos//2)*card_h))
        page.save(output/f'page-{start//4+1:02}.jpg',quality=96,subsampling=0)
    record = dict(schema='real-image-human-spotcheck-v1',count=len(items),items=records,
                  selection='Coverage sample: disputed adds4, comparison adds2, disputed keep2, changes2, disputed defer10. Deterministic SHA order with unique source images and event diversity; not failure-rate estimate.',
                  labels_written=0,opinions_changed=False,overview_sha256=sha(output/'overview.jpg'))
    save(output/'manifest.json',record)
    links=''.join(f'<a href="page-{i:02}.jpg">Page {i}</a> ' for i in range(1,math.ceil(len(items)/4)+1))
    html='<meta charset="utf-8"><title>20 real-image checks</title><style>body{font:18px sans-serif;margin:20px;background:#eee}img{max-width:100%}a{margin-right:16px}</style><h1>20 real-image checks</h1><p>'+links+'</p><p>Green: existing GT. Yellow: candidate target. Cyan: proposed addition. Left full scene; right target zoom.</p><img src="overview.jpg"><h2>Individual cards</h2>'+''.join(f'<p><a href="cards/{i:02}.png">Card {i:02}</a></p>' for i in range(1,len(items)+1))
    with (output/'index.html').open('x',encoding='utf-8') as stream: stream.write(html)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT): raise ValueError('Output outside project')
    record=render(ROOT,select(),output)
    print(json.dumps(dict(count=record['count'],output=str(output),overview_sha256=record['overview_sha256']),indent=2))
