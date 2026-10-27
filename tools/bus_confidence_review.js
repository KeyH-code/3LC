'use strict';
const D=JSON.parse(document.getElementById('confidence-data').textContent);
const $=id=>document.getElementById(id), COLORS=['#fcad63','#70dcdf','#b0e66a','#ef8ac4'];
const levels=D.confidence_levels, baseline=levels[0], imageCache=new Map(), statsCache=new Map();
let threshold=baseline,group=D.groups[0].id,items=[],index=0,tile=0,page=0,timer=null,selected=null,drawToken=0,galleryToken=0;
const current=()=>items[index]||null;
const retained=(v,t=threshold)=>v.bus_candidates.filter(c=>c.confidence>=t);
const hasCandidates=(s,t=threshold)=>s.views.some(v=>retained(v,t).length>0);
function matches(s,filter,t=threshold){
  if(filter==='all')return true;
  if(filter==='lead')return s.bus_boxes===0&&hasCandidates(s,t);
  if(filter==='nobuslabel')return s.bus_boxes===0;
  if(filter==='hasbus')return hasCandidates(s,t);
  if(filter==='edge')return s.views.some(v=>retained(v,t).some(c=>c.touches_crop_edge));
  return false;
}
function summarize(t){
  if(statsCache.has(t))return statsCache.get(t);
  const groups=D.groups.map(g=>{
    const ss=D.samples.filter(s=>s.scene===g.id),cc=ss.flatMap(s=>s.views.flatMap(v=>retained(v,t)));
    return {scene:g.id,images:ss.length,boxes:cc.length,images_with_candidates:ss.filter(s=>hasCandidates(s,t)).length,
      lead_images:ss.filter(s=>s.bus_boxes===0&&hasCandidates(s,t)).length,edge_boxes:cc.filter(c=>c.touches_crop_edge).length};
  });
  const total={boxes:0,images_with_candidates:0,lead_images:0,edge_boxes:0};
  groups.forEach(g=>Object.keys(total).forEach(k=>total[k]+=g[k]));
  const result={threshold:t,groups,total};statsCache.set(t,result);return result;
}
function snapshot(){return {threshold,tile,alltiles:$('alltiles').checked,labels:$('labels').checked,preds:$('preds').checked,heat:$('heat').checked,selected};}
function scoped(s,st=snapshot()){
  const views=st.alltiles?s.views:[s.views[Math.min(st.tile,s.views.length-1)]];
  return views.flatMap(v=>retained(v,st.threshold).map(c=>({v,c})));
}
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='▶ 连续浏览';}
function image(url){
  if(imageCache.has(url))return imageCache.get(url);
  const promise=new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>{imageCache.delete(url);reject(Error('图片无法加载：'+url));};im.src=url;});
  imageCache.set(url,promise);if(imageCache.size>80)imageCache.delete(imageCache.keys().next().value);return promise;
}
function updateStats(){
  const now=summarize(threshold),base=summarize(baseline),removed=base.total.boxes-now.total.boxes;
  $('totals').textContent=`全五组：bus框 ${base.total.boxes} → ${now.total.boxes}（筛掉 ${removed}）；有候选 ${now.total.images_with_candidates}/492 图；原bus为0且有候选 ${now.total.lead_images}/181 图；触边框 ${now.total.edge_boxes}。均不是人工正确率。`;
  now.groups.forEach((g,i)=>{
    const card=document.querySelector(`.groupcard[data-group="${g.scene}"]`),b=base.groups[i];
    card.classList.toggle('active',g.scene===group);
    card.querySelector('.small').textContent=`${g.images}图 / ${D.groups[i].rois.length}视野每图\nconf≥${threshold.toFixed(2)}：${g.boxes}框（.25基准${b.boxes}）\n有候选${g.images_with_candidates}图 / 无原bus线索${g.lead_images}图\n触边${g.edge_boxes}框（可重复/截断）`;
  });
  [...$('ladder').rows].forEach(row=>row.classList.toggle('active',+row.dataset.threshold===threshold));
  const pos=levels.indexOf(threshold);$('confprev').disabled=pos===0;$('confnext').disabled=pos===levels.length-1;$('conf').value=String(threshold);
}
function refresh(preserve=false){
  stop();const old=preserve?current():null,oldTile=tile;
  items=D.samples.filter(s=>s.scene===group&&matches(s,$('filter').value));
  const found=old?items.findIndex(s=>s.id===old.id):-1;
  index=found>=0?found:0;tile=preserve?oldTile:0;if(items.length)tile=Math.min(tile,items[index].views.length-1);
  page=Math.floor(index/12);selected=null;
  $('selection-note').textContent=old&&found<0?`原帧${old.frame}不满足当前筛选；${items.length?'已切换到筛选结果首帧。':'当前筛选为空，可切回所有图片。'}`:'';
  $('sample').replaceChildren();items.forEach((s,i)=>$('sample').add(new Option(`${i+1}/${items.length} · 帧${s.frame}`,String(i))));
  $('position').max=String(Math.max(0,items.length-1));
  $('filtered').textContent=`conf≥${threshold.toFixed(2)}：筛选 ${items.length}/${D.groups.find(g=>g.id===group).images} 图；不是确认漏标数`;
  updateStats();render();gallery();
}
function setThreshold(value){
  const t=Number(value);if(!Number.isFinite(t)||!levels.includes(t))throw Error('不支持的显示阈值');
  threshold=t;refresh(true);
}
function chooseGroup(id){if(!D.groups.some(g=>g.id===id))throw Error('未知组');group=id;$('group').value=id;refresh();}
function choose(i,t=null){
  if(!Number.isInteger(i)||i<0||i>=items.length)return;
  if(t!==null&&(!Number.isInteger(t)||t<0||t>=items[i].views.length))throw Error('裁剪越界');
  index=i;if(t!==null)tile=t;
  tile=Math.min(tile,items[index].views.length-1);selected=null;$('selection-note').textContent='';
  page=Math.floor(index/12);
  render();gallery();
  document.querySelectorAll('.thumb').forEach(b=>b.classList.toggle('active',+b.dataset.index===index));
}
function rect(ctx,b,color,text,dash=false){
  ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash?[6,4]:[]);
  ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.setLineDash([]);ctx.font='13px sans-serif';ctx.fillText(text,Math.max(2,b[0]),Math.max(14,b[1]-3));ctx.restore();
}
function originalBoxes(s){return s.labels.filter(l=>l[0]===3).map(([,cx,cy,w,h])=>[(+cx-+w/2)*640,(+cy-+h/2)*640,(+cx+ +w/2)*640,(+cy+ +h/2)*640]);}
function paintOriginal(ctx,s,st,orig,heat){
  ctx.drawImage(orig,0,0,640,640);
  if(heat){ctx.save();ctx.globalAlpha=.6;ctx.drawImage(heat,0,0,640,640);ctx.restore();}
  s.views.forEach((v,n)=>rect(ctx,v.roi,COLORS[n],v.tile,n!==st.tile));
  if(st.labels)originalBoxes(s).forEach(b=>rect(ctx,b,'#57a8ff','原bus'));
  if(st.preds)scoped(s,st).forEach(({v,c})=>rect(ctx,c.xyxy_original,c.id===st.selected?'#ffef7f':'#75efad',`${v.tile} ${c.confidence.toFixed(2)}`));
}
async function draw(){
  const token=++drawToken,s=current(),st=snapshot(),ca=$('original').getContext('2d'),cb=$('crop').getContext('2d');
  ca.clearRect(0,0,640,640);cb.clearRect(0,0,640,640);window.ConfidenceReview.lastDraw=null;
  $('message').textContent='';if(!s)return;const v=s.views[st.tile],g=D.groups.find(g=>g.id===s.scene);
  try{
    const [orig,inp,heat]=await Promise.all([image(s.image_url),image(v.input_url),st.heat?image(g.heat_url):Promise.resolve(null)]);
    if(token!==drawToken)return;paintOriginal(ca,s,st,orig,heat);cb.drawImage(inp,0,0,640,640);
    if(st.labels)for(const b of originalBoxes(s)){
      const q=[Math.max(b[0],v.roi[0]),Math.max(b[1],v.roi[1]),Math.min(b[2],v.roi[2]),Math.min(b[3],v.roi[3])];
      if(q[2]>q[0]&&q[3]>q[1])rect(cb,[(q[0]-v.roi[0])*2,(q[1]-v.roi[1])*2,(q[2]-v.roi[0])*2,(q[3]-v.roi[1])*2],'#57a8ff','原bus（裁剪内部分）');
    }
    if(st.preds)retained(v,st.threshold).forEach(c=>rect(cb,c.xyxy_input,c.id===st.selected?'#ffef7f':'#75efad',`bus ${c.confidence.toFixed(2)}`));
    window.ConfidenceReview.lastDraw={sampleId:s.id,scene:s.scene,frame:s.frame,tile:v.tile,threshold:st.threshold,alltiles:st.alltiles,
      originalCandidateIds:st.preds?scoped(s,st).map(x=>x.c.id):[],cropCandidateIds:st.preds?retained(v,st.threshold).map(c=>c.id):[],originalLoaded:true,inputLoaded:true};
  }catch(e){if(token===drawToken)$('message').textContent=e.message;}
}
const fmt=a=>a.map(x=>Number(x).toFixed(2)).join(', ');
function render(){
  const s=current();$('prev').disabled=!s||index===0;$('next').disabled=!s||index===items.length-1;$('play').disabled=items.length<2;
  $('sample').value=String(index);$('position').value=String(index);$('tiles').replaceChildren();$('candidates').replaceChildren();
  $('scope').textContent=`当前conf≥${threshold.toFixed(2)}；左侧原图、缩略图、候选表：${$('alltiles').checked?'全部T叠加（不融合）':'仅当前T'}；右侧始终只显示当前T。${$('preds').checked?'':'候选显示已关闭，统计仍保留。'}`;
  if(!s){$('frameinfo').textContent='此筛选没有图片；可降低显示阈值或切回所有train图片。';$('source').textContent='';$('cropinfo').textContent='';$('croptitle').textContent='当前无图片';draw();return;}
  const v=s.views[tile];s.views.forEach((vv,n)=>{const b=document.createElement('button');b.textContent=vv.tile+' · 2×';b.classList.toggle('active',n===tile);b.onclick=()=>{stop();choose(index,n);};$('tiles').append(b);});
  $('source').textContent=s.image_path;$('croptitle').textContent=`${v.tile} · 实际2×输入 · conf≥${threshold.toFixed(2)}`;
  $('cropinfo').textContent=`ROI半开 [${v.roi}] · 原图320→输入640\n当前T保留${retained(v).length}框；全视野保留${s.views.reduce((n,vv)=>n+retained(vv).length,0)}框。可跨视野重复，触边需检查截断。`;
  $('frameinfo').textContent=`${s.scene} · 帧${s.frame} · 第${index+1}/${items.length}张 · 原bus ${s.bus_boxes}框 · ${s.views.map(vv=>`${vv.tile}:${retained(vv).length}框`).join('　')}`;
  const list=scoped(s);
  for(const {v:vv,c} of list){
    const tr=document.createElement('tr');tr.dataset.candidate=c.id;
    const text=[`${vv.tile}:${c.id.split(':').at(-1)}`,c.confidence.toFixed(4),fmt(c.xyxy_original),c.max_original_bus_iou.toFixed(4),fmt(c.edge_distances_original)+(c.touches_crop_edge?' ⚠触边':'')];
    text.forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td);});
    tr.style.cursor='pointer';tr.onclick=()=>{stop();const n=s.views.findIndex(vw=>vw.tile===vv.tile);choose(index,n);selected=c.id;draw();gallery();};$('candidates').append(tr);
  }
  if(!list.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=5;td.textContent='当前阈值/叠加范围内无bus候选，不代表整张图没有bus。';tr.append(td);$('candidates').append(tr);}
  draw();
}
async function gallery(){
  const token=++galleryToken,st=snapshot(),start=page*12,end=Math.min(start+12,items.length),sid=group,pg=page;
  $('gallery').replaceChildren();window.ConfidenceReview.lastGallery=null;
  $('pagenum').textContent=items.length?`${page+1}/${Math.ceil(items.length/12)}页 · ${start+1}—${end}`:'0/0页';
  $('pageprev').disabled=page===0;$('pagenext').disabled=end>=items.length;
  const entries=[],tasks=[];let errors=0;
  for(let i=start;i<end;i++){
    const s=items[i],button=document.createElement('button'),canvas=document.createElement('canvas');canvas.width=240;canvas.height=240;
    button.className='thumb';button.dataset.index=String(i);button.dataset.sample=s.id;button.classList.toggle('active',i===index);
    const shown=scoped(s,st),total=s.views.reduce((n,v)=>n+retained(v,st.threshold).length,0),caption=document.createElement('p');
    caption.textContent=`帧${s.frame} · 原图${st.alltiles?'全部T':s.views[st.tile].tile} · conf≥${st.threshold.toFixed(2)}\n原bus${s.bus_boxes} / 当前范围${shown.length}框 / 全视野${total}框${st.preds?'':'（候选显示关闭）'}`;
    button.append(canvas,caption);button.onclick=()=>{stop();choose(i);$('reader').scrollIntoView({behavior:'smooth',block:'start'});};$('gallery').append(button);
    const g=D.groups.find(g=>g.id===s.scene);
    tasks.push(Promise.all([image(s.image_url),st.heat?image(g.heat_url):Promise.resolve(null)]).then(([orig,heat])=>{
      if(token!==galleryToken)return;const ctx=canvas.getContext('2d');ctx.save();ctx.scale(240/640,240/640);paintOriginal(ctx,s,st,orig,heat);ctx.restore();
      entries.push({sampleId:s.id,candidateIds:st.preds?shown.map(x=>x.c.id):[]});
    }).catch(e=>{if(token===galleryToken){errors++;$('message').textContent=e.message;}}));
  }
  await Promise.all(tasks);if(token===galleryToken)window.ConfidenceReview.lastGallery={group:sid,page:pg,threshold:st.threshold,alltiles:st.alltiles,tile:st.tile,images:end-start,errors,entries};
}
levels.forEach(t=>$('conf').add(new Option(t.toFixed(2),String(t))));
D.groups.forEach(g=>{
  $('group').add(new Option(g.id,g.id));const b=document.createElement('button');b.className='groupcard';b.dataset.group=g.id;
  const title=document.createElement('strong'),desc=document.createElement('div');title.textContent=g.id;desc.className='small';b.append(title,desc);b.onclick=()=>chooseGroup(g.id);$('summary').append(b);
});
levels.forEach(t=>{
  const result=summarize(t),tr=document.createElement('tr');tr.dataset.threshold=String(t);const td=document.createElement('td'),button=document.createElement('button');
  button.textContent=t.toFixed(2);button.onclick=()=>setThreshold(t);td.append(button);tr.append(td);
  [result.total.boxes,summarize(baseline).total.boxes-result.total.boxes,result.total.images_with_candidates,result.total.lead_images,result.total.edge_boxes].forEach(value=>{const cell=document.createElement('td');cell.textContent=String(value);tr.append(cell);});$('ladder').append(tr);
});
$('conf').onchange=()=>setThreshold($('conf').value);
$('confprev').onclick=()=>setThreshold(levels[levels.indexOf(threshold)-1]);$('confnext').onclick=()=>setThreshold(levels[levels.indexOf(threshold)+1]);$('baseline').onclick=()=>setThreshold(baseline);
$('group').onchange=()=>chooseGroup($('group').value);$('filter').onchange=()=>refresh(true);
$('prev').onclick=()=>{stop();choose(index-1);};$('next').onclick=()=>{stop();choose(index+1);};
$('sample').onchange=()=>{stop();choose(+$('sample').value);};$('position').oninput=()=>{stop();choose(+$('position').value);};
$('play').onclick=()=>{if(timer){stop();return;}if(index===items.length-1)choose(0);$('play').textContent='Ⅱ 暂停';timer=setInterval(()=>{if(index>=items.length-1){stop();return;}choose(index+1);},+$('interval').value);};$('interval').onchange=stop;
$('pageprev').onclick=()=>{stop();if(page>0){page--;gallery();}};$('pagenext').onclick=()=>{stop();if((page+1)*12<items.length){page++;gallery();}};
$('jumpgo').onclick=()=>{stop();if(!items.length)return;const f=Number($('jump').value);let n=0;for(let i=1;i<items.length;i++)if(Math.abs(items[i].frame-f)<Math.abs(items[n].frame-f))n=i;choose(n);};
for(const id of ['labels','preds','alltiles','heat'])$(id).onchange=()=>{render();gallery();};
$('zoom').onclick=()=>document.querySelectorAll('.framewrap').forEach(el=>el.classList.toggle('pixel'));
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowRight'){stop();choose(index+1);}if(e.key==='ArrowLeft'){stop();choose(index-1);}});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
$('identity').textContent=`来源批次 ${D.experiment_id}\n显示阶梯 ${levels.join(', ')}\n推理参数未改变：conf=.05 / IoU=.7 / 固定1239输入\n原始来源 ${D.source_report}`;
window.ConfidenceReview={data:D,retained,matches,summarize,scoped,setThreshold,chooseGroup,choose,refresh,stop,
  getState:()=>({threshold,group,index,tile,page,current:current()?.id||null,items:items.map(s=>s.id),playing:timer!==null}),lastDraw:null,lastGallery:null};
refresh();
