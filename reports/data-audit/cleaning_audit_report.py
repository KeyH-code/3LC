from pathlib import Path
from collections import defaultdict, Counter
import json,itertools,hashlib
O=Path(__file__).resolve().parent
ROOT=Path(r'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516')
DATA=ROOT/'multi-vehicle_kaggle_competition_starter_kit_files'/'data'
r=json.loads((O/'cleaning_audit.json').read_text(encoding='utf-8'))
m=json.loads((ROOT/'reports'/'data_manifest.json').read_text(encoding='utf-8'))
g=defaultdict(list)
for x in m:
 rel=x['relative'].replace('\\','/')
 if rel.startswith(('train/images/','val/images/')):g[x['sha256']].append(rel)
manifest_pairs=set()
for h,rels in g.items():
 for a,b in itertools.product([x for x in rels if x.startswith('train/')],[x for x in rels if x.startswith('val/')]):manifest_pairs.add((h,a,b))
actual_pairs={(p['train']['sha256'],p['train']['rel'],p['val']['rel']) for p in r['pairs']}
assert manifest_pairs==actual_pairs
for p in r['pairs']:
 assert (DATA/p['train']['rel']).read_bytes()==(DATA/p['val']['rel']).read_bytes()
 assert (DATA/p['train']['label_rel']).read_bytes()!=(DATA/p['val']['label_rel']).read_bytes()
r['independent_manifest_groups_confirmed']=len(manifest_pairs)
r['direct_byte_equality_all_16_pairs']=True
(O/'cleaning_audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
lines=[]
def w(s=''):lines.append(s)
w('# 只读数据清洗审计：实际副本独立复核')
w('## 1. 范围、方法与错误')
w(f'- 唯一项目范围：`{ROOT}`；只读取被允许的 dataset.yaml、reports/data_manifest.json、reports/DATA_DISTRIBUTION.md 和副本 data/train、data/val。未读 README（无必要），未读取测试数据、凭据、项目日志、其他历史产物、主目录或其他 worktree 内容。初始一次范围内 Glob 列出了根范围文件名元数据（含非目标产物名称），未打开这些非目标文件；后续均定向读取。')
w('- 初始只读子agent仅向会话scratch写脚本和结果；主agent将工具与证据归档到reports/data-audit。归档版仅刷新该目录内的JSON及报告，不修改数据、标签、划分、配置或训练入口；本轮未训练、安装、联网或启动3LC服务。')
w(f"- Python 仅 E:\\anaconda\\envs\\dl-3lc\\python.exe，{r['python']}；Pillow {r['pillow']}，numpy {r['numpy']}；使用 -B 禁止 bytecode 写入。")
w('- 从 manifest 的 train/val images 条目先按 SHA256 分组，独立使用当前磁盘全部图片字节重新分组并逐项比对，恰好一致的16组/16对；所有16对另作直接 bytes== 验证。')
w(f"- 共读取并重新核验 {r['verified_files']}/{r['manifest_files']} 个图片及标签文件的字节数与完整 SHA256；实际文件集合与 manifest 一致。7852 train + 982 val 图片，框数 72273/11615。无缺失/孤儿标签、命名解析失败、哈希/字节数不符。")
w('- 16对的32张图片及2张空标签图片均 Pillow Image.open -> load 完整解码，numpy 比较原生像素与 RGB 像素；全部16对尺寸640×640、RGB、JPEG且像素逐元素一致。其他图片仅读取字节/hash及文件名，不把本次称为所有图片解码检查。')
w('- 标签用原始字节/完整文本对比；非空行解析为 class + Decimal(cx,cy,w,h)。精确行顺序比较、精确无序多重集合比较；再按类做最小总 L∞ 坐标差矩形指派（自写 Hungarian，虚拟项承接多余框）。另做不按类别的几何指派，仅用于提示相近坐标/不同类别的条目，不能据此断言同一物体。')
w('- 16对的 val 坐标字段全部为6位小数；共同框逐字段绝对差<=0.0000005 为六位小数舍入相容（含半舍入方向差异），对应640输入参数差<=0.00032 px；不是精确标签相同。匹配之外的增减框、类别/几何变化不能用舍入解释。')
w('- 初始子agent统计与标签比较退出码均0、计算errors=[]。其报告生成最初因普通字符串反斜杠触发SyntaxWarning，修为raw字符串后退出码0；归档版执行情况见SUMMARY.md及run_*.log。第2步补齐所有16组矩形/跨类几何比对。')
w('- 复现顺序（仅重建本目录审计派生结果；不修改数据）：')
w('```powershell')
w("$py='E:\\anaconda\\envs\\dl-3lc\\python.exe'".replace('\\','\\'))
w(f"$s='{O}'")
w(r'& $py -B "$s\cleaning_audit.py"')
w(r'& $py -B "$s\cleaning_audit_labels.py"')
w(r'& $py -B "$s\cleaning_audit_report.py"')
w('```')
w('## 2. 16对逐组文件与标签证据')
w('- 类别映射来自 dataset.yaml:10–15：0=truck，1=car，2=van，3=bus。以下T/V为train/val；所有组去 .rf 散列后的 MVI/帧标识两侧一致，文件名不同。')
w('- **所有16对：图片完整字节/SHA256/尺寸/原生与RGB像素相同；标签原始字节和精确文本均不相同；精确有序/无序数值标签也均不相同。4对仅舍入相容；12对框数或类别计数有结构差异。**')
for i,p in enumerate(r['pairs'],1):
 a,b=p['train'],p['val'];l=p['labels'];cp=l['class_preserving_rectangular'];go=l['geometry_only_rectangular']
 w(f"### 组{i:02d} {a['original_id']}")
 w(f"- T图片：`{a['rel']}`")
 w(f"- V图片：`{b['rel']}`")
 w(f"- 两文件各 {a['bytes']} bytes，640×640 RGB/JPEG；SHA256=`{a['sha256']}`；直接字节及完整解码像素相同。")
 w(f"- T/V框数：{a['boxes']}/{b['boxes']}；T类别计数 `{a['classes']}`，V `{b['classes']}`。")
 w(f"- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 {a['label_bytes']} bytes，SHA256=`{a['label_sha256']}`；V标签 {b['label_bytes']} bytes，SHA256=`{b['label_sha256']}`。")
 if l['class_counts_equal']:
  w(f"- 仅六位小数舍入相容；同类别匹配全部保留行序，最大归一化差 {cp['max_abs_delta']}（×640=0.00032 px）；没有额外/缺少框。精确文本、精确数值有序和无序均不相同。")
 else:
  w(f"- 类别约束矩形匹配最大差 {cp['max_abs_delta']}；超过六位小数舍入的匹配框 {cp['beyond_6dp_rounding_matched_boxes']}；以下列出所有未匹配项（不能推定哪一侧标签正确）：")
  for split,k in [('T','unmatched_train'),('V','unmatched_val')]:
   for x in cp[k]:w(f"  - {split}独有行{x['line']}：class={x['class']} coords=({' '.join(x['coords'])})")
  for x in cp['matches']:
   if not x['rounding_6dp_compatible']:
    w(f"  - 同类最小代价关联 T:{x['train_line']}→V:{x['val_line']}：T=({' '.join(x['train'])})，V=({' '.join(x['val'])})；|Δ(cx,cy,w,h)|=({' '.join(x['abs_delta'])})；这是算法关联，不保证同一物体。")
  for x in go['changed_classes']:
   w(f"- **不按类几何关联** T行{x['train_line']} class{x['train_class']}→V行{x['val_line']} class{x['val_class']}：T=({' '.join(x['train'])})，V=({' '.join(x['val'])})；|Δ|=({' '.join(x['abs_delta'])})；最大差×640={float(x['max_delta'])*640:.8g}px。明显超过舍入，但不作为语义对应或标签真伪结论。")
  if not cp['beyond_6dp_rounding_matched_boxes']:w('- 除上述结构差异外，成功同类匹配的共同框各坐标差均<=5e-7。')
 w()
w('## 3. 文件名来源分组统计')
w(r'- 正则：`^(MVI_\d+)_img(\d+)_jpg\.rf\.([0-9a-f]+)\.[^.]+$`；8834/8834图片名解析成功；MVI完整编号为视频命名分组，MVI+原img编号为原始帧命名标识，数字差仅是命名帧号差。')
w('- 视频编号共100；train覆盖100，val覆盖99；**99个编号跨train/val**，涉及train7827/7852张、val982/982张。只有 MVI_20032 为train独占25张；val独占视频0。完整计数见下表。')
w('- 同原始帧命名标识跨组共16组、16个T×V配对，均为上述完全相同SHA的16对。**同原始帧标识跨组而图片SHA不同：0组、0对、0张。**')
w('- 完全重复集中于 MVI_40751 的5对，以及 MVI_40752 的11对。')
w('| MVI编号 | train图数 | val图数 |')
w('|---|---:|---:|')
for x in r['video_summary']:w(f"| {x['video']} | {x['train']} | {x['val']} |")
w('## 4. 两张空标签图定位')
for e in r['empty_labels']:
 a=e['image'];w(f"### {a['original_id']}")
 w(f"- 图片：`{a['rel']}`；MVI={a['video']}，img={a['frame']:05d}；640×640 RGB/JPEG，{a['bytes']} bytes；SHA256=`{a['sha256']}`。")
 w(f"- 标签：`{a['label_rel']}`；0 bytes、0框，SHA256=`{a['label_sha256']}`。")
 w('- train同图片SHA：0；train同原始帧命名标识：0。')
 for split in ('both','train','val'):
  near=e['nearest_labeled'][split]
  w(f"- {split}范围同视频最近非空标签图，帧号绝对差={near['distance_frame_id']}：")
  for x in near['files']:w(f"  - `{x['rel']}`（{x['boxes']}框，class计数{x['classes']}）")
  for side in ('before','after'):
   sidei=near[side]
   w(f"  - {side}最近：差={sidei['distance_frame_id']}；"+('; '.join(f"`{x['rel']}` / {x['boxes']}框" for x in sidei['files']) or '不存在'))
 w('- 上述距离仅命名帧编号差；未用图像内容判断是否真实相邻，不作漏标或正确负样本结论。')
w('## 5. 风险证据与未验证范围')
w('- **已证实**验证982张中16张（16/982=1.6293%）与训练图片字节和像素完全相同，构成跨划分输入重叠/验证独立性受限的直接证据；不能因文件名不同而忽略。')
w('- **已证实**16对标签均不精确相同：4对仅坐标舍入相容；12对出现不同框数量/类别结构，其中698、1286还包含超过舍入的几何/类别差异。不能把它们当作“图像和监督完全相同的16对”，也不能据此指定哪边漏标、错标。')
w('- **命名来源证据**：99个视频编号跨组，提示可能存在同源视频相关性；不能升级为已核实上游下载原件、真实视频身份、抽帧顺序/时间间隔或真正相邻帧视觉内容。')
w('- 未读/核验上游原始下载数据或视频；未读取测试集；未检测全量重编码同图、感知近重复或视觉相邻帧；未确认标签语义正确性、漏标/负样本真伪；未估算上述问题对任何指标的方向或幅度；未建议/执行具体删样本或重划分决策。')
w('- 原始标注是坐标数据，不是物体身份。几何最小代价匹配仅机械比较工具；跨类关联尤其不能称为已证实的同一实例重标注。')
w('## 6. 归档交付文件')
w(f"- 本文：`{O/'CLEANING_AUDIT.md'}`")
w(f"- 完整JSON（逐匹配行/全部差值/未匹配项/空图前后最近项）：`{O/'cleaning_audit.json'}`")
w('- 三个可复现脚本：cleaning_audit.py、cleaning_audit_labels.py、cleaning_audit_report.py；均在本报告所在目录。主agent目视结论与待处理清单另见SUMMARY.md。')
text='\n'.join(lines)+'\n'
# Correct the command literal to the exact authorized interpreter path.
text=text.replace("$py='E:\\\\anaconda\\\\envs\\\\dl-3lc\\\\python.exe'", "$py='E:\\anaconda\\envs\\dl-3lc\\python.exe'")
(O/'CLEANING_AUDIT.md').write_text(text,encoding='utf-8')
print('manifest_groups',len(manifest_pairs),'direct_bytes_equal',True,'report_lines',len(lines),'report_chars',len(text))
print('REPORT',str(O/'CLEANING_AUDIT.md'))