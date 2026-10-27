# 五组2×扩样预检：单ROI覆盖条件未通过，尚未推理

## 本次限定批准

学习者同意排除已完成MVI_20011，选MVI_40181、MVI_40851、MVI_39401、MVI_40761、MVI_63521五组，全部492张train图，仅2×。每组从既有val966 bus热区中心固定一个320×320裁剪，双线性放大到640×640；若热区分散、单裁剪无法覆盖，先停下报告，不自行增加视野。无需逐图填写/导出JSON，以组级或区段目视意见为主。模型/参数原定沿用，不改标签、不训练。

## 实际检查结果

基于既有bus-static-review的16位编码计数图（R+256G）计算非零支持的半开包围框。计数图和清单SHA256匹配原归档；未新推理val、未读取test。

|组别|train图数|bus热区宽×高（像素）|按热区中心并限界的320裁剪xyxy|完整覆盖|
|---|---:|---|---|---|
|MVI_40181|107|640×640|[160,160,480,480)|否|
|MVI_40851|72|638×83|[160,0,480,320)|否|
|MVI_39401|80|488×543|[236,208,556,528)|否|
|MVI_40761|114|640×174|[160,42,480,362)|否|
|MVI_63521|119|187×320|[320,1,640,321)|是|

4组的支持包围范围至少一维超过320，任何单个320正方形都无法完整包含，不能仅移动裁剪解决。热图描述的是原标签占用区域，范围大不证明标签错误，也不证明train每张都有相同目标。

**按照批准的停止条件，本次模型启动0次、输入完成0项、权重未加载，没有生成候选或审核页。没有擅自只跑最后一组，也没有换组选成功例。**

## 证据与边界

- `preflight.json`：各组原始热区边界、来源大小/SHA256、拟定裁剪、覆盖判定及状态。
- `planned_train_manifest.json`：492张train完整文件路径、原图片与标签的既有哈希身份。只是从已验证清单冻结身份，本轮未重新扫描所有原文件哈希，未生成492个裁剪输入。
- 7项合成几何检查通过，含原试点、刚好320及321超界场景；这不是模型测试。
- `source_box_pixel_mass_covered_fraction`仅用于说明拟定裁剪保留的原标注像素支持量，**不是目标覆盖率、召回率或漏标率**。
- 本轮只读项目内现有清单/五张计数图/受保护入口与旧证据；运行环境为指定 `E:\anaconda\envs\dl-3lc\python.exe`，没有模型库导入、凭据访问、联网、安装依赖、修改图片/标签/旧报告、暂存或提交。

## 下一步需学习者选择（均未执行）

1. 保留五组，改为覆盖全图的多个320裁剪。仍然每个输入2×，但视野数与总预算要重新明确；无重叠可能截断边界目标，重叠会增加重复候选和审核负担。
2. 保持每图一个局部320裁剪，接受只考察该局部，不把ROI外目标算模型漏检；本轮无法再声称覆盖整个场景bus热区。
3. 保持单ROI完整覆盖要求，重新挑热区集中且符合条件的五组。会偏向位置集中的场景，不代表这四组或整体bus的效果。

选择后才继续，不自动替学习者决定新实验设计。

## 只读复验来源

在项目根运行以下命令可核对原统计图/清单来源未变并重算五组热区边界，不加载模型、不重跑推理：

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B -c "import hashlib,json,pathlib,numpy as np; from PIL import Image; r=pathlib.Path('reports/bus-static-review'); d=json.loads(pathlib.Path('reports/bus-crop-expansion-preflight-20261025/preflight.json').read_text(encoding='utf-8')); assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in d['source_sha256'].items()); [(lambda a,g: (lambda y,x: print(g['scene'],[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1],g['full_heat_covered']))(*np.nonzero(a[:,:,0].astype(np.uint16)+256*a[:,:,1].astype(np.uint16))))(np.asarray(Image.open(r/g['source_path'])),g) for g in d['groups']]"
```
