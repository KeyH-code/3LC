# 三档标签已应用：数据版本交付，训练尚未授权

日期：2026-11-01。学习者在具体提案哈希汇报后明确“可以应用到标签”，授权文件为`LABEL-APPLICATION-APPROVAL-20261101.json`，SHA256 `b0f5a910b81b367317f64d84fb0f6f0490ec5e833b09f5d7ebf63ab351fe6b75`。

## 实际完成结果

每套均7852张训练图、966张验证图（11487个验证框），4类顺序固定truck/car/van/bus。完整独立复制，无硬链接；原始D0/D085、历史权重和审核/提案不改。

下表累计修改均相对D085（72665框），不把D085已有392补框/58改类重复计入：

| 版本 | 训练框数 | 累计新增框 | 累计改类 | 累计动作 | 实际变化标签文件 |
|---|---:|---:|---:|---:|---:|
| D080 | 73002 | 337 | 14 | 351 | 298 |
| D070 | 74080 | 1415 | 126 | 1541 | 1128 |
| D060 | 75634 | 2969 | 216 | 3185 | 2070 |

框数守恒：72665+337=73002；72665+1415=74080；72665+2969=75634。改类不改变框数。124个冲突排除key及全部keep/defer未应用；三档严格累计且对应批准提案。

与最初D0相比（包含此前D085的392补框/58改类）：D080/D070/D060分别累计729/1807/3361补框、72/184/274改类；实际变化标签文件573/1309/2207。文件差异已按D0源manifest哈希逐项计数，详见`label-application-d0-relative-verification-20261101.json`，不与相对D085的变化文件数混用。

| 版本 | truck | car | van | bus |
|---|---:|---:|---:|---:|
| D080 | 1259 | 59983 | 5575 | 6185 |
| D070 | 1256 | 61022 | 5515 | 6287 |
| D060 | 1259 | 62455 | 5546 | 6374 |

## 可直接使用的配置绝对路径

```text
E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\cleaning-expansion-080-070-060-20261029\datasets-20261101\D080\data.yaml
E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\cleaning-expansion-080-070-060-20261029\datasets-20261101\D070\data.yaml
E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\cleaning-expansion-080-070-060-20261029\datasets-20261101\D060\data.yaml
```

每个版本目录包含`dataset/images/train`、`dataset/labels/train`、`dataset/images/val`、`dataset/labels/val`；`manifest.json`保存成员及哈希，`applied_changes.json`保存具体动作和提案来源。`data.yaml`采用YAML兼容的JSON对象语法，包含绝对dataset根目录及相对train/val路径。配置存在不代表已启动/验收训练入口。

| 文件 | SHA256 |
|---|---|
| D080/data.yaml | `a150ddd5faa240bdea6e16818411d1d160c6221a926547ad5c852cdb910c00e0` |
| D070/data.yaml | `eaf8eb3a3ee54d18a6091e0122492edfb765ed828393ce5c6f237c5b5993791a` |
| D060/data.yaml | `92b44304b6aaabe5dc96c12077253ca4bf8cd8f81634ffd7f9929359f95ddd67` |
| application_seal.json | `572b98e3026a50697d18397f32e5e2562385f2c76269347f582de7d043dff20a` |

## 验证证据与原样保留

- 10项机械应用测试通过；全回归351项通过（30.334秒）。
- 应用完成时独立逐行重建类别替换/补框，验证原GT坐标/空格和未修改字节、补框坐标归一化、类数/框数/文件数及具体动作一致。
- 独立结构核验：52908个不同文件身份、每个链接计数1，实际数据文件1454281598字节。三套训练图ID/文件名/哈希顺序与A/B准确一致；没有把同帧同图哈希但不同文件名的成员去重丢弃。
- 固定验证标签原本存在贴边/越界框，本轮不裁剪、不重写；全套验证图片/标签逐字节保留。
- 三套`data.yaml`均经实际YAML解析器解析，train/val路径存在、类别映射正确；独立按D0原标签逐行复算补框和改类数量一致，原框几何全部保留。
- 应用后2026-11-01T09:47:32.914063Z，33520保护文件+20来源文件核验通过。准备工具报告label_writes=0仅描述该只读工具自身不写标签，不是本轮新派生数据写入计数。
- `datasets-20261101/verification.json`和`SUCCESS.json`记录应用核验；`label-application-independent-structure-verification-20261101.json`记录额外结构/顺序/独立文件验证。
- 已知预检问题和修正如实保存在`label-application-readiness-20261101.md`，未改旧封存工具、审核意见或提案。真实应用只执行一次，无失败重试/覆盖。

## 可复现只读验证命令

在项目根目录PowerShell执行（不要重复执行apply覆盖数据）：

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B tools/cleaning_expansion_apply.py verify --output reports/cleaning-expansion-080-070-060-20261029/datasets-20261101
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B tools/cleaning_expansion_prepare.py verify
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B tools/cleaning_expansion_regression.py
```

已使用的项目外资源仅为此前学习者指定的`E:\anaconda\envs\dl-3lc\python.exe`；未搜索其他项目外本机文件。

## 当前停点

本次标签应用与数据版本已交付。**训练/推理0次，未提交或推送Git**。原0.80/0.70/0.60及SS04规格保持不变。

停点B仍有效：本次“应用到标签”不自动授权三次训练、12次评估或完整执行器。后续运行准备、预算/次数及完整执行器需按原门禁确认后才能启动。
