# 固定6188候选全量审核完成：三档累计提案，停点A

日期：2026-11-01。范围：原冻结6188候选、3360张原图、D085参照；阈值0.80/0.70/0.60及SS04不变。

## 结果

6188/6188候选均已形成有效审核记录；未完成0、blocked 0、quality-held 0。采用学习者已批准的子Agent意见，不重复视觉审核或追加保证式问答。请求模型为`cpa/gpt-6-luna-high`；运行工具未披露后端，6188条`reported_model=null`，不宣称已独立证明运行后端。

| 档位（增量区间） | 审核数 | 原始补框意见 | 改类意见 | keep | defer |
|---|---:|---:|---:|---:|---:|
| D080 [.80,.85) | 1197 | 364 | 14 | 162 | 657 |
| D070 [.70,.80) | 2543 | 1106 | 112 | 451 | 874 |
| D060 [.60,.70) | 2448 | 1623 | 90 | 507 | 228 |
| 合计 | 6188 | 3093 | 216 | 1120 | 1759 |

## 已排除冲突后的三档提案

累计均相对D085；不是相对最初D0，也不包括D085已有修订数量。每档新增指该置信区间贡献，累计为D080→D070→D060逐步追加。

| 版本 | 本档补框 | 本档改类 | 本档修改数 | 累计补框 | 累计改类 | 累计修改数 | 累计涉及标签文件 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D080 | 337 | 14 | 351 | 337 | 14 | 351 | 298 |
| D070 | 1078 | 112 | 1190 | 1415 | 126 | 1541 | 1128 |
| D060 | 1554 | 90 | 1644 | 2969 | 216 | 3185 | 2070 |

- 75条冲突记录涉及124个拟修改key，全部排除，不自动合并或覆盖：20条与D085已有框重叠，55条为同目标/补框相互重叠。原意见仍保留。
- 1759条defer与1120条keep均不修改。
- 对账：3185条保留修改 + 124条冲突排除 + 1759条defer + 1120条keep = 6188。
- 详细类别、场景分布和具体actions见封存`summary.json`及三个版本JSON。

## 冻结文件与SHA256

目录：`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\cleaning-expansion-080-070-060-20261029\review_v3\final-proposals-20261101\`

| 文件 | SHA256 |
|---|---|
| D080.json | `bf5256bea98117e04bb074fefd48e2ee1a5b34c492c0259a6d4d89207bd4bf67` |
| D070.json | `88189a9714e31858656a4fc5de4d92bbcff529e84c3448b92dc583b4240369fd` |
| D060.json | `30c59b11303f24a6f92cddbede727360ad6b169b714bf8c0f6bcd31af67cd182` |
| review_snapshot.json | `15d63394247aae213a318e83df0644aaa892946599e7d67bcd0a34ff4c436ca3` |
| conflicts.json | `2ad3a0bdc632133241e919ca529207abf3eab202f4962e9eff0d7a3573b86933` |

`seal.json`记录全部封存文件哈希。三个提案绑定同一最终审核快照，目录禁止覆盖。

## 验证证据

- 全回归：341项通过，34.306秒；合成测试不代表新增真实训练。
- 最终准备只读核验：2026-11-01T06:36:48.505803Z通过33520保护文件、20来源文件；6188候选/3360图/129原批不变。准备工具的visual_reviews=0仅指该工具自身不做视觉裁定，不是当前审核账本计数。
- `full-review/FINAL-REVIEW-IDENTITY-VERIFICATION.json`：6188唯一key、544有效事件、3360原图，4448个去重文件哈希及schema/manifest/证据关联/GT绑定通过；不是重复视觉复核。
- `full-review/FINAL-PROPOSAL-VERIFICATION.json`：所有封存文件哈希、同快照绑定、累计包含关系、动作数量及冲突排除结果重算一致。
- 路径、枚举、候选接线、类别编码和证据ID等客观修复均使用新attempt及差异，保留原件。已知时间占位/顺序错误保留来源限制；修复后的artifact观测时间不冒充实测视觉审核起止或时长。

复现账本及保护源验证（项目根目录PowerShell）：

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B tools/cleaning_expansion_review.py status
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B tools/cleaning_expansion_prepare.py verify
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B tools/cleaning_expansion_regression.py
Get-FileHash 'reports/cleaning-expansion-080-070-060-20261029/review_v3/final-proposals-20261101/D080.json' -Algorithm SHA256
Get-FileHash 'reports/cleaning-expansion-080-070-060-20261029/review_v3/final-proposals-20261101/D070.json' -Algorithm SHA256
Get-FileHash 'reports/cleaning-expansion-080-070-060-20261029/review_v3/final-proposals-20261101/D060.json' -Algorithm SHA256
```

## 当前门禁

本次已完成全部候选审核与三档提案封存，**尚未实际写入标签数据版本**。未启动真实推理/训练，未提交或推送Git。

停点A：等待学习者批准上述三个具体提案及哈希后，才应用标签。此次批准不自动启动三次SS04训练；数据验收和停点B继续有效。无需学习者逐项重审6188条图像。
