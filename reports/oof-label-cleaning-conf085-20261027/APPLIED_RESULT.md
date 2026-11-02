# 阶段B：独立清洗版train已生成

本次只准备数据：未训练、未推理、未注册或替换3LC表。原始train/val保持不变。

## 可直接使用的数据路径

- 图像：`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\oof-label-cleaning-conf085-20261027\dataset\images\train`
- 标签：`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\oof-label-cleaning-conf085-20261027\dataset\labels\train`

目录为标准`images/train`与`labels/train`配对；图像和标签均为独立普通副本，不使用联动原件的硬链接。

## 实际改动

- 图片7852张、标签7852份；原图像素/文件字节均未变。
- 修改339个标签文件；其余7513份按字节保持。
- 新增392框：truck32、car297、van15、bus48。
- 改类58：van→car22、car→van33、van→bus2、truck→van1；原坐标和其他行保留。
- 原标签缺少末尾换行且需要追加框时，仅补必要的分隔换行；未改变原坐标或空格。
- 框数72273→72665。196条暂缓与2条边界待定不产生修改；同图其他已获准目标仍可修改。

| 类别ID | 类别 | 清洗后框数 |
|---|---|---:|
| 0 | truck | 1257 |
| 1 | car | 59671 |
| 2 | van | 5583 |
| 3 | bus | 6154 |

## 证据与复验

- 批准原话及范围：approval_record.json、phase-b-approval-20261028.json。
- 应用清单：applied_changes.json；全量图像/标签哈希：dataset_manifest.json。
- 独立复验：phase-b-independent-verification.json，含339份标签改前改后哈希及逐行验证结果。
- 冻结提案SHA256：`0f7aebb2d0dd68c8adc6f79f8f2bb4564356f5ed4754e06385e49bf9f72259e6`。
- dataset_manifest.json SHA256：`1898d635ab9808c00cab3ccc6dcbf30a564820393b91f260775204711de2c76e`。

项目根目录运行以下只读命令（不再次apply）：

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B reports\oof-label-cleaning-conf085-20261027\session-20261028\verify_applied_dataset.py
```

## 后续与回退

- 回退直接使用原始train目录/历史3LC表，不将派生标签覆盖回原目录。
- 当前未生成训练入口或改变3LC latest接线。后续应先由学习者确认A/B对照、实际配置、种子、运行次数、预算与评估口径。
- 标签修订完成不证明训练收益；本次数据批准不等于允许启动训练。
