# 车辆检测与标签质量治理

这个项目关注道路场景中四类车辆检测的训练标签质量，覆盖卡车、汽车、厢式车和公交车。基于 3LC Multi Vehicle Detection Challenge 数据，固定 YOLOv8n 并从随机初始化训练。排查跨集合重复图像后，按视频场景和精确图像哈希做五折分组，用折外预测筛查疑似漏框与错类。经审核确认的修订写入独立的标签版本，保留原始数据。

## Kaggle 赛后评分

| Kaggle 榜单 | 最高 mAP@0.5 | 历史榜单估算位置 |
| --- | ---: | ---: |
| [Public](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/leaderboard?tab=public) | **0.80246** | **第 48** |
| [Private](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/leaderboard) | **0.78850** | **第 53** |

两个最高分均来自 G02 的赛后提交。表中位置按 97 支队伍的历史榜单分数估算。

## 数据清洗效果

在相同训练设置下，清洗前后的最佳 mAP@0.5 从 **80.95% 提升至 85.49%**，增加 **4.54 个百分点**。

| 标签版本 | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: |
| 清洗前最佳（D0） | 80.95% | 65.60% |
| 清洗后最佳（D070） | **85.49%** | **67.84%** |

两组均使用相同的 7,852 张训练图、YOLOv8n、640 像素输入和 28 轮训练设置，在同一组 966 张开发验证图上统一终评。表中最佳版本按 mAP@0.5 选取。 [本地指标明细](reports/metrics.csv)

## 快速运行

用合成标签和模拟折外预测演示候选筛选、审核与标签应用。示例只需 CPU 和 NumPy，无需下载模型或赛事数据。

已在 Windows、Python 3.13.5 和 NumPy 2.4.6 验证以下命令。请在仓库根目录运行。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m examples.label_review
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

示例应找到 3 条候选，应用 1 处改类和 1 个补框，按示例审核决定保留其余 1 条。输出写入 `demo-output/revised.txt`，原 `examples/sample.txt` 保持字节一致。再次运行时，通过 `--output` 指定一个新目录。合成回归覆盖候选匹配、审批校验、标签副本及结果验证。

## 结果复查

[实验与复查说明](reports/results.md)记录数据版本、实际预算、统一终评口径、赛后反馈来源和适用范围。[指标表](reports/metrics.csv)支持重算对照差值，[逐类结果](reports/per-class.csv)和 [G02 配置](reports/g02.json)可核对类别变化和参数。重跑 G02 终评还需要冻结 val966、相应权重和实际评估入口，这些材料保留在实验工作区。

val966 排除了 16 张与训练集完全重复的图片，仍有同源视频重叠，并用于选模。已有 van 诊断同时看到低置信同类框和 car/van 混淆。后续可用视频互斥的验证划分检验两种现象是否持续。 [数据来源与获取](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/data)
