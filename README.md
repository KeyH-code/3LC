# 车辆检测与标签质量治理

基于 3LC Multi Vehicle Detection Challenge 的个人 CV 项目，检测道路场景中的卡车、汽车、厢式车和公交车。固定 YOLOv8n 并从随机初始化训练。先排查跨集合的重复图像，再用按视频场景和精确图像哈希分组的五折折外预测筛查疑似漏框与错类。审核后将修订应用到独立标签版本，保留原数据。

## 结果

| 本地对照 | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: |
| 原始标签（D0），28 轮 | 80.95% | 65.60% |
| 清洗标签（D060），28 轮 | 84.55% | 67.89% |
| D060 长周期对照（R01），80 轮 | 86.42% | 70.32% |
| D060 学习率对照（G02），80 轮 | **87.44%** | **70.84%** |

同为 28 轮时，清洗版的 mAP@0.5 提升 **3.60 个百分点**。固定 D060 并使用 80 轮配置时，G02 相对 R01 的 mAP@0.5 再提高 **1.02 个百分点**。训练集有 7,852 张图。终评使用冻结的 966 张开发验证图，含 11,487 个框。

G02 的 Kaggle **赛后提交**获得 Private **0.78850**、Public **0.80246**。相比旧 D060 赛后提交，Private 提升 1.677 个百分点。原目标为 Kaggle 隐藏测试 mAP@0.5 ≥ 0.90，G02 Private 距目标 11.15 个百分点。 [赛事与评分口径](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/overview)

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
