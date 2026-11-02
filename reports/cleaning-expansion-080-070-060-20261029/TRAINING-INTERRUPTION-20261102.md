# 服务重启后实验状态核对

## 实际观测

- 工具通知：整批运行shell `sh_197c481260014O6crraxw6udDU` 被取消，原文为`Command cancelled because the server restarted`。
- 学习者随后要求“继续”；本轮先做只读进程及产物核对，未将此推定为从头重训、checkpoint续训或预算重置的具体授权。
- 未发现命令行含`train_cleaning_expansion.py`的存活Python进程。
- D080已有`initialization.json`，初始state SHA256与原A/B一致。`epochs.jsonl`与`results.csv`均仅有第1–4轮完整记录，末条时间2026-11-01T15:58:42.511211+00:00。
- 第4轮训练期原生val AP50=.66031、AP50–95=.45293；这是未完成训练的过程指标，不是统一FP32评估，不与A/B最终best指标作结论比较。
- D080未见`train/result.json`、`train/exit.json`或failure.json；`stages.log`为空。不能认定28轮完成，也不能由文件缺失确定进程中断的精确时刻或是否已执行部分第5轮。
- D070、D060无stage目录；十二次统一评估均无结果。当前完整训练0/3，训练尝试1次（D080中断），统一评估0/12。

## 保留的checkpoint

共同绝对路径前缀：`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\train-cleaning-expansion-ss04-20261101\D080\yolov8n\weights\`。

| 文件 | 字节数 | 当前SHA256 |
|---|---:|---|
| best.pt | 24495591 | 6fe96d336204c1ce3189efdc49f73b8b78e9058242f0294ec6e7a4d43b445623 |
| last.pt | 24495591 | 6fe96d336204c1ce3189efdc49f73b8b78e9058242f0294ec6e7a4d43b445623 |

本轮仅读取文件元数据及原始字节计算哈希，未通过torch加载checkpoint、未检查内部优化器状态、未推理或训练。当前哈希用于固定中断现场，不能代替不存在的训练完成result哈希。

## 恢复边界（待学习者决定）

- 已封存执行器拒绝再次调用`run`，不删除或改写`run_started.json`、`supervision_started.json`及seal来绕过单次门禁。
- 选择D080从头重跑：需要明确批准新增一次训练尝试及保留本次失败/中断证据的新输出位置；属于协议偏离记录，不声称总共只尝试了三次。
- 选择从D080 last续训：需要明确批准一次`resume=true`偏离，并先检查checkpoint是否具备可恢复状态；不能称其完全等同于原定不中断的`resume=false`训练。
- 原整批上限为2026-11-01T15:40:01.440967+00:00起21600秒。恢复不得静默重置或沿用过期预算；需明确新的恢复预算，或明确如何扣除既往消耗。
- 原标签、模型、训练超参数、剩余矩阵不重新决策；不重开视觉审核、不修改旧代码/seal/权重，不Git提交或推送。

## 可复现只读检查

```powershell
Get-Content 'reports/train-cleaning-expansion-ss04-20261101/D080/train/epochs.jsonl'
Get-Content 'reports/train-cleaning-expansion-ss04-20261101/D080/yolov8n/results.csv'
Get-FileHash 'reports/train-cleaning-expansion-ss04-20261101/D080/yolov8n/weights/best.pt','reports/train-cleaning-expansion-ss04-20261101/D080/yolov8n/weights/last.pt' -Algorithm SHA256
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*train_cleaning_expansion.py*' } | Select-Object ProcessId, ParentProcessId, CommandLine
```

问题解决来源：`AI 直接解决`（只读状态核对、现场哈希及记录）；恢复方式尚未裁定，实验尚未完成。没有读取目录外文件。
