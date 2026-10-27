# 3LC Multi Vehicle Detection · 学习与离线诊断记录

本仓库记录四类车辆检测项目的官方 starter、数据审计工具、实验决策、离线诊断源码与轻量结果摘要。**它不是包含数据和权重的完整实验备份。**

## 目录

- `multi-vehicle_kaggle_competition_starter_kit_files/`：原 starter 代码与配置；使用说明见其 `README.md`。
- `tools/`：数据审计、只读可视化与获准的独立 OOF 诊断工具。
- `tests/`：工具的单元及回归测试，不依赖重新训练模型来验证。
- `reports/`：历史审计、baseline、OOF 指标、候选分析和初审文字记录。
- `.pi/plan/`：相关计划的历史记录；后来作出的明确覆盖决定见 `PROJECT_LOG.md` 和相应授权说明。
- `PROJECT_LOG.md`：真实操作、决策、问题来源与学习记录。
- `AGENTS.md`、`RULES.md`：协作边界和竞赛规则。

## 当前结果（2026-10-26）

- 五折 MVI/精确图像哈希分组 OOF 完成，五个独立随机初始化模型各训练 24 轮；每折独立 monitor 决定 best，best 轮次为 21 / 22 / 23 / 23 / 21。
- 7,852 张 train 图片各有唯一外折预测；后端只读复验通过。完整浏览器功能验收不冒充完成。
- 全体保存记录的自定义评估：mAP50 **0.7494**、mAP50–95 **0.5667**；匹配口径与限制见对应 `oof_metrics.json`，不是官方测试成绩或标签质量真值。
- 高置信切片整理出 176 条标签不一致信号、涉及 165 张图。它们不是 176 个已确认标签错误。
- 首批 8 张实际目视 AI 初审已记录（7 张来自上述切片，另 1 张补充代表），均待人工裁定，没有改标签。

主要结果入口：

- [OOF 结果](reports/oof-diagnostic-5fold-24e-retry1-20261026/RESULT.md)
- [OOF 指标](reports/oof-diagnostic-5fold-24e-retry1-20261026/oof_metrics.json)
- [标签不一致分析](reports/oof-label-diagnostic-20261026/ANALYSIS.md)
- [高置信候选 CSV](reports/oof-label-diagnostic-20261026/high_confidence_candidates.csv)
- [首批 AI 初审](reports/oof-ai-preliminary-review-20261026/REVIEW.md)

## Git 中保留什么

保留源码、测试、模板、计划、决策记录、轻量指标和候选表。`.gitignore` 明确排除：

- 原始数据、派生数据副本、3LC 本地状态、虚拟环境和 linked worktree；
- `.env`、凭据、私钥及机器会话配置；
- 模型权重、模型导出、缓存、压缩包和运行日志；
- 新诊断的原始逐图预测流、全量 manifest/哈希清单、进程 worker 配置；
- 竞赛图像、裁剪、审核截图和自动生成的本地 HTML 页面。

**忽略只影响版本控制，不删除本地文件。** 旧历史中已跟踪的审计清单继续保留；新诊断目录采用轻量产物白名单。没有用 Git LFS 隐式上传数据或权重。

因此，报告里部分本机绝对路径、截图和生成页链接在 GitHub/新 clone 上不可用，这是本地证据引用，不是远程已包含的附件。完整本地证据仍保留在原工作目录中；没有宣称单靠 clone 即可复现既有训练。分析阈值和审核结果均需结合原始输入核对。

## 运行与检查

本次已验证的本地环境：Python 3.12.15、PyTorch 2.14.1+cu130、Ultralytics 8.4.165、CUDA 13.0；GPU 为 RTX 4060 Laptop。以下命令假定已具备项目使用的环境，不会自动安装依赖：

```powershell
python -B -m unittest discover -s tests -p 'test_*.py'
git status --short
git check-ignore -v reports/oof-diagnostic-5fold-24e-retry1-20261026/oof_predictions.jsonl
```

部分运行/复验 helper 是当次本机、固定路径及截止时间的历史快照，并非通用启动器。不要为了复现报告直接再次运行 `prepare`、`run` 或恢复脚本；训练、数据、环境及授权边界应先核对。

本机若保留了已过截止时间的旧 OOF 报告，准备流程单测会读取旧 `PREVIOUS` 路径而出现时间依赖。此次发布检查在仅测试进程内隔离该路径后，214 项测试通过；未修改已封存源码、测试或真实运行预算。普通命令在原本机曾有这一项失败，新 clone 不包含旧运行配置。

## 使用边界

官方规则要求的竞赛流程仍以原 starter / 3LC 集成为准。独立 Ultralytics OOF 仅用于本次获准的离线诊断，**不代表其训练入口、模型集成、伪标签或其他用途已获得主办方许可**。仓库中没有以候选线索替代人工标签裁定，也没有自动清洗训练或竞赛提交。
