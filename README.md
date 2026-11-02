# 3LC Multi Vehicle Detection · 学习与离线诊断记录

本仓库记录四类车辆检测项目的官方 starter、数据审计工具、实验决策、离线诊断源码与轻量结果摘要。**它不是包含数据和权重的完整实验备份。**

## 目录

- `multi-vehicle_kaggle_competition_starter_kit_files/`：原 starter 代码与配置；使用说明见其 `README.md`。
- `tools/`：数据审计、审核/标签应用、获准的独立训练评估与提交文件工具。
- `tests/`：工具的单元及回归测试，不依赖重新训练模型来验证。
- `reports/`：历史审计、baseline、OOF 指标、候选分析和初审文字记录。
- `.pi/plan/`：相关计划的历史记录；后来作出的明确覆盖决定见 `PROJECT_LOG.md` 和相应授权说明。
- `PROJECT_LOG.md`：真实操作、决策、问题来源与学习记录。
- `AGENTS.md`、`RULES.md`：协作边界和竞赛规则。

## 当前结果（2026-11-02，三档实验完成，D060提交文件已生成）

SS04固定YOLOv8n、640输入、28轮；train7852张，统一验证集val966。原A/B结果复用，三档新模型各对D0/D085/self/val966完成12次FP32评估，交付后只读终验通过。

| 训练标签 | val AP50 | val AP50–95 | AP50–95相对D085（百分点） |
|---|---:|---:|---:|
| A / D0 | 0.809510 | 0.656022 | -1.371 |
| B / D085 | 0.828407 | 0.669736 | 基准 |
| D080 | 0.824848 | 0.662334 | -0.740 |
| D070 | **0.854866** | 0.678405 | +0.867 |
| D060 | 0.845502 | **0.678893** | +0.916 |

- 扩展6188候选已完成审核，批准后独立应用；D080/D070/D060相对D085累计351/1541/3185动作，73002/74080/75634训练框。原始D0/D085和val字节保留。
- **D080在服务中断后从第4轮checkpoint续至总28轮**，D070/D060从零训练；不保证续训等同不中断训练，原尝试和失败证据未覆盖。
- D070/D060总体有收益，但不严格单调，van AP50–95均低于B；固定参照train观察到D070后反升。单种子、val选best及候选组成变化限制因果解释，不宣称显著性。
- [完整实验报告](reports/train-cleaning-expansion-resume-r1-20261102/RESULT.md) · [分类收益与代价](reports/train-cleaning-expansion-resume-r1-20261102/class_deltas.csv) · [真实修改规模](reports/train-cleaning-expansion-resume-r1-20261102/change_scale.csv) · [标签应用报告](reports/cleaning-expansion-080-070-060-20261029/LABEL-APPLICATION-RESULT-20261101.md)。

### D060阶段性提交文件（尚无Kaggle成绩）

学习者指定D060 best对test982张执行一次推理，沿用starter参数conf=.25、iou=.7、imgsz640、batch1、max_det300、GPU0，FP32、无TTA/集成/阈值搜索。已生成982行、10557框的`submission.csv`，样例列名、原id/image_id及顺序、六元组/有限值/归一化范围核验通过。

- **CSV仅在本地**：`reports/submission-d060-20261102/submission.csv`（不纳入Git）。
- SHA256：`2751c6dc8603293605b2f1ffe1d6b844c1ba3c9fb547da3c9362da85eba87db5`。
- [生成及验证说明](reports/submission-d060-20261102/README.md)。由学习者手动上传，AI未上传Kaggle；没有官方Public/Private成绩或位次，原官方AP50≥.90目标尚未验收。
- CSV格式通过不代表训练工作流获得主办方许可；规定官方入口/3LC要求的适用性由学习者审阅，不将本次离线流程冒充已合规。

## 历史阶段：D085清洗（2026-10-28，当时未训练）

- 当前主线是冻结OOF confidence≥0.85的744条候选（591图、69场景）逐项目视审核，不是全7852图重标。
- 744/744条已有真实审核记录，未审/图像阻塞0；新意见与历史意见均已校验合并，14个人工漏标目标均已复看，12个框边界定稿、2个暂缓。
- 最终拟改450处/339个标签文件：补框392（car297、bus48、van15、truck32）、改类58；96条保持原样，196条暂缓及2个人工漏标边界待定保持不改。修改量不是错误率或收益估计。
- [最终规模报告](reports/oof-label-cleaning-conf085-20261027/FINAL_SCALE.md) · [当时停点记录](reports/oof-label-cleaning-conf085-20261027/NEXT_STEP.md)。逐项审核页`index.html`及图像仅本地保留，不是远程附件。
- 第08/10批旧文字乱码原件保留，另存重新实际看图的意见；新会话审核模型单独留痕，不冒称历史配置模型。
- 学习者已确认应用冻结450处清单；独立副本含7852图/7852标签，339份标签修订、7513份按字节不变，框总数72273→72665。原train/val不覆盖，196条暂缓及2条边界待定保持不改。
- [实际应用报告](reports/oof-label-cleaning-conf085-20261027/APPLIED_RESULT.md)：图像在`reports/oof-label-cleaning-conf085-20261027/dataset/images/train/`，标签在`dataset/labels/train/`；独立逐行/哈希复验通过，17812个受保护文件不变，图像字节一致、无硬链接。
- 此节描述2026-10-28的历史状态；后续清洗训练和扩展实验已经完成，见上方当前结果。离线实验未注册或替换3LC表。

## 已归档OOF结果（2026-10-26）

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
- 竞赛图像、裁剪、审核截图、自动生成的本地 HTML 页面及提交预测CSV。

**忽略只影响版本控制，不删除本地文件。** 旧历史中已跟踪的审计清单继续保留；新诊断目录采用轻量产物白名单。没有用 Git LFS 隐式上传数据或权重。

因此，报告里部分本机绝对路径、截图和生成页链接在 GitHub/新 clone 上不可用，这是本地证据引用，不是远程已包含的附件。完整本地证据仍保留在原工作目录中；没有宣称单靠 clone 即可复现既有训练。分析阈值和审核结果均需结合原始输入核对。

## 运行与检查

本次已验证的本地环境：Python 3.12.15、PyTorch 2.14.1+cu130、Ultralytics 8.4.165、CUDA 13.0；GPU 为 RTX 4060 Laptop。以下命令假定已具备项目使用的环境，不会自动安装依赖：

```powershell
python -I -B tools/cleaning_expansion_regression.py
git status --short
git check-ignore -v reports/oof-diagnostic-5fold-24e-retry1-20261026/oof_predictions.jsonl
```

部分运行/复验 helper 是当次本机、固定路径及截止时间的历史快照，并非通用启动器。不要为了复现报告直接再次运行 `prepare`、`run` 或恢复脚本；训练、数据、环境及授权边界应先核对。

本机若保留旧OOF报告，部分历史单测会读取`PREVIOUS`路径而出现时间依赖；上述回归helper仅在测试进程内隔离历史scratch/PREVIOUS，不改冻结源码或实验预算。当前包含CSV契约的373项合成测试通过；不以合成测试冒充模型实验或竞赛合规验收。

## 使用边界

官方规则要求的竞赛流程仍以原 starter / 3LC 集成为准。独立Ultralytics诊断、清洗对照和本次CSV生成仅按学习者具体授权执行，**不代表替代训练入口等用途已获得主办方许可**。审核后动作经学习者批准才应用，不自动采用预测当标签；未由AI上传竞赛提交，Git源码和轻量报告也不构成完整实验备份。
