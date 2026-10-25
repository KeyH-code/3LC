# 审计至 baseline 归档索引（2026-10-25）

## 从哪里开始

1. [根项目日志](../PROJECT_LOG.md)：保留main/本分支起点已提交历史，只追加本阶段摘要。
2. [本批独立日志](../logs/score-search-20261019-150516.md)：原时间线与本次补记，区分学习者决定、人工观察及AI机械工作。
3. [审计摘要](data-audit/SUMMARY.md)、[补充审计](data-audit/FOLLOWUP_20261024.md)：16对确定同图与4862对近重复候选分别记录。
4. [人工反馈归档说明](data-audit/random-label-review-20261024/USER_FEEDBACK_ARCHIVE.md)：已经交回的24张已审结果，不是新审核任务。
5. [评估标准](EVALUATION_STANDARD_DRAFT.md)：C档官方隐藏测试集AP50≥0.90，不等于本地val分数。
6. [Starter baseline](baseline-starter-20261024/RESULT.md)：10轮、batch16、640、train7852/val966，最终AP50=0.7847740054/AP50-95=0.6068032980。
7. `git-preparation-20261025/COMMIT_MANIFEST.json`与`CHECKS.json`：本次精确提交白名单和提交前检查；不是一次新的模型运行。

## 当前有效决定

- 旧“将16张val移入train”方案已作废，`manual-review-20261024/confirmed_move_pending.json`是未执行历史方案，**不是待执行任务**。原train7852/val982文件及标签不动；实际取用val时只跳过16张完全同图。
- 新的项目相对路径清单为 `data-audit/random-label-review-20261024/val_use_list_excluding_16.relative.txt`，相对于starter kit根，包含966行。它由旧绝对路径清单机械转换、集合一致，非新抽样。
- 10月24日生成清单时尚未接入；本次baseline前已经派生独立3LC val966表。旧README中的“未接入”“均未审”是工具交付时点，不描述后来的训练/反馈。
- val足够可靠是学习者选定的工作假设，不改变已知数据证据；隐藏test真值未公开，质量未知。未清洗、未全量重标、未启动新实验。
- 两轮人工反馈均保留。随机反馈的原JSON共有96条，其中24张已审、72张未审；不要求再次审核，也不能将17/24外推全量或test。
- 当前AGENTS与main规则一致；历史临时授权仅为logs中的非生效附件。

## 官方规则核查记录（既有2026-10-24在线核查，非本轮重新联网）

来源：匿名只读页面；没有联系主办方、登录操作或提交。以下为当时已读取内容的归档，不冒充新截图或新核验：

- [官方Overview/Description](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/overview/description)：鼓励`inspect predictions, fix labels, and retrain`；Val行写`Dashboard review and optional label fixes`。
- [官方Data](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/data)：train/val有YOLO标签，test只给图片；数据来自UA-DETRAC经Roboflow改编。没有测试真值独立清洗正确的证明。
- [官方Rules](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/rules)及本地RULES.md §4.b：限制验证/测试手工标注信息，且基础规则声明冲突时优先。train许可不能扩成val/test通用许可。
- [主办方：训练标签改进](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/discussion/697032)：允许用外部脚本/方法找漏标或标签问题，前提只用所给数据并在重训前回写3LC新revision。提问范围为training set。
- [主办方：重新划分](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/discussion/702180)：`Merging train+val and making your own holdout is Allowed.`；允许所列条件下过采样。**本项目未采用合并后重划分，也未新增过采样。** 此答复不直接解决人工改val的规则冲突。
- [主办方：人工重标价值](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/discussion/747021)：强调按错误模式筛选处理，而非从第一张起逐框重画；未声明test标签已经清洗。

## 提交了什么，没有提交什么

**Git包含**：文本日志、结构化审计结果、文件哈希manifest、人工原反馈、模板/审计构建脚本、历史验证快照、相对val清单、本次baseline配置/CSV/args/小型监督状态、权重索引。

**仅保留本地（不是删除）**：

- 原数据与标签目录、模型best/last `.pt`、runs下曲线/混淆矩阵/批次图；3LC Tables/Runs/Parquet/嵌入；runtime和缓存。
- 两份自包含`review.html`（内嵌原图，约31.1MB和7.3MB）；`data-audit/evidence/`和`supplemental/evidence/`中的图片。旧文档链接到这些文件时，在新clone中将缺少目标。
- 完整console和资源采样日志、旧四次SS01–SS04完整产物和旧报告；其历史信息在独立日志保留，未将旧val982分数与新val966混比。
- 旧机器绑定的`tools/score_*.py`和`tools/baseline_starter_20261024.py`、`tools/verify_baseline_starter_20261024.py`。它们引用本地凭据路径/隔离缓存/旧预算；未复制凭据，也未在此任务改造成通用工具。
- 在线starter `config.yaml`的本地隔离项目/表名保持原样，**不提交**；入库的是报告配置快照，不改变main的默认配置。
- 原绝对val清单`val_use_list_excluding_16.txt`保留本地；原`val_use_list_metadata.json`的list_sha256仍指向该绝对清单，不改写它。新相对清单的哈希和对应关系另见其provenance JSON。

## 复现与验证的真实边界

- 本归档是证据/源码包，不是包含所有竞赛数据和认证状态的离线可运行包；合并后仍需另行安排数据/3LC表接线，未在本次授权中改入口或训练。
- 旧脚本部分使用脚本位置推导路径，另一些保留执行时scratch路径。查清路径和原始命令后才可复跑；不要在main直接执行旧复制/启动工具。补充审计脚本副本仍保留当时实际输出位置，不伪装曾在项目目录执行。
- 基线报告中的完整核验命令引用仅本地机械工具；在只有Git的新clone中不能直接运行该命令。当前提交检查只核对已有CSV/JSON/哈希和源码语法，不执行模型，也不宣称复跑了历史浏览器测试。
- 旧验证JSON是各自当时快照。例如C档确认后评估MD/官方JSON曾变更，旧followup验证中的文档哈希不会全匹配当前版本；不得为掩盖差异回写旧验证记录。
- 审阅README和verification中的“未审/未接入/未训练”描述各交付时点；当前状态以本索引、反馈原件和baseline记录为准。
- 所有文档中的外部账号/认证原件路径只是历史来源记录，不代表Git包含凭据或允许扩大读取范围。

## 归档方式与后续

根规则与main字节一致；根日志保留旧正文，新增摘要指向独立日志。提交只用明确文件白名单，按“审计决策”和“baseline证据”分两次。检查后才提交，Git身份沿用已有配置，不绕过钩子。

本次只commit后准备合并；不merge/push/stash/clean/reset，不删分支/worktree或本地证据。后续实际合并和训练开发另待学习者明确指令。

编码留痕：`supplemental_data_audit_execution.txt`和`supplemental_data_audit_postprocess_execution.txt`为原始UTF-16执行日志（非UTF-8源码/结果）。与完整运行日志一样仅本地保留、不入库，不转码或改写；相关结构化JSON、脚本和退出码记录照常归档。
