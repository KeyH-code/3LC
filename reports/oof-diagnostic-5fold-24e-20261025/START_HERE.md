# 本次5折分组OOF离线诊断：入口与实时状态

> **当前实际状态：第一折架构加载失败，整批已停止且未重试。完成0折、0轮、0张OOF记录；后台模型任务已退出。请先看[失败报告](RESULT.md)。下面的入口说明保留本次提交和预检经过，不代表任务仍在训练。**

这是原始train 7852图的独立离线诊断，不是竞赛提交、自动判错或补标签流程。**后台已提交不等于五折完成**；以实际状态和最终验收证据为准。

## 当前状态从哪里看

- [实时进度](progress.json)：state、training_attempts、当前折/阶段、completed_folds、outer_images。启动前复验期间尚未增加training_attempts是正常现象。
- [宿主后台任务](host_supervision.json)：本次只提交一次run；任务ID `216d6692-ba2f-4faa-abbe-796bede8b3f3`，日志保存在本会话scratch。
- `fold_01`至`fold_05`的 `train_stage/console.log` 是实际子进程训练日志，`stopper_trace.jsonl` 是逐轮monitor fitness与原生早停轨迹；只有实际创建后才有内容。
- `run_status.json` 由执行器在整批结束/失败后创建。失败保留锁及部分结果，不自动重试。

**不要再次执行prepare或run，不要删除锁。** 已批准的最多五次训练顺序执行，无额外试跑、续训、减batch或第六个模型。

## 已完成且可核验的检查

- [实现/测试记录](oof_implementation_checks.json)：55项OOF合成/mock + 157项原bus回归，共212项通过；AST与git diff --check通过。
- [测试原始日志](oof-all-tests-final.log)：PowerShell Tee-Object保存，编码为UTF-16LE。mock回调打印并非真实训练轮次。
- 合成浏览10图、5折90项断言通过：阈值/类别/折/图片/画布/表一致，空预测保留、开关、坏图清空恢复、原记录不变；668px无横向溢出。尚不代表真实OOF结果已验收或人工判框。
- [准备冻结](preparation_verified.json)：62850项准备产物。
- [折分统计](split_verification.json)：五个outer恰好覆盖7852图；fit/monitor/outer组、路径及完全同图哈希互斥；fit与monitor均四类覆盖。
- [固定配置与来源](plan_config.json)：源码/依赖/保护文件哈希、初始17668原文件完整性和实际环境。

|折|fit|monitor|outer|
|---|---:|---:|---:|
|01|5629|628|1595|
|02|5637|638|1577|
|03|5664|641|1547|
|04|5677|612|1563|
|05|5644|638|1570|

## 准备阶段的宿主等待中断

原prepare的600秒宿主等待到期并停止命令时，model-free环境探测已exit0、五折副本和清单已完成，缺少最后的核验冻结文件。没有存活OOF进程，没有任何模型启动。没有重跑prepare、探测或复制，没有改源码、种子或预算。

仅用后台只读收尾脚本核验62816个派生文件的SHA256/尺寸/非硬链接属性、17668原文件、4227项历史保护文件及源码/依赖，并补写缺失冻结证据，仍遵守原六小时截止时间。详见[中断与收尾证据](preparation_host_wait_reconciliation.json)、`preparation_finalization_started.json`及归档脚本和日志。原超时事实与所有现场保留，不将其写为模型早停或训练失败。

## 固定训练边界

YOLOv8n本地架构、每折独立随机初始化，SS03：epochs最多24、patience8、batch16、640、workers2、device0、trainer seed0/deterministic；split seed42。只有monitor决定best/早停，outer在冻结best后预测；val966不进入模型，test不读取。AMP保留，但外部权重一致性自检未执行，不能视为该自检通过。OOM/NaN自动恢复关闭；Python层I/O/network约束不等于原生OS沙箱。

外折全图四类FP32预测：conf.001、IoU.7、max_det300、batch1、无裁剪/TTA/集成，包含零预测空记录。置信度筛选只用于显示，不是标注正确性或补框阈值。

## 五折完成后仍需验收

最终应有 `oof_predictions.jsonl`、`OOF_INDEX.json`、五折best/last、`fold_summary.json`、`oof_metrics.json`、`RESULT.md`、`review.html`、`verification.json`及 `final_checks.json`。不能仅凭这些预期文件名宣称当前已经存在或完成。

只读复验命令（仅在整批成功后运行，不启动模型）：

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B .\tools\oof_diagnostic.py verify
```

还需检查真实浏览页、最终原数据完整性并如实追加PROJECT_LOG。总体AP从全部保存记录统一计算并明示匹配规则，不冒充Ultralytics原生精确匹配、加权折AP或标签质量真值。停止点是五个独立模型、7852唯一OOF记录和证据，不自动进入cleanlab、改标、清洗对照、集成或竞赛提交。
