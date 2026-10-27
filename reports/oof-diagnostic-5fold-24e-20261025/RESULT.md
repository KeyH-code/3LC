# 5折分组OOF：本次启动失败，未完成

**最终批次状态：`failed_preserved_no_retry`。没有五折模型或7852张OOF结果，不能作完整OOF交付。**

## 实际执行结果

- 已实现执行器、固定配置、只读浏览模板及合成/mock测试，并按批准范围执行prepare与唯一一次后台run。
- 212项测试通过（55项OOF + 157项bus回归，31.828秒），AST/git diff --check通过；90项合成浏览断言通过。这些不等于真实训练已通过。
- 环境无模型probe成功：Python3.12.15、torch2.14.1+cu130、Ultralytics8.4.165、CUDA13.0、RTX4060 Laptop GPU；ONLINE=false、sync=false、guard_events=[]。
- 五折独立副本及分组清单已核验冻结：62816个派生文件，fit/monitor/outer组、路径和完全同图哈希互斥，outer并集7852图。
- **训练worker启动1次；模型架构加载阶段失败，未进入训练循环；完成0折、0轮、0个模型、0张OOF记录。** 没有启动fold02–05或任何outer预测，没有best/last、CSV或stopper轨迹。零覆盖不表示图片预测为空或没有目标，而是尚未执行预测。
- 失败worker进程22.2896205秒，worker内20.7993498秒；exit1、timed_out=false、guard_events=[]。这些是启动失败耗时，不是训练耗时或安全早停。

## 原始报错与发生位置

```text
FileNotFoundError: 'E:\anaconda\envs\dl-3c\Lib\site-packages\ultralytics\cfg\models\v8\yolov8n.yaml' does not exist
```

批准环境为 `E:\anaconda\envs\dl-3lc\`。执行器传入该环境内不存在的虚拟缩放文件 `yolov8n.yaml`，预期由本地通用 `yolov8.yaml` 解析出n档。实际调用栈在执行器 `train_worker` 的 `YOLO(c['architecture'])` 处终止，尚未到 `model.train()`。

只读核对已安装库 `nn/tasks.py:2272–2273`：fallback使用的正则 `re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))` 作用于整个路径字符串；本次字符串中较早的 `3l` 位于环境名 `dl-3lc`，结果变成报错所见 `dl-3c`，而非只处理模型文件名。这说明当前执行器的本地架构路径接线未通过实际启动；不是一次已完成的训练或模型表现评估。没有读取该错误环境目录内文件或另找环境，没有修改site-packages或冻结源码。

原始证据：
- [worker结果和完整异常](fold_01/train_stage/worker_result.json)
- [实际worker控制台](fold_01/train_stage/console.log)
- [启动/退出状态](fold_01/train_stage/status.json)
- [整批停止状态](run_status.json)
- [失败后原数据与保护文件核验](failure_integrity.json)
- `failed-startup-sources/` 保存本次四个实现/测试文件及批准计划的字节副本，原文件不改。

## 失败策略及保留内容

整批按批准条件停止，没有自动修复路径、续训、重试、减batch、换模型/设备或追加第六次训练。fold02–05保留准备好的独立副本，但没有训练入口。原图/标签、官方入口、默认配置、AGENTS/RULES、依赖、历史报告及本次失败锁均保留；没有改标、伪标签、3LC Tables/Runs、联网或提交。

源码中的路径适配目前**仍未修复**，故不要再次执行run/prepare或删除启动标记。该运行集不能复用为隐式重试；后续修复和新的模型执行需要另一次明确、限定范围的委托。本次不擅自启动。

## 准备阶段的另一项独立事件

原prepare的600秒宿主等待到期发生在五折副本/清单已完成、最终核验冻结未写完时。此时没有模型启动。保留现场后，仅用独立后台脚本补完只读核验、冻结缺失证据；没有重新prepare、probe或复制，继续遵守原六小时截止时间。[收尾证据](preparation_host_wait_reconciliation.json) 与原日志已归档。此事与后来的YAML模型加载失败分别记录，不混作训练早停。

## 可复验入口

- [固定配置及源码/依赖/保护哈希](plan_config.json)
- [准备产物冻结](preparation_verified.json)
- [实现验证证据](oof_implementation_checks.json)
- `failure_verification.json` / `final_checks.json` 在独立只读失败核验通过后创建，明确标记OOF未完成，不能冒充成功verification。

失败现场只读复验（不调用模型）：

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B .\reports\oof-diagnostic-5fold-24e-20261025\failure_verify.py --check-only
```

原执行器的 `verify` 是完整五折验收入口，本次状态失败，它应拒绝，不声称已通过完整OOF验收。尚无真实OOF浏览页、监控/外折指标或人工标签质量结论。
