# OOF 已恢复后台执行（非五折完成报告）

状态核验时间：2026-10-26 17:07:17 +08:00。之后的实时状态以 `progress.json`、每折 `stopper_trace.jsonl` 和后台任务为准。

- 本次已撤销“失败即停、不自动重试/恢复前再请示”的约定，生效范围见 `../oof-recovery-authorization-20261026.md`。旧计划和失败现场保留作为历史证据，不再限制本次可恢复实现故障的修复续跑。
- 第1折：原训练24轮、best第21轮，未重训。best SHA256：`a5759ac4ccee56cc98fac268a5d2db90f5cd31399d023e9bb12222aaac586c8d`。
- 第1折 outer：**1595/1595** 唯一记录核验通过，worker exit0/未超时/guard_events为空，无原始guard日志。实际后端FP32、权重torch.float32；保留原有输入张量/像素与参数检查。预测流SHA256：`343afae83b35a6f704022ee35cf9fc0e9433ac9b081ae723117d8779dc7309a0`。
- 第2折：已实际进入训练并完成第1轮；生效训练参数与冻结配置一致。监督器仍为running，当前有效OOF覆盖1595/7852。
- 第3–5折：由同一任务随后顺序执行，未预写成功。全体7852覆盖、最终只读验收和真实浏览器验收仍待完成。

## 当前后台任务

- 训练/预测顺序监督：`e80d3567-63a1-4366-be29-238b509d3e92`。
- 完成后只读验收等待器：`bea6469f-885a-49ab-91f3-fdbd4123caac`；仅在完整五折封存后调用verify，不调用模型，明确保留browser_pending。
- 原批次绝对截止时间保持不变（deadline_unix=1793017704.8626258），没有因恢复重置六小时预算。

## 修复与证据

1. 移除args.half后，预测进程本地适配为从实际backend.fp16读取有效half，并要求quantize=None和FP32权重；训练/预测口径不变，不升级依赖。
2. 对两个明确版本的适配器脚本允许源码只读内省，仅增加精确文件许可，不开放目录或忽略guard。
3. 首次API失败、第一版完整预测后自身源码guard失败、恢复前日志列表检查失败均保留。原始journal实际8条自身源码被拒只读：6条在worker结果内，2条发生于结果写入后的异常栈输出。严格核对前缀/路径/只读/时间后归档，没有改写失败状态，也未将该尝试计入完成覆盖。
4. 首折成功预测与前一失败尝试完整预测流哈希相同。独立续跑核验为 `continuation_launch_verified.json`；完整失败阶段和原始日志在fold_01下的独立历史目录及runtime_reconciliation中。
5. 本轮检查：57项OOF测试、10项原生精度映射/适配检查、8项严格日志反例检查、版本身份检查及git diff --check通过。续跑核验未导入torch/Ultralytics或重新调用模型。
6. 原数据、标签、官方入口、AGENTS/RULES、冻结执行器/配置及已安装依赖未修改。训练保持SS03、24轮上限、patience8、batch16、640、workers2、device0、独立monitor。AMP外部权重自检仍明确未执行。

本次外部接口只读核对目标（批准Python环境）：`E:\anaconda\envs\dl-3lc\Lib\site-packages\ultralytics\cfg\__init__.py`、`engine\predictor.py`、`nn\autobackend.py`；均未修改。源/依赖冻结哈希复核通过。

## 可复现检查

在项目根目录执行：

```powershell
# 实时状态（只读）
Get-Content -LiteralPath .\reports\oof-diagnostic-5fold-24e-retry1-20261026\progress.json -Raw

# 复核首折有效记录、失败归档与第二折实际训练配置；不调用模型
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B .\reports\oof-diagnostic-5fold-24e-retry1-20261026\verify_continuation_launch.py

# 只有五折封存后才运行完整验收
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B .\tools\oof_diagnostic.py verify
```

这仅为离线诊断续跑状态，不是标签质量结论或竞赛/提交许可；未自动改标、伪标签、集成、追加全量模型或提交。
