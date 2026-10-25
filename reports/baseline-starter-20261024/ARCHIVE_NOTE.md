# Baseline 入库说明（2026-10-25）

这是10月24日已完成训练的证据归档，**本次没有重新训练、模型验证或推理**。

- `results.csv`、`args.yaml`是原run文件的逐字节副本，来源、大小和SHA256见`artifact_snapshots.json`。
- `MODEL_ARTIFACTS.json`记录本地best/last权重位置与哈希；权重本体、曲线、混淆矩阵、验证批次图、3LC逐图预测及嵌入仅保留在原worktree，不进入Git。
- `RESULT.md`、`preparation.json`、`verification.json`及监督状态都是各自原始时点的记录，包含本机路径，未为入库改写旧哈希或伪造重跑结果。
- `launch-interrupted/`保留首次启动的原小型状态记录和未知退出码说明；完整console/资源日志仅本地保存。后续成功执行退出0不覆盖首次未知退出码。
- `RESULT.md`中的`verify_baseline_starter_20261024.py`完整机械验证工具未入库（它与本地隔离环境/凭据路径耦合）；新clone没有这项工具，不应直接声称可执行其命令。归档提交只核对已有CSV/JSON/文件哈希和脚本语法，详见`../git-preparation-20261025/CHECKS.json`。
- 配置快照仅是历史证据；在线starter config指向隔离项目的改动未提交。合并到main不会自动配置val966的3LC表、搬运模型或改变其默认训练行为，后续开发接线另议。

复核CSV/args副本的SHA256，可使用PowerShell `Get-FileHash -LiteralPath <文件路径> -Algorithm SHA256`，与artifact_snapshots.json比较；这不运行模型。

原件空白例外：`config.before.yaml`、`config.starter.yaml`、`config.used.yaml`第2行均保留starter源注释末尾的一个空格。Git空白检查会提示这三处；本次按精确哈希验证原快照，不格式化或回写证据。其余入库文本要求空白检查通过，具体例外路径与原件哈希记入CHECKS.json。
