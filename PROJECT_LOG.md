# PROJECT_LOG.md — 项目决策与实验记录

> 本文件按 [AGENTS.md](AGENTS.md) §4「决策记录规范」维护。
> 每个学习块/实验一节，字段固定：**日期 / 阶段 / 决策·操作 / 原因 / 结果 / 收获 / 踩坑 / 问题解决来源 / 后续计划**。
> **问题解决来源**必须如实标注：`独立解决` / `借助 AI 解决` / `AI 直接解决`。
> 禁止事后编造或美化；如记录与实际过程不符，当事人可随时更正。

---

## 2026-10-15　阶段：环境搭建 + Baseline 训练（第一次全链路跑通）

### 决策 / 操作

1. **环境策略**：不在已有的 `dl-reboot` 环境里改动，**新建独立 conda 环境 `dl-3lc`**（Python 3.12.15），避免污染其他项目。
2. **工具链选择**：确定**必须使用 3LC**（竞赛规则 "3LC is required for the prescribed workflow"），放弃"绕过 3LC 直接用 Ultralytics"的方案。
3. **Baseline 配置**：`config.yaml` **一个参数都不改**，按官方默认跑：`yolov8n.yaml` 从零训练、`epochs=10`、`batch=16`、`imgsz=640`、`lr0=0.01`、`seed=42`、增强开启（mosaic 1.0 / mixup 0.05 / copy_paste 0.1）。
4. **修改了 kit 的两个批处理脚本**（仅适配性改动，训练逻辑未动，详见"踩坑"）：
   - `register_tables.py`：3 处（新增 `_read_categories()` 从 `dataset.yaml` 读类别映射；两处注册调用改用 `from_yolo_url`）
   - `train.py` / `predict.py`：各注释掉 1 行对已有失效补丁的调用

### 原因

- 先跑通最小闭环、建立**对照基准**，不追求分数。没有 baseline，后续任何"改了 X 结果变好"都缺少依据。
- 架构被规则锁死（YOLOv8n、禁用预训练权重、imgsz=640、禁集成/TTA/伪标签/蒸馏），可调空间很小，因此更需要一个干净的基准点。

### 结果

**成功。全链路 4 个环节全部跑通。**

| 环节 | 命令 | 结果 |
| :--- | :--- | :--- |
| 环境体检 | `verify_setup.py` | **ALL CHECKS PASSED**（8 项包 + CUDA + 5 个数据目录） |
| 数据注册 | `register_tables.py` | train **7852** / val **982** 张注册为 3LC Tables |
| 训练 | `train.py` | 10/10 epoch 完成，用时 **609 秒（≈10 分钟）** |
| 推理 | `predict.py` | 生成 `submission.csv`（982 行，**17 秒**，10209 个框） |

**Baseline 指标（`best.pt` 在验证集上）**

| 指标 | 值 |
| :--- | :--- |
| **mAP50** | **0.785** |
| **mAP50-95** | **0.606** |
| Precision | 0.890 |
| Recall | 0.692 |
| 显存峰值 | **2.49 GB / 8 GB** |

**分类别表现（暴露了清晰的改进方向）**

| 类别 | 验证实例数 | P | R | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| all | 11615 | 0.890 | 0.692 | 0.785 | 0.606 |
| truck | 207 | 0.945 | **0.594** | 0.703 | 0.564 |
| car | 9470 | 0.897 | 0.810 | 0.879 | 0.651 |
| van | 954 | 0.797 | 0.614 | **0.727** | 0.556 |
| bus | 984 | 0.923 | 0.750 | 0.830 | 0.654 |

**训练产物**：`runs/detect/yolov8n_baseline/`（`weights/best.pt` 6.2 MB、`results.csv`、`results.png`、`confusion_matrix*.png`、`Box*_curve.png`、`args.yaml`）
**3LC Run**：项目 `ua_detrac_vehicle_detection` / run `yolov8n_baseline`，含 7852 张图的 3D 图像嵌入（umap）

### 收获

1. **`optimizer=auto` 会让 `lr0` 失效**——这是本次最重要的发现。日志明确写着
   `optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937'`，
   实际用的是 `AdamW(lr=0.00125)` + 余弦退火。**做学习率实验前必须先显式设置 `optimizer`**，否则改 `lr0` 等于没改。
2. **8 GB 显存完全够用**：实测只用 2.49 GB，因此 `batch` 有很**大**的向上调整空间（不是向下）。
3. **理解的 V2 价值**：训练时 3LC 自动回写逐样本 IoU，配合训练集 3D 图像嵌入，可以在 Dashboard 上按"指标最差"排序找坏样本——规则的"data-centric 循环"指的就是这个。
4. **类别不均衡是主要矛盾**：car 有 9470 个实例，truck 只有 207 个；truck 召回率 0.594 全场最低。

### 踩坑

**坑 1：清华镜像没有 `3lc` 这个包**
- 现象：`pip install 3lc-ultralytics` 报 `ResolutionImpossible`，所有版本都缺依赖 `3lc`
- 根因：`3lc` 核心包**在 PyPI 上存在（3.4.1），但清华镜像未同步**
- 解决：加 `--extra-index-url https://pypi.org/simple` 从官方源补装

**坑 2：`ultralytics` 版本被 `3lc-ultralytics` 钉死**
- 现象：`3lc-ultralytics 0.5.1` 要求 `ultralytics >=8.4.104,<8.4.166`，而 `dl-reboot` 里是 **8.4.171**（超出上界 5 个版本）
- 处理：因为不想改动 `dl-reboot`，选择**新建独立环境**，让 pip 自然解析到 **8.4.165**（正好是上界），无需降级任何已有环境

**坑 3：3LC 3.0 移除了 `Table.from_yolo`（官方 starter kit 与公开版 3LC 不匹配）**
- 现象：`register_tables.py` 报 `AttributeError: type object 'Table' has no attribute 'from_yolo'`
- 根因：3LC 3.0 是 breaking change，`from_yolo` 被 `from_yolo_url` 取代（入口从 `dataset.yaml` 变成"图片目录 + 显式类别映射"）
- 关键判断：**降级到 3LC 2.x 不可行**——PyPI 上没有 3lc 2.23.x，且支持 2.x 的 `tlc-ultralytics` 版本要求 `ultralytics <8.4.7`，会连降两个大包
- 解决：改 `register_tables.py` 用 3.x API，类别映射从 `dataset.yaml` 的 `names` 字段动态读取（保持配置驱动）

**坑 4：`train.py` 自带的兼容补丁在新版 `tlc_ultralytics` 下直接崩溃**
- 现象：`AttributeError: TLCDetectionValidator has no attribute '_process_detection_predictions'`
- 排查：我下了 4 个公开版本的 wheel 逐个解包搜索，**0.1.x–0.5.1 全部都没有这个方法** → 推断 kit 是针对**尚未公开发布的 tlc_ultralytics** 开发的
- 根因：新版 `tlc_ultralytics 0.5.1` 的 `_process_predictions` **已原生处理 dict 形式的预测**（kit 那段 90 行补丁修的就是这个旧 bug），补丁本身已过时
- 解决：注释掉 `train.py:189` 和 `predict.py:343` 对补丁的调用，**函数体保留原样**便于评审对照

**坑 5（工具噪声，值得记以免再被误导）**
- 现象：多个脚本明明成功，PowerShell 却报 `[exit code: 1]`
- 根因：3LC 的 Sentry 日志写到 **stderr**，PowerShell 把 stderr 当作错误
- 应对：**以脚本自身输出（如 `OK — TRAINING COMPLETE`、`ALL CHECKS PASSED`）为准**，不要被 exit code 误导

### 问题解决来源

**`借助 AI 解决`**

AI 的具体贡献：搭建独立环境、定位 4 个坑的根因（含下载 wheel 逐个排查）、给出与竞赛规则的合规性分析、执行训练与推理并整理指标。
我的具体贡献：确认竞赛规则原文（提供完整规则文本）、决策采用 3LC 而非绕过、**决策修改 kit 脚本还是降级依赖**、指定"不改 config 跑原样 baseline"。

（若你对该标注有异议——例如某个坑你是自己定位的——直接改这一行，如实最重要。）

### 后续计划

1. 跑 `predict.py` 生成 `submission.csv`，核对 982 行 / id 连续 / 格式对齐官方样例。
2. 在 3LC Dashboard 上按指标排序找坏样本，先看 **truck（召回 0.594）和 van（mAP50 0.727）**。
3. 设计**单变量实验**，建议顺序：
   - 第一批：**先修"不生效的参数"**——设 `optimizer: SGD`（或 `AdamW`）后，`lr0` 才能真正被调；这是后面所有学习率实验的前提。
   - 第二批：`batch_size`（显存只用 2.49 GB，可往 32 调）、`epochs`（10 轮曲线还在上升，可试 30–50）、`use_augmentation` 开关。
   - 第三批（预期收益最大）：针对 truck/van 的**数据侧改进**（清洗标签、检查漏标、样本权重）。
4. 每次实验只改一个变量，结果追加到本文件。

---

## 2026-10-20 至 2026-10-24　泄漏核查、人工反馈与固定 val966 baseline

> 于2026-10-25按学习者授权整理归档；本节只追加，不改写上方已提交历史。完整时间线见 [独立日志](logs/score-search-20261019-150516.md)，本阶段当前状态见 [归档索引](reports/ARCHIVE_INDEX_20261025.md)。

- **阶段**：数据质量审计 → 评估口径确认 → 未清洗训练集 baseline。
- **决策/操作**：先核查跨train/val泄漏，确认16对完全同图；将4862对近重复仅视为候选。学习者人工发现漏标后，将关注点转到训练数据质量。旧“移val入train”方案作废，原文件不动，仅在取用val时跳过16张同图。已交回的随机人工审核JSON归档，不要求重新审核。
- **原因**：区分确定的输入重叠、同场景相似和标签正确性；不凭自动筛查或局部人工观察决定全量重标。学习者选择先取得可复现baseline及错误样本证据，再决定训练集处理。
- **评估决定**：C档目标为官方隐藏测试集AP50≥0.90。val是否干净尚未证实，学习者决定暂以足够可靠为工作假设、固定为本地参照；隐藏test质量未知。train修标允许，val修标规则冲突保留；虽有主办方允许重划分的答复，本项目未据此重新划分。
- **人工反馈结果**：第一轮口头反馈确认两张空标签漏标、多个样本存在漏标，相似场景不等同完全同图；第二轮原JSON的前两页已审24张（train12/val12），其中train10、val7被学习者判为漏标，其余72张未审。未填缺框总数，不能外推全量或test。
- **baseline结果**：使用starter原始训练参数，YOLOv8n随机初始化，10轮、batch16、640；train7852图/72273框，val966图/11487框。最终best验证AP50=0.7847740054、AP50-95=0.6068032980、Precision=0.8921241164、Recall=0.6919714212，入口耗时839.359秒，退出0。配置与实际trainer参数差异、首次启动中断及权重哈希详见 [baseline报告](reports/baseline-starter-20261024/RESULT.md)。
- **收获**：val966是本阶段固定参照，不与旧val982直接横比；本地分数不等于官方达标。格式合法、候选相似及可见未标对象分别需要证据，不能混为全量脏数据结论。
- **踩坑/限制**：首次后台启动中断且退出码未知，随后同配置恢复完成，一个完成实验、两次启动尝试。旧报告与验证JSON是历史快照；新增归档不回写旧哈希。实际清洗、本轮错误样本目视分析、追加训练和官方提交尚未执行。
- **问题解决来源**：策略/目标选择为`独立解决`；学习者人工发现与AI审阅工具协作为`借助 AI 解决`；机械审计、配置核对、执行、统计和归档为`AI 直接解决`。
- **后续计划**：先查看本轮指标和具体错误样本，由学习者决定训练集处理；当前仅授权归档与commit，合并main另待明确授权，不自动加训或发布。

## 2026-10-25　归档与Git提交授权

- **阶段/决策**：学习者批准先整理并完成审计、baseline两次commit，然后只准备合并；AGENTS.md以main已提交版本为准。历史临时实验授权仅作为非生效附件，不替换长期规则。
- **原因**：保留原日志历史，避免实验隔离配置、原图、权重、认证状态和缓存进入main。
- **结果**：已有独立日志与人工反馈原件已保全，提交结果以实际Git提交哈希及交付为准，不预写成功；文件白名单及检查见reports/git-preparation-20261025/。
- **收获/踩坑**：生成HTML内嵌原图不适合直接入库；Git内文本记录不等于完整数据/运行环境。规则恢复核对了CRLF与末尾换行，原件按字节保留。
- **问题解决来源**：提交范围与main规则选择为`独立解决`；归档、白名单检查与Git操作为`AI 直接解决`。
- **后续计划**：commit后只读检查main祖先关系及目标worktree状态；不merge/push/stash/clean/reset，不删除worktree或本地证据。
