# Starter 原始训练配置 Baseline — 2026-10-24

## 结果与范围

**已完成一个 baseline 实验，10轮训练及官方入口内置的训练集收集、best权重验证、UMAP处理均完成，退出码0。未清洗、补标或移动任何原数据；未使用test、未生成或提交submission。**

- 学习者决定：先以val足够可靠为工作假设建立未清洗train的baseline，观察指标/错误样本后再决定训练集处理；本次明确选择starter原始配置开始训练。
- “原始配置”来源：当前HEAD `5c0930033c3710e21c904d85fbf215aed9ba33d2`中的starter `config.yaml`，快照为`config.starter.yaml`。训练参数、reproducibility与嵌入设置已机械比对匹配；3LC项目/表名、运行名称/描述保留隔离设置，避免覆盖旧实验。
- 验证口径按此前学习者决定为 **val966**：仅取用时跳过16张与train完全同图的val样本；原val982及磁盘文件不改。不是逐字恢复全套starter数据/名称配置。
- 入口保持当前提交内已有的本地兼容适配，本轮没有修改train.py/register_tables.py/predict.py/dataset.yaml；不声称是未经本地适配的上游字节原件。
- val干净是本阶段工作假设，不是新获得的质量验证结论。

## 数据与参数

| 项目 | 本轮设置/事实 |
|---|---|
| train | 7852图、72273框，原标签未清洗 |
| val | 966图、11487框，取用时排除16张同图；保留图的原框坐标/类别与源表相同 |
| 模型 | YOLOv8n，yolov8n.yaml随机初始化，4类，3011628参数（未融合） |
| 训练 | epochs=10、batch=16、imgsz=640、device=0、workers=2、fraction=1.0、resume=false |
| 配置增强 | use_augmentation=true，入口传入mosaic=1、mixup=0.05、copy_paste=0.1 |
| 嵌入 | image_embeddings_dim=3，umap；train7852/val966逐图均有有限3维坐标 |
| 设备 | NVIDIA GeForce RTX4060 Laptop GPU，8188MiB（运行日志显示） |
| 版本 | Python3.12.15，torch2.14.1+cu130，3lc3.4.1，3lc-ultralytics0.5.1，ultralytics8.4.165，umap-learn0.5.12 |

### 原配置与实际生效行为必须区分

1. `optimizer=auto`实际选择 **AdamW(lr=0.00125, momentum=0.9)**，忽略配置中的lr0=0.01；`args.yaml`仍保留请求值0.01。
2. 配置seed=42在入口开始时应用，但trainer实际seed=0；配置patience=20没有传入trainer，实际patience=100。本轮按原入口执行，不修补接线。
3. 默认`close_mosaic=10`且总轮数10，日志在第1轮前显示`Closing dataloader mosaic`。不能仅凭配置增强开关就声称本轮全程使用mosaic/mixup，也不表示关闭所有增强。
4. `args.yaml`的`pretrained: true`是保留的默认字段；入口明确构造`YOLO(yolov8n.yaml)`，并非加载.pt预训练模型，日志无Transferred权重记录。AMP的YOLO26n检查因离线跳过，未下载到yolo26n.pt。
5. `args.yaml`的`data: coco8.yaml`是通用默认字段；实际训练输入由3LC提供，日志、`tables_used.txt`与Run inputs一致确认train7852/val966，不是COCO数据。

## 最终 best.pt 验证指标

采用官方入口训练结束后的best权重验证，完整精度来源为本地3LC Parquet；表中按6位小数显示。

| 类别 | 标注框数 | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|
| **全部（val966）** | **11487** | **0.892124** | **0.691971** | **0.784774** | **0.606803** |
| truck | 205 | 0.953156 | 0.600000 | 0.708373 | 0.569132 |
| car | 9363 | 0.896789 | 0.807754 | 0.877956 | 0.649989 |
| van | 945 | 0.796012 | 0.612698 | 0.725162 | 0.555567 |
| bus | 974 | 0.922539 | 0.747433 | 0.827605 | 0.652525 |

- 训练集收集指标：Precision=0.8938089013、Recall=0.8394543529、AP50=0.9113369584、AP50-95=0.7274465561。训练集指标不作为泛化成绩或目标达标依据。
- `results.csv`中按入口默认AP50-95选出的最佳轮次为第10轮，训练期验证AP50=0.78482、AP50-95=0.60687。与训练结束后的best验证0.784774/0.606803略有差异，两份原始观测均保留，不混用。
- AP50从第1轮0.27441上升至第10轮0.78482，第9轮相对第8轮有回落；不能据此单独裁定已收敛、过拟合或数据根因。
- 学习者目标仍是 **官方隐藏测试集 AP50>=0.90**；本轮无测试集评分，不能把本地val分数当作官方成绩，也不将两者的数值差直接定义为官方目标差距。
- val966不与此前val982的旧实验做直接胜负比较。

## 时间、尝试与中断

### 成功执行
- 开始：2026-10-24T20:52:39.200020+08:00。
- 结束：2026-10-24T21:06:38.564000+08:00。
- PID：40668；退出码0；未超时；观察到的子进程均已退出，未额外清理残留。
- 入口墙钟 **839.359秒（13分59.359秒）**；10轮循环累计621.863秒（包含逐轮验证），其余为初始化和入口自带收集/验证/嵌入等开销。
- 45分钟安全看门狗；不续训、不调参、未追加第二个baseline实验。

### 首次启动中断（不可隐去）
- 最初后台启动时间20:49:08，监督PID33244、训练PID37388，约10秒后状态不再更新；20:50:45确认两者均不存在，console/stdout/stderr为空，没有模型运行目录、CSV或权重。
- **退出码未捕获、具体中断原因未证实**，不记成0，也不宣称已确定宿主或模型根因。
- 原始证据保存在`launch-interrupted/`，随后以前台持续等待方式恢复同一配置。合计两次启动尝试、一个完成的baseline实验。
- 准备阶段`-W error`曾把UMAP可选TensorFlow组件的ImportWarning提升为异常；按入口正常警告处理方式重跑prepare成功，无安装依赖。这不是训练入口执行。

## 产物与证据

工作区绝对根目录：
`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516`

- 本报告：`<工作区>\reports\baseline-starter-20261024\RESULT.md`。
- 配置：同目录`config.before.yaml`（SS04快照）、`config.starter.yaml`（HEAD原配置）、`config.used.yaml`（本轮实际配置）。
- 监督证据：`preparation.json`、`training_attempt.json`、`supervision.json`、`console.log`、`resources.jsonl`、`tables_used.txt`、`launch-interrupted/`。
- 机械核验：同目录`verification.json`，全部20项检查通过；无额外模型运行。
- 原始训练产物：`<工作区>\multi-vehicle_kaggle_competition_starter_kit_files\runs\detect\baseline_starter_val966_20261024\`，含`results.csv`、`args.yaml`、`results.png`、曲线/验证批次图和权重。
- 本地3LC Run：`<工作区>\3lc\projects\score_search_20261019_150516\runs\baseline_starter_val966_20261024\`。4条指标流已实际读取Parquet：train逐类5行、train逐图7852行、val逐类5行、val逐图966行；逐图记录保存预测框和3维嵌入，可用于后续错误样本分析，不等于已经完成FP/FN根因分类。
- val966源清单SHA256：`412109aeabf71f163a7eb7b3396b48a64be512c453ce83e8774bf9d95d17a048`。新表名`vehicle_fixed_split-val966-baseline-20261024`；表版本文件哈希见verification.json。

### 最佳权重

绝对路径：
`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516\multi-vehicle_kaggle_competition_starter_kit_files\runs\detect\baseline_starter_val966_20261024\weights\best.pt`

- 大小：**6245802字节**。
- SHA256：`3a90ef9c30824cefbe29a379814dac9cab64eeafc06c73243f24d32bf0dbe4b1`。
- last.pt同目录，6245802字节，SHA256：`0e7a20c6b2332e29af7905852ec671c70a4f3f76e5dbc16845a33deed84436e2`。

## 验证与复现命令

以下仅核验已有文件，不会启动训练/推理：

```powershell
$w = 'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516'
& 'E:\anaconda\envs\dl-3lc\python.exe' -B "$w\tools\verify_baseline_starter_20261024.py"
Get-FileHash -LiteralPath "$w\multi-vehicle_kaggle_competition_starter_kit_files\runs\detect\baseline_starter_val966_20261024\weights\best.pt" -Algorithm SHA256
```

实际被监督执行的入口命令（历史记录，**不是再次运行授权**）：

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -B -u "$w\multi-vehicle_kaggle_competition_starter_kit_files\train.py"
```

该入口需要本轮准备中记录的隔离环境；不能只复制裸命令就称完全复现隔离。机械工具`baseline_starter_20261024.py`保留环境配置、单实验防重复记录和看门狗；现有训练尝试文件会阻止再次启动，未修改保护。

## 边界、访问与下一步

- 训练前后17668个原图片/标签文件尺寸/SHA256均匹配manifest；原val982表文件哈希未变。无入口或dataset.yaml改动，Git空白检查通过。
- 未启动Dashboard/Object Service、未生成提交、未上传数据或发布、未安装依赖、未修改Git暂存/历史。正常3LC认证使用此前指定凭据，值不打印或写入新文件；本轮证据/工具检查未发现完整明文凭据。
- 目录外资源仅此前指定环境`E:\anaconda\envs\dl-3lc`和凭据原件`C:\Users\胡成凯\AppData\Local\3LC\3LC\3lc_api_key`。不读取本地test或主工作区原数据。
- 本次训练与核验无剩余执行阻塞。**错误样本目视分析和训练集处理尚未执行**；由学习者下一步决定检查对象与处理策略，不自动补标、加训或修补入口。
- 问题解决来源：实验流程与starter配置选择为学习者`独立解决`；本次机械准备、执行、监督和核验为`AI 直接解决`。
