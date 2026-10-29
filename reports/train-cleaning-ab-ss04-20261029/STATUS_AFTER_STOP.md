# SS04 A/B阶段结果：两次训练完成，统一评估失败后停止

## 真实状态

- A、B各自完成28轮；A best为第28轮，B best为第27轮。
- A耗时3020.81秒（约50分21秒），B耗时3203.13秒（约53分23秒）；这不是先前历史36分钟耗时。
- 首次额外统一评估A/eval_A在加载权重时失败，六次统一评估完成0次。没有自动重试，没有重训、加训或改参数。
- 原生训练末尾best验证已由两次训练各自完成，以下可报告这部分阶段性结果，但不能冒充计划中的六次统一FP32评估。

## 训练末尾best在共同val966上的原生验证

| 指标 | A原始标签 | B清洗标签 | B−A（百分点） |
|---|---:|---:|---:|
| AP50 | 0.808583 | 0.827796 | +1.921 |
| AP50–95 | 0.659806 | 0.673297 | +1.349 |
| Precision（各自最佳F1点） | 0.913778 | 0.901606 | −1.217 |
| Recall（各自最佳F1点） | 0.734393 | 0.760559 | +2.617 |

这是同一配对、共同val966的观测：B的AP及召回高、最佳F1点精确率低。各自最佳F1点不是同一个固定置信度；不能据此代做清洗效果根因、显著性或泛化结论。未完成train统一评估，不能计算或判断train–val差距。

## 失败证据与待检查处

- `A/eval_A/console.log`末尾原文：`FileNotFoundError: ...\A\model\weights\best.pt`。
- 执行器evaluate硬编码了model目录，而训练实际结果记录为 `A\yolov8n\weights\best.pt`。后续若获授权，仅从对应 `train/result.json` 读取并校验真实权重路径/哈希，不重新训练。
- 失败评估还记录3条被guard拒绝的socket事件：一次bind、两次getaddrinfo。事件均被阻止，不等于已连网；仅凭当前日志不确定调用来源，需在补评估前追踪，不得通过放开网络或忽略事件使评估通过。训练阶段没有guard事件。
- `run_failure.json`、`A/eval_A/failure.json`、`A/eval_A/exit.json`及runtime下事件日志原样保留。原执行器和execution_seal不改写。

## 停止后的独立静态核验

见 `post-stop-verification.json`，不加载模型：

- 两组results.csv各28行，epochs.jsonl均为完整1–28。
- A/B初始化参数哈希相同：`b29d1a75cc221bee6f6f2b3722dbc7c65a266fad8cdea8295e938c975d78f8c0`。
- best/last四个权重哈希均与训练结果一致；两个best各6248362字节。
- 33520个保护源文件、33340个副本数据文件和15项execution_seal哈希均匹配。
- 无训练阶段guard拒绝事件；失败评估3条拒绝事件如实记录。

## 已保存权重

- A：`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\train-cleaning-ab-ss04-20261029\A\yolov8n\weights\best.pt`
  - SHA256：`2f7bc93b907fe6f2559cbb0788a70f4fe4e436188bf104297f7cabc208978d8d`
- B：`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\train-cleaning-ab-ss04-20261029\B\yolov8n\weights\best.pt`
  - SHA256：`ca6da799dda679b91fa60eb2d93690aada0ca370be09fab90d8f5dc3d224d701`

## 下一步边界

等待学习者批准仅补评估：修正权重读取及离线兼容问题、补测试；保留旧执行器/失败证据，在新目录执行六次评估，不重训、不变更模型或训练超参数。尚未执行任何恢复动作。

问题解决来源：**AI 直接解决**（训练执行/核验/记录）；评估路径错误属于AI编写的执行器缺陷，测试和审查未检出，不归因于学习者或标签清洗。
