# D060阶段性Kaggle CSV（学习者赛后手动提交已评分）

## 交付

- 绝对路径：`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\submission-d060-20261102\submission.csv`。
- 文件558948字节，SHA256 `2751c6dc8603293605b2f1ffe1d6b844c1ba3c9fb547da3c9362da85eba87db5`。
- 与官方sample完全一致的列名`id,image_id,prediction_string`，982行，原id/image_id逐行匹配；本轮所有图均有预测，无`no box`行。
- 10557框：truck157、car8831、van708、bus861。计数是预测框数，不是真值或质量评价。

## 固定模型/参数

学习者指定D060第28轮best：`reports/train-cleaning-expansion-resume-r1-20261102/D060/yolov8n/weights/best.pt`，SHA256 `b36f72c507a13cc134092f3589d181c25b2014ae8f6845113b662ce654ed1eac`。没有用其他档模型、集成或加训。

沿用starter配置conf=.25、iou=.7、imgsz640、batch1、max_det300、GPU0，FP32，无TTA、无阈值搜索；6元组为`class confidence xc yc w h`，类别整数0–3，其他数值有限且位于[0,1]，无多行单元格。真实无检测输出`no box`，缺图/覆盖错误拒绝而非伪造空预测。

一次受离线guard和Windows进程树监督的worker完成，退出0、未超时，worker61.456秒/监督63.118秒，无3LC导入/guard事件。输入模型、982图、sample及配置前后哈希核对通过；原始文件不修改。

## 验证（只读，不重推理）

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B tools/submission_d060.py verify
Get-FileHash 'reports/submission-d060-20261102/submission.csv' -Algorithm SHA256
```

CSV、test图、权重、runtime和含逐图身份的inputs.json仅本地保存；Git仅收录工具、测试及本说明/result/verification/exit等轻量证据。新clone需具备原输入才可执行完整复验，不因seal引用路径就认为远端已有这些文件。

学习者自行上传Kaggle；本工具没有上传功能。下方为后续真实反馈，不能将工具生成时“未上传”的历史状态当作当前状态。CSV格式验收及平台成功评分不等于训练入口/3LC流程已获主办方批准，规则审阅由学习者负责，历史限制继续披露。

## 2026-11-02 学习者提交后的精确截图反馈

来源：学习者在对话中提供三张截图，明确第一张为Private榜单区间、第二张为Public区间、第三张为本人带列标题的提交结果。此处人工读取截图文字，未联网重新拉取榜单，未把截图原图另存或声称持有平台提交ID。

| 项目 | 官方显示分数 | 相邻历史榜单分数 | 单个分数假设插入位置 |
|---|---:|---|---|
| Private | **0.77173** | 原第64名0.77291 > 本次分数 > 原第65名0.77132 | 约第**65**位 |
| Public | **0.77408** | 原第63名0.77586 > 本次分数 > 原第64名0.77060 | 约第**64**位 |

本人提交截图显示`submission.csv`、`Complete (after deadline)`、`first submission`；评分已完成，但没有本人的正式名次截图。上述65/64是按给定区间假设插入的历史分数位置，不是正式参赛名次或获奖资格认定。此前口头“两榜约65”更新为Private约65、Public约64；前次裁剪截图无列标题的对应不确定性现已解除。

- Public比Private高0.00235，即**0.235个百分点**；数值接近不证明跨场景稳定性或无过拟合。
- 原目标官方隐藏测试集AP50≥0.90：Private尚差0.12827（**12.827个百分点**），本次未达标；Public距0.90为12.592个百分点。
- D060统一本地val AP50为0.8455019720，相对本次Private/Public分别高约**7.377/7.142个百分点**。这里比较的是不同数据子集和评估/推理工作点，不能单凭差值给出分布差异、标签问题、后处理或过拟合的根因结论。
- 当前仅有D060一次官方反馈；缺少A/B官方对照，既不能宣称清洗在隐藏测试获得相同收益，也不能认定清洗无效。Private反馈已被查看，今后若据此选择模型或调整方案，需记录该反馈使用，不再称其为未触碰的盲测。
- 本次仅记录反馈，不触发重训、test重推、其他模型提交或阈值搜索。原CSV、权重、result/verification以及已封存训练报告不变。

问题解决来源：`AI 直接解决`（本次点名的完整产物、机械推理、校验）；模型选择与手动提交由学习者负责。
