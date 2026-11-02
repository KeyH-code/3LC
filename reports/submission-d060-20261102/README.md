# D060阶段性Kaggle CSV（本地生成，未上传）

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

学习者自行上传Kaggle；本工具没有上传功能。目前无Public/Private成绩或位次，不宣称官方AP50>=.90达标。CSV格式验收不等于训练入口/3LC流程已获主办方批准，规则审阅由学习者负责，历史限制继续披露。

问题解决来源：`AI 直接解决`（本次点名的完整产物、机械推理、校验）；模型选择与手动提交由学习者负责。
