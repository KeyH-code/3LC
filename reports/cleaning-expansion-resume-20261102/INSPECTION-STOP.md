# 恢复checkpoint检查失败：尚未续训

- 学习者已明确批准B方案、本次完整恢复执行器和5400/3600/21600秒新预算，授权文件在`reports/cleaning-expansion-080-070-060-20261029/TRAINING-RESUME-AUTHORIZATION-20261102.json`。
- 本次检查命令：`& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B -u tools/resume_cleaning_expansion.py inspect`。
- shell ID：`sh_19aa83bab0015mi7N68T7Fdd5g`。检查worker退出码1、未超时，耗时12.7180474秒。
- 报错原文：`RuntimeError: Checkpoint option drift: device`，见`inspection/failure.json`、`inspection/console.log`与`inspection/exit.json`。
- 已将原D080 last复制到本目录`checkpoint/last.pt`，worker核验SHA256与已批准的`6fe96d336204c1ce3189efdc49f73b8b78e9058242f0294ec6e7a4d43b445623`一致并在CPU加载；元数据检查的epoch==3、optimizer非空、EMA/scaler/update存在及best fitness有限断言已先于报错执行通过。但完整参数、全部tensor有限值、架构/类别检查尚未完成，不能认定checkpoint完全可恢复或不可恢复。
- 已知请求训练参数`device=0`，原训练`initialization.json`的有效配置`device="0"`；当前检查先把checkpoint参数与请求字典直接比较，尚未执行随后与原有效配置逐项对照。这是需要检查的类型/规范化位置，不以该错误认定GPU配置实际改变或标签有问题。
- 无guard拒绝事件文件。未启动D080续训、D070/D060新训练或十二次统一评估；未发生优化器更新、未修改原中断目录/权重/旧源码/seal。
- 恢复执行器仍为开发中产物：现有3项合成契约通过；后加的参数接线及恢复报告契约尚待实现/运行。当前不是可交付完整执行器，不应调用其prepare/run等未实现分支。
- 按失败即停约定，不自动修复后重跑检查。后续需具体批准修正有效参数核对、补齐恢复执行器及测试，并在新检查/运行目录重试一次；既有B方案、超参数、评估矩阵和新预算不重新选择、不自行扩大。

问题解决来源：`AI 直接解决`（开发中工具、机械检查和停止记录）；检查由AI编写，不能归为学习者或模型实验决策错误。没有完成恢复，无新实验指标。
