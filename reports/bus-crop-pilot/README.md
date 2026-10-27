# Bus局部放大小规模试验：准备完成，推理阻塞

13图/39视野已冻结，26张局部640输入和13张ROI示意已生成。A不复制原图。B=[186,0,506,320)，C=[266,0,426,160)，双线性2×/4×；无超分辨率。

**没有预测，没有人工候选审核页，不能判断想法是否有效。** run明确拒绝执行，绝不自动切换接口、读取凭据或下载。

阻塞依据见INTERFACE_BLOCKER.md。只读verify不导入任何模型库，只验证数据、指定权重哈希、准备图像和坐标。

```powershell
$py='E:\anaconda\envs\dl-3lc\python.exe'
& $py -I -B -m unittest discover -s .\tests -p test_bus_crop_pilot.py -v
& $py -I -B .\tools\bus_crop_pilot.py verify
# prepare是独占创建，当前目录已存在，不要删除或重建覆盖。
# run会报明确阻塞，不是推理入口。
```

原标签不完整，任何未来预测都仅作候选，不计算AP/P/R，不用于标签/训练/竞赛提交；本轮不扩大样本、不更改批准参数。
