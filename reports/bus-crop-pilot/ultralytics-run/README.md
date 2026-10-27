# 固定39视野离线审核

先读RESULT.md，再打开review.html。

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B .\tools\bus_crop_pilot.py verify-run
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B .\tools\bus_crop_pilot.py summarize <项目内显式反馈JSON路径>
```

选择判断后点击“保存当前判断”，才会写入本地存储；切换前未保存会提示。可显式导出/导入JSON，不写标签。原标签仅参考，默认≥.25、弱候选单独开关。类别四类均保存，bus未检出不可由其他车辆框替代。反馈中的目标组号由人工归组；没填组号不自动推断去重或局部收益。

## 审核顺序

1. 选原图，依次切换A/B/C。左画布是映射回原图，右画布是实际640输入；蓝框是原标签，绿框是模型候选，选中框为黄色。
2. 在候选下拉框选目标框，填写判断。同一原图同一车辆在A/B/C用相同目标组号（如bus1），不要把跨视野同一辆车当多个新增目标。
3. 若目标没有对应候选，选下拉框最后一项或“另记一个未检出目标”；填写未检出/视野外/不确定。未审核不必填负面判断。
4. 弱候选开关单独控制0.05≤conf<0.25，不能把弱候选收益当成默认阈值收益。
5. 点击保存，结束后导出JSON交回汇总。真实候选质量由学习者判断，当前正式记录0条。

`verify-run`会先打印**历史准备阶段**的blocked状态（上级目录旧run_status），再打印本次`verified=true, model_inputs=39`；前者不是当前运行状态。当前状态以本目录run_status.json为准。原准备源码哈希保留为历史身份，本次实际源码哈希另存run_config.json与verification.json。

实际浏览器检查及未检查项见browser_checks.json。未实际触发原生下载保存对话框或系统文件选择器；导出Blob和导入File处理已测。单测：`& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B -m unittest discover -s .\tests -p 'test_bus*.py'`。
