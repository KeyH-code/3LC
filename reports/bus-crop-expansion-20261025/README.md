# 五组热区2×扩样

先读RESULT.md，打开review.html。所有492张按组和命名帧号保留，非视频时间/车辆身份。可筛“原图无bus标签且有高bus候选”，仅为待核对线索；不要据筛选数直接认定漏标。无需逐框填意见，直接给组级/区段反馈即可。

左侧原图显示当前视野的映射候选或全部视野叠加，右侧为真实640PNG输入。蓝色原bus标签、绿色高候选、橙色弱候选；tile ID区分跨裁剪重复。与原bus标签的最大IoU只披露相似程度，不设自动真值阈值。

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B .\tools\bus_crop_expansion.py verify
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B -m unittest discover -s .\tests -p 'test_bus*.py'
```

verify只读，不再推理。run有独占保护，已启动即不自动重跑；不删除已有证据重试。所有新增缓存/工作目录在本会话scratch。历史13图试点和失败的单ROI预检均保留。
