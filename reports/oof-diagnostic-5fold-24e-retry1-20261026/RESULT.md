# 5折OOF离线诊断完成

每折只用fit更新参数、monitor决定早停/best；outer在权重冻结后才预测。没有使用val966或test。没有自动判错/改标签、模型集成或第六次训练。

|折|fit|monitor|outer|轮数|best轮|早停|训练秒|
|---|---:|---:|---:|---:|---:|---|---:|
|1|5629|628|1595|24|21|False|1200.06|
|2|5637|638|1577|24|22|False|1526.20|
|3|5664|641|1547|24|23|False|1948.63|
|4|5677|612|1563|24|23|False|1493.72|
|5|5644|638|1570|24|21|False|1473.85|

全体7852张每图唯一一次OOF预测，含空预测，低置信结果完整保留。指标相对于可能错漏的原标签，不是数据质量真值；场景外分布变化、近重复残留、内层标签噪声和24轮能力限制仍存在。oof_metrics.json为明示匹配规则的统一saved-record评估，不冒充原生Ultralytics精确匹配或加权平均AP。

打开review.html查看原图四类原标签与预测；预测阈值仅用于显示。不收集逐框JSON，不自动给错误概率排序。

SS03训练增强保留；plots=false仅避免字体/画图副作用。AMP启用但跳过外部权重自检，没有声称自检通过。Python层I/O/network guard是纵深防御，非原生OS沙箱。

训练/预测产物仅作本次获准离线诊断，未获主办方独立训练入口认可，不用于伪标签/提交或竞赛集成。未改官方入口/默认配置/AGENTS/原标签，不安装依赖、联网或读取3LC凭据。

只读复验：
```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B .\tools\oof_diagnostic.py verify
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B -m unittest discover -s .\tests -p 'test_oof*.py'
```
