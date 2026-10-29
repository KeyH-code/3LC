# 2026-10-29 SS04 A/B完整对照完成

- **阶段**：本地完整A/B训练后的六次统一评估、汇总与独立静态核验。
- **决策/操作**：按学习者明确授权，仅补六次评估；A/B已有权重不变，无重训、加训或自动评估重试。普通Ultralytics，无官方入口或3LC。原训练执行器、seal和失败目录原样保留；恢复程序/参数另封存。
- **原因**：修复首次评估的错误权重路径，并在同一FP32口径下完成原始train/清洗train/val966对照。
- **结果**：六次评估全部退出码0；报告生成且success.json存在；恢复监督耗时862.998秒（约14分23秒），六次worker评估合计622.9秒。统一val966上：A/B AP50=0.809509661/0.828406936，AP50–95=0.656022221/0.669736304；B−A分别+1.890/+1.371个百分点。四类AP50–95增量：truck +1.482、car +0.681、van +1.558、bus +1.764个百分点。
- **工作点区别**：各自最佳F1点P/R变化−2.278/+2.751个百分点，不冒充同阈值；固定conf.25/IoU.50宏平均P/R变化+0.485/+1.609个百分点。固定点truck误检18→13、漏检66→58；car误检801→803、漏检1627→1603；van误检170→180、漏检254→248；bus误检34→40、漏检225→209。不将这些计数推广到所有阈值。
- **train与差距**：同原始train标签评分AP50–95：A .812963530、B .799106005；同清洗标签评分：A .812333355、B .806348966。相对共同val966的train−val AP50–95差距分别15.694→12.937、15.631→13.661个百分点；train AP下降与val AP提高均参与差距缩小，不据此直接断言过拟合已解决。
- **独立核验**：不加载模型；best/last四个权重哈希、相同初始参数哈希、原15项seal与恢复13项seal、33520个保护源文件、原/恢复各33340个数据文件、三份报告哈希均匹配。六次结果对应正确权重/标签参照、图像数/框数/FP32参数一致、每次两个受guard保护子进程，六次评估无guard拒绝事件；诊断及旧失败预检事件仍保留且不计成功。证据 `reports/eval-cleaning-ab-recovery-20261029/independent-final-verification.json`。
- **收获**：未验收学习者理解；客观展示同阈值P/R与最佳F1点P/R的区别，不代写学习者收获。
- **踩坑**：旧路径缺失引起在线查找尝试（均被阻止），恢复改为只接受真实已校验best；首次加载预检的__pycache__探测拒绝保留，改为直接加载自有源码。准备阶段298项回归通过，不能以测试通过代替真实评估核验。
- **问题解决来源**：`AI 直接解决`（恢复工具/测试/机械执行/报告/核验）；A/B及SS04配置由学习者指定。
- **限制与后续计划**：单种子单配对，共同val也参与best选择，无独立test盲测；不声称显著性或竞赛流程合规。不自动选模型、重训或设计下一实验，等待学习者决策。未提交或推送Git。

## 产物与验证命令

- 报告绝对路径：`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\reports\eval-cleaning-ab-recovery-20261029\RESULT.md`
- 机器可读：同目录 `comparison.json`（整体/分类原生与固定点指标、两种train参照差距），`metrics.csv`（六次汇总），`success.json`（报告哈希）。
- 评估参数：640/batch32/workers2/FP32 quantize=None/rect=false/conf.001/NMS IoU.7/max_det300/无TTA；固定conf.25先过滤再原生IoU.50匹配。
- 本次目录外读取仅授权环境 `E:\anaconda\envs\dl-3lc\`，缓存与输出在项目内。

在项目根目录复核三份最终报告哈希（只读，无模型执行）：

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B -c "import pathlib,json,hashlib; p=pathlib.Path('reports/eval-cleaning-ab-recovery-20261029'); s=json.loads((p/'success.json').read_text()); assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in s['reports'].items()); print('3 report hashes verified')"
```

不直接重跑一次性prepare/preflight/run，不删除证据绕过单次限制。
