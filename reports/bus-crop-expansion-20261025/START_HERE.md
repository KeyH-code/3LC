# 当前状态：输入已准备，启动失败；未执行模型，不自动重试

本轮按学习者批准的方案1保留五组492张train，使用多个固定320×320裁剪放大2×到640，共1239输入。单裁剪尺寸不再是阻塞项。

## 实际执行

- 输入准备完成，生成1239个PNG及5张ROI示意图；坐标和来源冻结于plan_config.json、sample_manifest.json、crop_manifest.json。
- 唯一一次子进程启动约0.256秒后退出，退出码1，未超时。
- inference.log报 `KeyError: '_worker'`：原CLI通过globals按命令名查函数，但内部函数名为worker，不是_worker。此错误在进入工作函数前发生。
- 无worker_started.json、无worker_result.json；partial目录为空；predictions.json无记录。没有完成模型加载或推理。run_status.json的model_attempts=1表示已消耗一次子进程启动，不表示模型实际执行过。
- **所有零候选数都是未执行的占位计数，不是“未发现bus”或“没有漏标”的观测。没有生成review.html；旧README中打开review.html及成功运行verify的说明不适用于本次失败状态。**

## 保存与修复

1. 原准备、启动状态、日志、预测、RESULT、README、verification保持原样，未覆盖失败证据。
2. failed_startup_verification.json记录修复前的只读独立复验：重建全部1239输入、核对全部17668原始文件、保护入口/配置、指定权重及旧试点101项产物，均未变；确认工作函数未进入。
3. failed-startup-sources/按原路径保存失败时6份源码/模板/测试，哈希与plan_config.json的source_sha256一致。
4. 项目当前tools/bus_crop_expansion.py已改为显式命令到函数映射；当前tests/test_bus_crop_expansion.py新增四命令的mock分派检查，不执行任何模型。最终124项相关测试通过（25.722秒）。
5. **当前源码哈希有意不同于旧计划冻结哈希**，仅上述执行器及测试变化。不会修改旧计划伪装修复在运行前发生；旧run锁仍保留，不能直接再次run或通过删除锁重试。工具verify用于成功运行，本次不声称它通过。

## 已做/未做的浏览检查

新模板在内置浏览器通过脚本语法和34个唯一DOM ID检查；图片加载失败清空旧画面问题已修复并有回归测试。由于本次没有预测，未检查正式数据页交互、渲染或候选质量，不声称完成目视审核。

## 验证命令

在项目根目录执行以下命令，仅检查已归档哈希，不推理：

```powershell
@'
from pathlib import Path
import json,hashlib
r=Path('reports/bus-crop-expansion-20261025')
v=json.loads((r/'verification.json').read_text(encoding='utf-8'))
for name,h in v['artifact_sha256'].items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
cfg=json.loads((r/'plan_config.json').read_text(encoding='utf-8'))
for name,h in cfg['source_sha256'].items():
    assert hashlib.sha256((r/'failed-startup-sources'/name).read_bytes()).hexdigest()==h,name
assert not (r/'worker_started.json').exists()
assert not (r/'worker_result.json').exists()
assert not list((r/'partial').iterdir())
assert json.loads((r/'predictions.json').read_text(encoding='utf-8'))['records']==[]
print('Failed-startup evidence and frozen source hashes verified; no inference executed')
'@ | & 'E:\anaconda\envs\dl-3lc\python.exe' -I -B -
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B -m unittest discover -s .\tests -p 'test_bus*.py'
```

## 后续边界

本次明确约定失败后不自动重试，因此修复只完成到可重新授权状态。需要学习者明确允许本轮启动失败后再执行一次，才可在保留上述证据并冻结修复版来源的前提下继续；仍限相同492图/1239视野、相同权重和参数、单子进程900秒，不扩样、不换模型、不读test、不新推理val、不训练或修标。

热区覆盖仍只是已知原标签像素覆盖，可能存在裁剪截断、跨视野重复与未知train目标不在热区等限制；候选不能直接视为漏标。轻量浏览、组级/区段口头反馈的目标不变，不要求逐框JSON。未联网、未访问3LC凭据、未安装依赖、未暂存/提交。
