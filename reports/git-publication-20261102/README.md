# Git阶段提交记录（已批准范围；实际commit以Git记录为准；不push）

## 只读检查结论

- 当前分支`main`，HEAD短哈希`d75cd26`；检查时暂存区为空。
- 已跟踪改动仅`PROJECT_LOG.md`及`README.md`，合计607行新增、1行删除；保留既有改动，不撤销或覆盖。
- 准备文件生成前，未忽略的未跟踪文件29791份、396242089字节（约377.9 MiB）。其中26454份为txt，许多位于真实派生数据/审核运行目录。当前忽略规则不足以阻止误入库，不能使用`git add .`。
- Git对象当前松散116个、约464 KiB；已打包103个、约2185 KiB。该值只描述本地对象存储，不代表GitHub配额或远程仓库大小。本次未联网检查远端。
- 现有`.gitignore`已排除权重和常见数据目录，但`datasets-20261101`、新清洗报告树、运行配置及大manifest仍有漏网项。建议新实验目录默认排除，仅放行精确轻量产物；不删除本地文件。

## 建议提交范围

精确候选清单见`commit_files.json`：新增机械审核/应用/训练恢复/汇总源码及测试、schema/计划、授权与中断恢复说明、应用规模和七份最终指标报告。旧A/B核心汇总已在HEAD，无需重复添加旧runtime。

排除图片、标签副本、模型权重、缓存、凭据、内嵌图片HTML、大型候选/逐图清单、会话输出及临时视觉脚本。seal仅作为轻量历史来源索引；其引用的完整本地文件不会全部入库，不宣称clone后已有完整数据和历史证据。

学习者已确认范围并要求更新README，且明确先D060 test生成CSV后继续本地Git提交。README已同步三档实验/终验与CSV生成状态，保留历史阶段并明确本地链接限制；`.gitignore`已补齐新实验树默认排除与精确放行。不触碰AGENTS/RULES和已封存执行源码。

本次新增D060 CSV生成工具/测试及5份轻量说明/验证记录纳入清单；预测CSV、test、权重和inputs/runtime不入库。373项合成回归通过（41.596秒），一次982图推理已完成且CSV格式核验通过，AI未上传Kaggle。

## 验证

- 当前合成回归370项通过，47.660秒，退出0。命令：

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B tools/cleaning_expansion_regression.py
git -c core.safecrlf=false diff --check
git diff --cached --stat
```

- 候选文本按UTF-8检查、凭据特征及内嵌图片扫描；结论见`checks.json`。模式扫描未检出不等于证明绝无敏感信息，实际暂存前仍需复核diff与精确名单。
- 待README/忽略规则获准更新后，重新核对候选文件哈希、忽略规则命中和暂存体积；当前清单是准备方案，不是已经执行的commit。

## 建议commit说明

`feat: archive cumulative label cleaning and verified SS04 recovery results`

按学习者已批准的精确清单执行本地暂存/commit，不push，不删除任何未跟踪或忽略文件，不上传Kaggle。准备期原始检查数字保留，实际入库体积、检查状态和后续commit以checks及Git记录为准，不预写提交成功。
