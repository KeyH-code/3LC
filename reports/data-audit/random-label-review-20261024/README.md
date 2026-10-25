# 随机样本漏标人工审阅：48行／96张

日期：2026-10-24。问题解决来源：AI直接解决（机械抽样、审阅工具和验证）；图像审核结论由学习者填写。

## 打开

[打开本轮随机漏标审阅HTML](review.html)

绝对路径：

`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516\reports\data-audit\random-label-review-20261024\review.html`

- 单文件约7.34MB，内嵌96张原图及原标签，无外部图片/脚本依赖、无网络请求。
- 左train48张，右val48张，合计48行；**左右只是展示配对，没有相似场景或近邻筛选条件**。
- 每张图显示现有标签框数，可隐藏原框、放大、填写漏标判断、疑似漏标数量和备注。
- 请选择“发现漏标／本轮未发现漏标／不确定”；没有审核的图保持“未审核”。漏标数量可留空，填写时是缺少的框数，不是原有框数。
- train/val分开统计。未审与不确定不计入明确判断分母；已填疑似漏标数量也只汇总你填写的数，不生成精确新框。
- 审核记录自动保存在当前浏览器本地。建议经常点击**导出审核JSON**备份，审核完把JSON交回；可导入同批次JSON续审。
- 当前所有96张均未填写人工结论；测试数据已清除，不把AI测试当作学习者审核。

## 用户确认的抽样口径

学习者原先说0.1%和约48对，数量不一致。经提问，学习者明确选择**全量train/val分别随机48张、左右48行**，而不是从4862对近重复中继续抽。随机种子由学习者明确委托AI随机选择。

- 种子：**3323167602**，由`secrets.randbelow(2**32)`生成，持久化在[selection_config.json](selection_config.json)。重建读取该种子，不重新挑选。
- 方法：候选路径排序后，使用Python `random.Random(seed)`，先从train池不放回抽48，再从val池不放回抽48；不使用模型分数或疑似漏标程度挑样本。
- 排除上一轮审阅HTML内嵌的全部413张图，另跳过已确证16张val同图；两类排除存在交集，不能简单相加。
- 实际抽样池：train **7620**、val **787**；分别抽取48张，约为各自池的0.630%与6.099%。这次不再标为0.1%。
- 因为排除了已展示图，这是剩余候选池内的随机审阅，不是对全部原始8834图的无条件简单随机抽样。两组各48也不代表按原数据比例抽样，不能直接把两组混合百分比称为全量漏标率。
- 样本ID：`c34fe86f03c3eccb96c2a5cd27205916cb034a986bee0a9c11f9fe9d26bca4a7`。
- 路径、框数、原图/标签SHA256和完整排除清单见[review_manifest.json](review_manifest.json)。

## val16张的处理已澄清：不移动，只生成取用清单

用户澄清：原train/val划分不变，仅不取用val中的16张同图。因此此前“移入train”方案不执行，也不再等待迁移范围确认。

- 原文件仍train7852图、val982图，所有图片/标签位置及内容不变。
- 已保存[val_use_list_excluding_16.txt](val_use_list_excluding_16.txt)：**966个绝对图片路径**，只排除那16张val图片；不是把本轮排除413图也永久排除。
- [val_use_list_metadata.json](val_use_list_metadata.json)记录原数量、排除项及清单SHA256。
- 这只是清单产物，**没有改训练/评估入口或配置，没有运行评估**。若以后用它评估，应注明val966口径，不能直接同原val982的旧结果横比。
- 本轮96图人工审核及其意见仅用于质量审计，不直接补val标签或进入训练；相关竞赛规则边界不因生成清单而解除。

## 验证

- 构建逐图核对原图/标签字节及SHA256，并完整解码。
- 浏览器全部8页、96图加载成功，左右列和SVG框数匹配原标签；分组统计、漏标计数、未审/不确定分母处理、保存、搜索、筛选、框开关、放大、JSON序列化/导入校验均已实际测试。未触发下载按钮，不擅自写下载目录。
- 初版图片已加载但`Image.decode()`等待未结束，改为complete/load/error判断后重测全部8页通过。
- 截图发现窄面板下780px表格超出668px逻辑视口，已将最小宽度调整为600px并使表单可收缩；最终CSS窄面板复测因宿主提示当前会话不可见而受阻。该最后一项未报为通过，核心功能与样本未受CSS变更影响。详见[browser_checks.json](browser_checks.json)。
- 文件/抽样排除/内嵌原图原标签/val966清单和全部17668原数据文件未改动核验见[verification.json](verification.json)。
- 没有训练、推理、提交、补标、删样本、裁剪或移动；旧审阅HTML及其意见不覆盖。

## 复现

```powershell
$py='E:\anaconda\envs\dl-3lc\python.exe'
$r='E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516\reports\data-audit\random-label-review-20261024'
& $py -B -W error "$r\build_review.py"
& $py -B -W error "$r\build_review.py" --verify
```

第一条仅重建本目录审阅产物，保留种子和样本ID，生成时间可变。第二条只重建本目录验证JSON。环境仅使用指定的`E:\anaconda\envs\dl-3lc`，不安装依赖。
