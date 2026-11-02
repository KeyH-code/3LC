# 新一批30张：人工反馈归档

- 30/30收到反馈，29例已对应到所选目标；案例19保留目标/框对应待核对。
- 漏标11、错类7、原标正确/模型错10、不应标注1、目标对应待核对1。
- 案例9、12的漏标结论结合“应标范围”回答与已核实无原框覆盖得出；car类别来自问题上下文，不冒称用户逐字明确了类别。
- 案例22人工判van，推翻AI倾向保留truck的意见；不覆盖原AI记录，不据本样本计算泛化准确率。
- 案例19人工指认黄色van，van判断保留；在匹配对应车辆与GT前，不将它纳入已确认错类数量。
- 案例01不应标注仅适用本例，不批量追改前8张或推广到所有警用车辆。
- 原反馈文件按字节保留；标签写入0、模型调用0，无训练/推理/提交/推送。
- 学习者决定后续采用Agent明确意见，低置信或不确定部分交人工审核；详见WORKFLOW_DECISION.md。

## 逐例记录

### 案例01 · MVI_40902/235
- 原话：不应该标注
- 归档状态：not_required_to_annotate；类别：不适用/未单独指定
- 解释：Human says this selected police work vehicle should not be annotated. Scope applies to this case; do not generalize to all special vehicles or rewrite prior case1 feedback.

### 案例02 · MVI_40743/30
- 原话：确实漏标
- 归档状态：missing_label；类别：bus
- 解释：Human confirms missing annotation or in-scope/should-annotate status; verified no original coverage establishes a missing-label finding for this selected target.

### 案例03 · MVI_39361/1706
- 原话：确实漏标
- 归档状态：missing_label；类别：bus
- 解释：Human confirms missing annotation or in-scope/should-annotate status; verified no original coverage establishes a missing-label finding for this selected target.

### 案例04 · MVI_39501/518
- 原话：是的
- 归档状态：missing_label；类别：car
- 解释：Human confirms missing annotation or in-scope/should-annotate status; verified no original coverage establishes a missing-label finding for this selected target.

### 案例05 · MVI_40904/919
- 原话：漏标，当前界定成立
- 归档状态：missing_label；类别：car
- 解释：Human confirms missing annotation or in-scope/should-annotate status; verified no original coverage establishes a missing-label finding for this selected target.

### 案例06 · MVI_40244/111
- 原话：漏标，属于van
- 归档状态：missing_label；类别：van
- 解释：Human confirms missing annotation or in-scope/should-annotate status; verified no original coverage establishes a missing-label finding for this selected target.

### 案例07 · MVI_39401/1331
- 原话：漏标
- 归档状态：missing_label；类别：bus
- 解释：Human confirms missing annotation or in-scope/should-annotate status; verified no original coverage establishes a missing-label finding for this selected target.

### 案例08 · MVI_39511/318
- 原话：漏标
- 归档状态：missing_label；类别：bus
- 解释：Human confirms missing annotation or in-scope/should-annotate status; verified no original coverage establishes a missing-label finding for this selected target.

### 案例09 · MVI_40962/1425
- 原话：在标注范围内
- 归档状态：missing_label；类别：car
- 解释：Feedback confirms scope/should annotate, not an explicit missing-label sentence; combined with verified absence of original annotation, record missing label. car class comes from question/selected target context.

### 案例10 · MVI_40904/1065
- 原话：属于，漏标
- 归档状态：missing_label；类别：bus
- 解释：Human confirms missing annotation or in-scope/should-annotate status; verified no original coverage establishes a missing-label finding for this selected target.

### 案例11 · MVI_39501/233
- 原话：应该标，这里确实漏标
- 归档状态：missing_label；类别：bus
- 解释：Human confirms missing annotation or in-scope/should-annotate status; verified no original coverage establishes a missing-label finding for this selected target.

### 案例12 · MVI_39511/90
- 原话：应该标注
- 归档状态：missing_label；类别：car
- 解释：Feedback confirms scope/should annotate, not an explicit missing-label sentence; combined with verified absence of original annotation, record missing label. car class comes from question/selected target context.

### 案例13 · MVI_41063/1018
- 原话：应该改为van
- 归档状态：wrong_class；类别：van
- 解释：Human class differs from existing GT: record a class correction finding for the existing target, not a new box.

### 案例14 · MVI_40172/503
- 原话：确实是van
- 归档状态：wrong_class；类别：van
- 解释：Human class differs from existing GT: record a class correction finding for the existing target, not a new box.

### 案例15 · MVI_40852/1072
- 原话：应该为van
- 归档状态：wrong_class；类别：van
- 解释：Human class differs from existing GT: record a class correction finding for the existing target, not a new box.

### 案例16 · MVI_40981/1623
- 原话：van
- 归档状态：original_label_correct_model_wrong；类别：van
- 解释：Human retains the original class and rejects the conflicting model class; no class change needed for this selected target.

### 案例17 · MVI_40903/689
- 原话：应该改为van
- 归档状态：wrong_class；类别：van
- 解释：Human class differs from existing GT: record a class correction finding for the existing target, not a new box.

### 案例18 · MVI_20012/113
- 原话：应该改为car
- 归档状态：wrong_class；类别：car
- 解释：Human class differs from existing GT: record a class correction finding for the existing target, not a new box.

### 案例19 · MVI_40213/782
- 原话：对应的是一辆黄色的van
- 归档状态：target_mapping_pending；类别：van
- 解释：Human explicitly identifies a yellow van. Original AI observation/selected overlay described a dark higher-bodied right-edge vehicle with occlusion and neighboring targets. Preserve van judgment, but do not silently attach it to the selected GT index, declare a correction, or contradict the human target identification before matching the object.

### 案例20 · MVI_40855/2
- 原话：应该是van
- 归档状态：original_label_correct_model_wrong；类别：van
- 解释：Human retains the original class and rejects the conflicting model class; no class change needed for this selected target.

### 案例21 · MVI_40191/53
- 原话：应该是van
- 归档状态：original_label_correct_model_wrong；类别：van
- 解释：Human retains the original class and rejects the conflicting model class; no class change needed for this selected target.

### 案例22 · MVI_63561/676
- 原话：应该是van，不是truck
- 归档状态：wrong_class；类别：van
- 解释：Human class differs from existing GT: record a class correction finding for the existing target, not a new box.

### 案例23 · MVI_63563/637
- 原话：是的，应该保留van
- 归档状态：original_label_correct_model_wrong；类别：van
- 解释：Human retains the original class and rejects the conflicting model class; no class change needed for this selected target.

### 案例24 · MVI_40775/425
- 原话：保留bus
- 归档状态：original_label_correct_model_wrong；类别：bus
- 解释：Human retains the original class and rejects the conflicting model class; no class change needed for this selected target.

### 案例25 · MVI_40161/1206
- 原话：保留van
- 归档状态：original_label_correct_model_wrong；类别：van
- 解释：Human retains the original class and rejects the conflicting model class; no class change needed for this selected target.

### 案例26 · MVI_40854/711
- 原话：属于bus
- 归档状态：original_label_correct_model_wrong；类别：bus
- 解释：Human retains the original class and rejects the conflicting model class; no class change needed for this selected target.

### 案例27 · MVI_40905/27
- 原话：是van
- 归档状态：wrong_class；类别：van
- 解释：Human class differs from existing GT: record a class correction finding for the existing target, not a new box.

### 案例28 · MVI_40853/375
- 原话：应该保留van
- 归档状态：original_label_correct_model_wrong；类别：van
- 解释：Human retains the original class and rejects the conflicting model class; no class change needed for this selected target.

### 案例29 · MVI_40963/643
- 原话：是van
- 归档状态：original_label_correct_model_wrong；类别：van
- 解释：Human retains the original class and rejects the conflicting model class; no class change needed for this selected target.

### 案例30 · MVI_40864/349
- 原话：是van
- 归档状态：original_label_correct_model_wrong；类别：van
- 解释：Human retains the original class and rejects the conflicting model class; no class change needed for this selected target.

## 复现验证

在项目根目录运行：

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' 'reports\oof-human-review-30-20261027\archive_feedback.py' verify
```
