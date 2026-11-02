# 三档清洗实验：Agent优先审核与下一会话交接计划

> **执行会话**：使用 `executing-plans` 按阶段推进；以下是已批准原计划的补充，不替换其标签应用/训练门禁。

**Goal:** 以指定gpt-6-luna-high完成固定6188条候选审核，形成三档累计提案，再按批准门禁开展三次SS04训练及统一指标产出。

**Architecture:** 候选、D085参照和原实验规格已经冻结。直接给Agent绑定原图/必要局部和机器可读候选，用短JSON反馈收集视觉判断；主Agent做身份、证据、几何、去重与提案核验，不为人工网页设计审核流程。

**Tech Stack:** 指定模型 `cpa/gpt-6-luna-high`、项目内Python/Pillow工具、已有OOF/审核规则；后续训练为现有Ultralytics环境，非官方入口/3LC。

---

## 1. 本会话收尾状态与优先读取顺序

**本会话到此仅完成计划补充和交接。不再启动视觉审核、标签应用、推理或训练，不派发新的实验Agent。**

下一会话在项目根目录 `E:\kaggle\3LC_Multi_Vehicle_detection_Challenge` 打开，先读：

1. `AGENTS.md`：授权、读取边界、记录义务。
2. 本文件：最新用户裁定、Agent输出协议和恢复入口。
3. `.pi/plan/三档累计清洗实验-080-070-060-SS04固定-20261029.md`：原完整实验假设/SS04/12次评估/门禁。原文“草案/模型待确认”等是冻结时状态，批准及最新模型选择见补充授权；不要修改原文件破坏来源哈希。
4. `reports/cleaning-expansion-080-070-060-20261029/review_authorization_v2.json`。
5. 同目录 `freeze.json`、`experiment_spec.json`、`independent-preparation-verification.json`。
6. `logs/cleaning-expansion-preparation-20261029.md`、`PROJECT_LOG.md`最新段落；再按需要读已有结果。

所有外部本机资源只允许已明确指定的 `E:\anaconda\envs\dl-3lc\`；如要读其他路径先请用户指定。不要寻找旧宿主全局Agent配置、兄弟项目或凭据。工具可返回的托管输出路径不扩展成任意外部读取权限。

## 2. 最新用户裁定与不可变实验条件

- 新增三次训练，阈值锚点提前固定 **0.80 / 0.70 / 0.60**，SS04不变。
- 固定旧OOF和D085标签参照、累计修订，先冻三档提案再看新val；不新跑OOF、不自适应改后续范围。
- 审核Agent明确使用 **gpt-6-luna-high**，不是一般gpt-6-luna、当前会话模型或别的high模型。
- 本会话通过模型目录发现精确可选标识 `cpa/gpt-6-luna-high`。下一会话重新检查可用性，明确传入该标识；不猜另一个variant，不擅自加fallback。不将菜单记录冒充实际运行后端证明。
- 保留Agent意见来源，以Agent明确意见为主，不要求学习者逐项重看6188条。此前人审经验支持采用这一工作方式，但不等于本轮Agent结论已被人审确认或全量正确。
- 不确定/冲突项目保留defer和精简例外清单；默认不改，不强迫用户逐条裁定。只有确实需要范围/规则裁决时集中提问。
- 审核形式Agent优先：不用完成人工网页交互/布局验收，不强求浏览器截图、每页三图或人工可编辑界面；可以原图读取＋必要局部，所有视觉输入仍须真实进入审核Agent上下文。
- 原`index.html`文字有旧作用域，已保留；`review_index_v2.html`已改正确作用域，但仅是可选辅助，不作为Agent审核前置条件。

## 3. 当前确切进度

| 分档 | 置信区间 | 候选 | 涉及图片 | 相对D085累计候选 |
|---|---|---:|---:|---:|
| D080 | [.80,.85) | 1197 | 937 | 1197 |
| D070 | [.70,.80) | 2543 | 1855 | 3740 |
| D060 | [.60,.70) | 2448 | 1853 | 6188 |

- 合计6188条/3360图/129批（现有批次最多48项，机械分组不是每次必须一次审完48项）。旧>=.85残余295项不纳入。
- `review.json`仍是6188条unreviewed，占位不是意见；视觉审核0、修订标签0、三轮新增训练0。
- 候选重算、33520保护文件及20来源文件哈希核验通过；独立复算1298个GT稳定身份。原准备工具6项单测，全回归308项通过；页面工具另2项静态测试通过，不混称310项全回归已跑。
- 原/派生train图像与val、旧A/B权重均保留。上次空间约73.8GiB，仅当时观测；下一会话重新核算。
- 原计划与准备产物已封存，不覆盖重跑prepare。`AUTHORIZATION.json`、`experiment_spec.json`内“model pending”保留历史，最新裁定由`review_authorization_v2.json`补充，不回填成自始授权。

## 4. 批次短JSON反馈协议

机器约束参考：`docs/plans/cleaning-expansion-review.schema.json`。下一会话先实现/测试验证器，再接收真实审核记录；本会话不生成假审核结果。

### 4.1 批次元数据（只写一次）

- `schema_version`、`batch_id`、`attempt_id`、`requested_model`。
- `reported_model`：运行工具实际披露的标识；未披露则null，不能自填成已验证。若返回标识与指定模型不符，停止该批且不采纳，不将请求成功当后端已核验。
- `input_manifest_sha256`、`started_at`、`finished_at`。
- `vision_inputs`：实际接收的图像路径、SHA256、输入工具和`actual_image_received`；不把“路径已打开/图像解码/文字描述”当图像输入。
- `results`：逐项短记录；`uncompleted_keys`：分配但没审完的key。工具失败/上下文不足留在未完成，不批量defer冒充看过。

### 4.2 每候选记录

| 字段 | 含义与限制 |
|---|---|
| `key` | 冻结候选key，必须属于该批，不以图片文件名替代 |
| `category` | `add_box` / `change_class` / `keep` / `defer` / `blocked` |
| `confidence` | Agent对本条视觉结论的`high` / `medium` / `low`，不是YOLO置信度或校准概率 |
| `proposed_class` | 改动时0=truck、1=car、2=van、3=bus；非改动null |
| `final_xyxy` | 补框需原640坐标[x1,y1,x2,y2]；改类保持原框，填null |
| `object_key` | 补框的同图稳定目标标识，便于去重；非补框null |
| `reason_code` | 短枚举，见schema |
| `reason` | 最多160字/字符，解释为什么该候选如此处理，不写长推理 |
| `notes` | 最多160字/字符，补充限制或空串；实质疑虑不能藏在notes后仍判明确修改 |
| `visual_completed` | 真正收到并看过相关目标才true |
| `evidence_ids` | 引用批次vision_inputs，必须绑定当前候选对应图像或可追溯局部 |

明确采用项`add_box/change_class/keep`需视觉证据清楚且confidence=high；这是本轮协议的操作分流，不宣称high为统计概率。无法支持明确结论时defer；工具/图像不可得则blocked且visual_completed=false。keep也需真实看过，不能当默认。

原YOLO confidence从冻结候选机械关联到汇总记录，绝不能被Agent覆盖。来源由接收器记录`source=agent_adopted`及requested/reported model、session/run ID；人审例外另存，不把Agent意见标human_verified。

### 4.3 几何和身份门禁

- add_box需可辨目标、可辨类别、可定位边界；不得把预测框未经视觉校正直接当标注答案。
- change_class只改冻结GT的类别，保持坐标/空格；绑定D085标签哈希、原行内容与stable_target_id，不用后续漂移行号。
- 跨档/同图重复目标、补框高度重叠、相同GT不同建议、触及D085已批准动作：进入冲突队列，不自动覆盖或合并。
- 删除、改原框几何、换图片或标注语义均不属于本轮默认动作。
- reason/notes是短解释与证据限制，不要求展示长篇思维链，也不能用纯文字候选信息替代实际看图。

## 5. 下一会话的第一批执行任务

### Task 1：只读接管和能力检查

1. 读第1节文件并核对当前Git工作区，不恢复/删除其他未提交工作。
2. 运行准备只读验证命令；冻结文件不能擅改。
3. 发现模型并精确选择`cpa/gpt-6-luna-high`；不可用则报告阻塞，不替换模型。
4. 用一个本轮候选做真实图像输入能力测试；实际视觉记录单独保存，若后续导入正式审核须同样通过schema/证据门禁，不重复计数。
5. 不必重新申请已明确的模型/Agent采用方式；只在工具能力不足或协议冲突时询问。

### Task 2：实现Agent输入/输出收集（新工具，不改旧封存工具）

建议新增`tools/cleaning_expansion_review.py`、`tests/test_cleaning_expansion_review.py`。先失败测试再实现，覆盖：

- 非批次key/重复key/漏项/重复导入、schema缺字段、反馈过长、非法分类/框坐标；
- YOLO分数与Agent置信度分离、未看图不得keep/add/change、证据不对应候选拒绝；
- stable GT绑定、冲突/重叠/重复对象、旧D085动作保护；
- 错误输出不改master，原始批次/原意见不可覆盖；中断恢复及阶段计数；
- 全量未完成禁止形成可应用最终提案，未经停点A不得apply。

输入可按候选或同图分组，附原图与局部/叠框；按批次key回收记录，不必为每项生成供人工浏览的大页面。保留source image＋crop/plate生成参数和哈希；图像缩放/拼板坐标转换须可复算，不能把缩略图坐标当640真值。

### Task 3：能力通过后分批审核并可靠落盘

- 明确传入指定模型；每批输出一个JSON文件，原始返回另存，批次写独立路径，接收器串行验证并导入，避免并发覆盖master。
- 现有48项批次可按图像/上下文拆小；能力试验后再确定技术并发度，不承诺一次并发跑满129批。拆批不得改变候选归属/档位。
- 技术失败保留原始异常与未完成keys；只恢复未完成部分，不重复审核已导入结果来挑更有利意见。
- 明确意见按Agent来源采用；不确定项先留defer未应用，集中短表报告，不强迫学习者重新看6188条。

### Task 4：形成三档累计提案，停点A

- 三档审核完成后，按同一最终快照汇总补框/改类/keep/defer/blocked、类别与场景分布、实际改动文件数。
- 解决或排除跨档对象冲突，产生D080/D070/D060三档新增及累计提案和哈希。
- **用户选择Agent为审核依据，免除逐项人工视觉复核；不自动取消“实际修改规模/提案哈希确认”门禁。** 学习者只需对汇总提案范围作确认，不必重审每条图像。
- 具体提案获准后才生成标签数据版本；之后仍按原计划数据验收/停点B进行三次SS04训练和12次评估。

## 6. 后续实验固定条件速查

- 三次均随机初始化YOLOv8n、28epochs、batch32、imgsz640、seed0、deterministic、workers2、AMP；SS04完整参数见原计划/experiment_spec，不从B或上一档微调。
- D085→D080→D070→D060累计标签；在第一轮新训练前冻结三档修订，不看中间val改后续数据。
- 三新best×固定D0 train / 固定D085 train / self train / val966 = 12次核心统一FP32评估。A/B六次旧结果复用。
- 不混用训练期验证与统一FP32口径；主指标val AP50–95；原生最佳F1点P/R与固定conf.25/IoU.50分列。
- 停点B确认完整执行器及次数/预算后再训练；此前不写完整可交卷训练产物。原提议上限：每训练5400s、每评估3600s、整批21600s，无自动重试；不得继承旧过期实验预算。

## 7. 验证命令与真实证据

在项目根目录，只读验证现有准备：

```powershell
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B tools/cleaning_expansion_prepare.py verify
& 'E:\anaconda\envs\dl-3lc\python.exe' -I -B -m unittest discover -s tests -p test_cleaning_expansion_prepare.py
```

新schema仅是协议，不是已实现的导入验证器。下一会话不得仅凭schema文件存在就宣称审核门禁完整。

## 8. Git和交接可用性

- 当前最新commit `d75cd26`，仅上轮决策/轻量结果；本计划、补充授权、schema、P0/P1工具和候选多数仍为本地未提交文件。
- 新会话必须打开同一本地项目目录；仅clone该commit不会自动拥有本次完整实验材料。
- 本轮不自动提交/推送、不修改.ignore或清理未跟踪路径。如需后续提交，先按范围核对，禁止git add .混入图像、数据、权重或runtime。

## 9. 下一会话开场提示（可直接复制）

> 请在当前项目先读AGENTS.md和docs/plans/2026-10-29-cleaning-expansion-handoff.md，按已批准的0.80/0.70/0.60三档SS04计划继续。6188候选已冻结，视觉审核尚未开始。审核Agent必须明确用cpa/gpt-6-luna-high，先验证真实图像输入；以Agent明确意见为主要依据，短JSON反馈，保留来源，不要求逐项人工重看。先实现并测试接收/证据/去重门禁，再分批审核。完成三档提案后汇总并停下等我批准具体修改，不直接apply或训练；后续数据交付后还有停点B。本会话不重新选择阈值或训练参数，不覆盖旧冻结文件，不推送Git。
