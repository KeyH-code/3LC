# OOF ≥0.85 清洗准备状态

- 当前状态：`awaiting_user_scope_confirmation`。模型选择授权与实际图像输入能力分别核验，不能互相替代。
- 候选：744条 / 591张 / 69场景。
- 逐项目标状态：{"defer": 196, "change_class": 58, "keep": 96, "add_box": 394}。
- 旧人工目标关联：38；新增视觉裁定：706。
- 审核来源分布：{"gpt-6-luna": 377, "unknown (historical AI; human adjudication retained)": 24, "current-session-subagent": 313, "current-session-parent": 30}；当前会话实际模型：{"cpa/gpt-6.1-sol-high": 343}。
- 当前具备原框定位条件的改类建议：58；补框建议：392。这不是最终批准规模。
- 人工已确认漏标但坐标仍待定稿：2。
- Visual review complete; proposed scale only. Await separate phase-B confirmation; no labels applied. 没有标签修订数据集。
- 先验证获准审核路径实际接收图像，再完成审核；最终汇总后仍必须等待第二次确认才可apply。
- 核查页index.html三列显示，图片原始上下文与目标局部均可查看；加载成功不代表视觉审核完成。
- 模型置信度仅作筛选，不是标签错误概率。人审意见按具体目标关联，不能推广到同图/邻帧其他目标。
