# 仓库结构

| 路径 | 内容与命名 |
| --- | --- |
| 根目录 | 项目首页、目录约定和示例依赖，例如 `README.md`、`requirements.txt`。 |
| `tools/` | 标签候选筛选、审核和版本应用的 Python 模块，以及关联的审核模板。脚本用小写下划线命名，例如 `oof_label_cleaning.py`。 |
| `docs/plans/` | 审核反馈的机器可读 schema。保留现有模块读取的路径，例如 `cleaning-expansion-review.schema.json`。 |
| `examples/` | 可匿名获取的合成输入及示例入口。输入以 `sample` 命名，入口按工作用途命名。 |
| `tests/` | 合成回归测试。文件以 `test_` 开头，随后使用模块名。 |
| `reports/` | 可公开的汇总指标、实验配置和复查说明，例如 `metrics.csv` 和 `g02.json`。 |
| `.venv/`、`demo-output/` | 本地依赖和示例输出，均由命令生成并经 Git 忽略。 |
