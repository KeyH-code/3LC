# 离线接口阻塞（静态源码核查，未执行import）

库根：E:\anaconda\envs\dl-3lc\Lib\site-packages\。

- tlc_ultralytics/engine/model.py 顶层import tlc；predictor虽为普通DetectionPredictor，但导入先发生。
- tlc/__init__.py导入_core，tlc/_core/__init__.py调用init_global_objects。
- tlc/_core/init_global_objects.py:40起先调用ensure_key_activated，随后才配置/Telemetry/内部表初始化。
- _tlc_auth_state.py:639起要求有效认证handle，否则走license/API-key激活；971起的_resolve_api_key在未提供key时回退到native默认凭据文件。
- TLC_CONFIG_FILE可关闭用户配置发现，TLC_DISABLE_3LC_ROOT_DIR=1可禁根标记发现，TLC_DEFAULT_ROOT_DIR可隔离Python默认数据路径；TLC_SENTRY_ENVIRONMENT=local禁用3LC Sentry，TLC_COLLECTION_DISABLE=true可禁collection。**这些都不是跳过导入认证的开关。**
- 凭据候选路径来自native组件，未核实它是否遵循APPDATA/LOCALAPPDATA/HOME重定向；Python socket/open保护不能证明native完全受控。没有读取该native组件中的凭据值、没有反编译或执行认证。

批准计划要求：若无法在不读取全局配置/凭据、不产生未授权副作用的情况下加载，则报告阻塞，不自行切换模型栈。故本轮未导入模型库、未加载权重、未运行推理。

可能的后续变更（**本轮未授权或执行**）：单独批准同环境普通ultralytics接口离线加载同一best.pt、固定样本/参数不变；或者另行明确授权3LC认证及精确凭据路径/网络范围。未修改依赖或monkeypatch认证。
