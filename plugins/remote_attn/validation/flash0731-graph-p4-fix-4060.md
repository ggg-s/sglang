# Flash-0731 分段图与主动 D2D 入口修复

基线：`eae838a180`。代码与 4060 验证完成；161 H20 整模性能和真实主动迁移验收待执行。

- V4 C4/C128 可选择 DSV 分段 backend，不再固定返回 full backend。投影捕获在图内，SWA/indexer/独立 compressor 使用 live metadata；C128 不调用 indexer。
- 启用分段图时在原生 warmup 前验证 split ABI；显式关闭后仍保留原生完整图。默认 eager/full prepare dispatch 与 upstream main 的 AST 一致。
- `--language-model-only` 接受 V4 和 V4.1 原生 PD。
- P4 gateway 新增 `--src-control-dir/--dst-control-dir`，可连接现有 D0/D1 控制目录；nonce 和逐 rank ready 校验继续使用原 driver。D2D 缺 ready 时拒绝无 nonce 的 legacy fallback。
- 新增 `bench_p4_dsv_migration.py`：由 P4 提交可控 rid，cut 自动执行预留后冻结，消费 SSE 直至目标接管和 DONE。自然排空不计为迁移成功；部署命令见 P4 DSV 文档。

361 项主回归、9 项原生接口/新回归和额外 1 项 DSV4 ABI 启动门控通过。真实本地 HTTP/SSE 校验了迁移成功与自然结束负例（本机测试绕过系统代理）。
4060 的 DSV4 默认/融合两种配置及 V4.1 兼容配置均通过：每组 78 次 remote 重放、24 个原生 full shape、5 个 remote bucket，误差 0；投影不在 eager replay 重复，live SWA/carry/indexer 状态与 Local 恢复得到校验。真实 Mooncake TCP Q/O 传输执行；原生模型方法/图 backend 为源码加载，小型投影/norm/store、压缩器规划与3层模型仍为 fixture。

这些结果不证明 H20 Flash-0731 整模、RDMA HCA 或性能收益。分段图不消除逐层 RPC，`HANDOFF_REMOTE_GPU_TRANSFER=1` 与 engine RDMA 初始化日志也不能代替逐请求 backend 与 path 证据。需要在161复测 native/remote，并实际驱动 P4 主动迁移。当前未获该机器 SSH 连接信息，未改其运行服务或工作树。

[原始结果与源码哈希](flash0731-graph-p4-fix-4060.json)
