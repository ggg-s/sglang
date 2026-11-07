# H20 验证反馈：aux 默认配置与 D2D 推迟

基线：`98fd13c507`，`codex/remote-attn-pre-layerwise`。

本文记录 `5240abdb8b` 的历史结果。Issue B 后续已实现部分 Remote 冻结迁移，
当前协议与 c=4/8 小池验证见 [DSV-D2D-PARTIAL-REMOTE.md](DSV-D2D-PARTIAL-REMOTE.md)。

## Issue A：默认关闭 customized-info 时仍可发布 aux strides

根因是 `validate_aux_buffers` 把显式目的端 stride 与启用 customized-info 绑定。
插件提前发布 D 的实际 stride 后，两侧尺寸相同也会失败。

修复后无需设置 `SGLANG_DISAGG_CUSTOMIZED_INFO_MAX_BYTES`：默认 0 时，显式
metadata stride 按双方相同的必需布局校验；保留原生可选 checksum 的兼容规则。
customized-info 开启时仍要求对应 slots 大小相同，且 D 的其他 stride 不小于 P。
错误会列出失败子条件、实际开关状态及双方尺寸，不再把所有失败归为尺寸不匹配。

回归覆盖首次 DSV 注册消息到 P 侧校验的完整调用边界，以及实际原生
MetadataBuffers 的默认配置、双侧/单侧 customized-info、可选 checksum 和非法 stride。

## Issue B：显式等待全部 Local；未实现部分 Remote 迁移

本次采用反馈中的方案②。`cut` 是迁移触发阈值，不保证到达该 token 时即能冻结。
仍依赖 P 端 Remote entries 的请求不能立即迁移；并发下可能一直等到源请求结束。
本修复不宣称改善这类负载的迁移次数或性能。

- worker status 报告 `migration_deferred_reason`，例如 `remote_entries_pending`、
  `previous_transfer_pending` 或 `decode_boundary_pending`。
- gateway 显式等待。journal 持久化等待状态、计数和累计/开始时间；router status、
  drain status 与 SSE status 暴露 `migration_deferred`、reason、count、wait_ms。
  稳定轮询不重写 journal，也不增加次数。
- `d2d_plan` 再次通过 TP 投票检查准入。任一 rank 尚未就绪时，所有 rank 返回
  `deferred=true`，不创建 plan、不预留、不冻结；gateway 回到 SOURCE 等待，
  不消耗迁移重试次数，也不误触发 abort/resume。
- 等待期间在源完成时，`migration_outcome=completed_on_source_before_migration`；
  实际切换至目标后才记录 `migration_outcome=migrated`。
- 既有 journal 不需要迁移。新增字段使用默认值；等待状态可随 gateway 重启恢复。

回归覆盖并发 TP 的单 rank 未就绪、status/plan 竞态、等待恢复、gateway 重启、
计数不随轮询增长、HTTP/SSE 可见性及等待期间源自然完成。
现有 `bench_p4_dsv_migration.py` 会把 SSE status 保存到结果的 `statuses`，
可直接用于复测时解释 0 次迁移；只有观察到输出 owner 切到 dst 才计为迁移。

## 验证边界

使用 Ubuntu/RTX 4060 的独立源码快照和 Torch 2.13.0+cu130。
原生 wire 测试所需 `blake3` 安装在该快照独立的 `deps/`，不修改共享虚拟环境。
完整 HTTP server 导入仍受测试环境缺少 `referencing` 限制。

验证期间还定位了一项既有 CPU fake capture 的不确定性：`empty()` 占位张量被
后续模拟模型操作读取。在修复前基线上开启 deterministic 填充可复现；现仅将
测试中的张量占位值初始化为零，metadata-only 的 None 输出保留，原断言不变。
实际 CUDA 图执行代码没有因此改变。

最终验证结果：

| 套件 | 结果 |
|---|---|
| `test_remote_attn.py` | 367 项通过，168.7 s，含真实 TP2 单 rank 推迟测试 |
| `test_disaggregation_wire.py` | 83 项通过 |
| `test_remote_main_contracts.py` | 21 项通过 |
| `test_remote_main_abi.py` | 6 项通过 |
| deterministic CPU padding 负向/正向对照 | 原基线失败，初始化占位值后通过 |
| 4060 P4 D2D CUDA/Mooncake TCP | 通过，预留/冻结/commit 恢复、state 字节一致、源释放一次、最终 Local |
| 静态检查 | Ruff、空白检查、50 个 hooks / 131 个 core interfaces 通过 |

477 项 CPU 回归通过。9 个改动 Python 文件在测试快照与工作树之间逐一核对
SHA-256，无差异。机器可读结果见
[h20-validation-followup-20261107.json](h20-validation-followup-20261107.json)。
原始日志和 CUDA JSON 归档在 `/private/tmp/h20-followup-results/results.tar.gz`。

未运行 H20 完整模型/跨机 RDMA，也未重测反馈中的 c=1/4/8 性能数据。
