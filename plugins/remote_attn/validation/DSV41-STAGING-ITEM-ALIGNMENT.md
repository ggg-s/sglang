# DSV4.1 index staging 完整 item 对齐修复

2026-11-03，`codex/remote-attn-pre-layerwise`，与 Remote capture / Marlin workspace
修复一起交付；未另建分支。

## 原因及改动

ratio-1/2 index 池使用 64-token 物理页，PD 的一个 FULL-page item 则包含多个物理页。
`_init_compressed_pools` 创建低压缩率 index 池时遗漏 `global_page_size`，原生 allocator
因此只补一个物理 guard 页。这会让注册缓冲区以不足一个传输 item 的尾条结束，触发
`dsv41_staging.buffers` 的整除校验；该校验没有误报。

现在向原生 `_make_indexer_pool` 传入 `global_page_size=page_size`，与已有 C4 index
路径一致。保留包含 reserved FULL page 0 的原始 `size`，再由原生 allocator 分配完整
逻辑 guard 页。物理页格式、行 stride、PD item stride 和传输协议不变，注册仍覆盖完整 tensor。
staging 的整除、注册长度、连续性、重叠及逐块边界校验全部保留。

使用报告对应的 `full_size=619776`、FULL page 256、physical index page 64、每 token
68 字节，通过修复前后的实际原生分配方法得到：

| ratio | 修复前注册字节 | item stride | 修复前余数 | 修复后注册字节 | 修复后余数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 42,166,528 | 17,408 | 4,352 | 42,179,584 | 0 |
| 2 | 21,085,440 | 8,704 | 4,352 | 21,098,496 | 0 |

每个 index producer 增加 13,056 字节 guard 空间。`full_size` 仍由原生 planner 按
FULL page 向下取整；可分配的真实 token 数没有增加。没有截掉已注册内存或把尾条
当成完整 item 发送。

## 验证

插件 CPU 契约 221 项通过，含真实 TP2 Gloo。新增测试源加载原生
`_init_compressed_pools`、`_make_indexer_pool`、index `_create_buffer` 及注册方法，
覆盖 FULL page 128/256/512、ratio 1/2、报告的原始字节数、真实 native stager 绑定、
末个完整 item、跨界一字节拒绝。模型 KV 与 CUDA 设备属性使用 fixture；CPU 上
确实分配和检查 index tensor，没有以手工对齐后的假数组替代 allocator。
原生 compressed-pool CI 文件另补 V4/V41 完整构造测试。

RTX4060 / PyTorch 2.13.0+cu130 上运行真实 CUDA D2H 和 Mooncake TCP，两组均通过：

- `--native-index-allocation`：原生低压缩率 index 分配与注册，FULL page 256，所有
  注册长度除以各自 item stride 的余数为 0。
- 原有分离 payload/scale 布局回归；两组都发送每个 entry 的最后一个完整 item，
  源/目标逐字节一致，跨 entry 边界多读一字节被拒绝。
- 相邻 CUDA allocation 的基址允许解析为下一个有效 region，测试不会把它误判为
  上一个 region 越界。跨界访问仍从原 region 的内部起算并严格拒绝。
- 257 字节有界 pinned slot、非默认 copy stream、producer event、普通 middle
  传输复用、独立 state 直传及失败后 quarantine/禁止重试的检查保留。
- 更新后的 DSV V4 分段图回归：78 次 Remote / 8 次 full replay，最大绝对误差 0。

数据与源码 SHA256：
[原生 index](dsv41-staging-aligned-native-4060.json)、
[原有布局](dsv41-staging-aligned-legacy-4060.json)、
[分配前后审计](dsv41-staging-geometry-audit.json)。
分段图结果更新在 [DSV V4](h20-capture-fix-4060-dsv41-v4.json)。

```bash
python test/registered/unit/plugins/test_remote_attn.py
PYTHONPATH=plugins/remote_attn python test/manual/remote_attn/dsv41_host_staging_cuda_probe.py --sglang-root . --output /tmp/staging-native.json --native-index-allocation
PYTHONPATH=plugins/remote_attn python test/manual/remote_attn/dsv41_host_staging_cuda_probe.py --sglang-root . --output /tmp/staging-legacy.json
```

已修复本次报告的三处代码阻塞：Remote capture error mode、Marlin 捕获期间 workspace
分配、V4 index staging 尾条不对齐。H20 整模、FlashMLA/MXFP4 MoE、多卡 NCCL 和
c=16 服务性能仍需在目标机器验收；这些未运行的验证不等同于仍留有已知代码阻塞。
