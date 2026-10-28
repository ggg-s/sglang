# 5090 内网复验与 KV 重复释放修复（2026-10-28）

本页记录用户反馈的内网实验；未获取原始日志和 token 文件，不表述为本机独立复跑。
基于 `6c445b5`（含 `cc7fa77`、`0223be4`），现场另有 KV release 幂等保护及
retraction-backup 配置改动，均尚未提交。以下真机结果包含这两项现场改动。
本次仓库同步版本增加了 pending-prefix 和独立 Mamba 状态的保护，不能视为与现场补丁逐字节相同。
历史报告见 [首轮 5090 反馈](5090-intranet.md)。

## 环境

RTX 5090 ×8，使用 6 卡；Qwen3-8B；P / D0 / D1 各 TP2；Mooncake TCP；
radix cache、CUDA Graph 关闭。端口为 31016 / 31017 / 31018，gateway 31080。

| 组件 | 本轮内网版本 |
| --- | --- |
| torch | `2.14.1+cu130` |
| torchvision | `0.29.0+cu130` |
| sglang-kernel | `0.4.10` |
| FlashInfer | `0.7.2` |
| Mooncake | `mooncake-transfer-engine-cuda13==0.3.13.post1` |
| kernel 版本检查 | 已移除 `SKIP_SGL_KERNEL_VERSION_CHECK` |

现场观察：kernel 0.4.9 / 0.4.10 搭配 torch 2.13 出现
`undefined symbol: CUDAErrorLogCapture`，升级 torch 2.14 并配套 torchvision 0.29 后解决。
这是该 CUDA 13 wheel 组合的验证记录，不是所有平台和构建版本的兼容性声明。

## 性能与 token 等价性

32K / C4 / new=128，时间单位 ms，ITL 为 ms/token。

| 模式 | TTFT p50 | E2E p50 | Gap p90 | ITL | 与旧栈 token |
| --- | ---: | ---: | ---: | ---: | --- |
| Natural | 3,921 | 7,616 | 84 | 29.1 | 4/4 一致 |
| Copy | 3,939 | 7,766 | 1,423 | 30.1 | 4/4 一致 |
| Layer | 3,910 | 7,855 | 1,022 | 31.1 | 4/4 一致 |
| Remote | 3,932 | 8,091 | 218 | 32.8 | 4/4 一致 |
| 旧栈 Remote | 3,956 | 6,999 | 385 | 24.0 | — |

Remote Gap 从 385 降至 218 ms，约降低 43%；E2E 仍高于旧栈，也高于本轮 Natural。
升级依赖未消除 Natural 的 decode 慢速，支持继续调查新 main / attention backend 基线；
现有数据不足以单独证明 FlashInfer 是根因，也不能推断 Remote 的额外开销为零。
后续 Prefill layerwise transfer + Remote Attention 实验应保留本轮 Natural 和 Remote 对照，
同时量测交接等待、Remote Window、ITL 与 E2E，不能仅以 Gap 改善判断整体性能。

## C1、故障及 TP2 稳定性

C1：4K / 96 token / cut=32，每个模式使用全新的 P+D。

| 模式 | drain 终态 | token | owners | transition gap |
| --- | --- | --- | --- | ---: |
| Copy | SAFE_TO_STOP | 96/96 exact | src+dst | 378 ms |
| Layer | SAFE_TO_STOP | 96/96 exact | src+dst | 181 ms |
| Remote | SAFE_TO_STOP | 96/96 exact | src+dst | 208 ms |

此前三模式均 ROLLED_BACK；本轮确认 `6c445b5` 的 reservation epoch 修复在该配置生效。

| 故障 | 内网报告结果 |
| --- | --- |
| Import Fail | 3.51 s；source resume；96/96 exact |
| LOCAL Drop | 2.85 s；migrated；96/96 exact |
| Target Kill | 19.05 s；ROLLED_BACK；96/96 exact |
| Controller Kill @ Freeze | source 96/96，target 0；registry 0/0/0 |
| Controller Kill @ Commit | target 96/96，source 0 且不 resume；两侧清理成功 |

新提交与现场修复下 TP2 稳定性 **23/23**：rank1 8、rank0 5、any-rank 4、
import in-flight 4、rank1 补充 2。现场 CPU 单测 **36/36**，服务 P/D0/D1 health 均 200。
这 23 轮属于本轮插件复验，与首轮报告中的旧 P5.18 23 轮和插件抽样 3 轮分别记录。

## 本次同步修复

失败请求先被插件 cleanup 释放并 `mark_kv_released()`；decode deferred release 再进入原生
`release_kv_cache`，触发 `Only a mamba-capable tree cache allows freeing before alloc`，
导致 D scheduler 崩溃。

`hooks.py::_release_kv` 现在按顺序执行：

1. 停止并 fence target receiver，确保旧写入不再触及将复用的 KV。
2. 若尚未 admission，先释放独立的 `handoff_prefix_indices`。
3. KV 已释放、没有 row 且没有独立 Mamba state 时，重复释放直接返回。
4. 其余情况继续原生释放流程。

不能把 `is_kv_released` 判断放在函数最前面：尚未 admission 的新 `ReqKvInfo` 已满足该属性，
但 pending prefix 仍占用 allocator slots；Mamba state 也可能早于 KV 分配。

部署配置新增 `HANDOFF_RETRACTION_BACKUP=none`，D0/D1 显式传入
`--disaggregation-decode-retraction-backup "$HANDOFF_RETRACTION_BACKUP"`，P 不传该参数。
这对齐内网验证配置，不替代 KV 幂等修复。`none` 表示原生 decode 因显存压力 retraction 时
不保存 KV backup，返回 503；不是透明恢复。可显式覆盖为 `cpu_tensor` 或 `host_pool`，
但这些配置不属于本轮新增真机验证范围。handoff 自己的注册 host 传输缓冲区不受该开关控制。

本机回归：修复前从真实原生 `release_kv_cache` 复现相同断言（source cleanup / pending import
两条路径）；修复后 **38/38 CPU 用例通过**，包含真实 Gloo TP2 广播。
新增验证覆盖重复释放不影响已复用 slots、pending prefix 正确回收、独立 Mamba state 正常释放；
启动脚本测试覆盖 D0/D1 默认值及显式 override、P 不带 decode-only 参数。
本机 CPU harness 隔离了未安装的模型服务依赖，未在本轮重新启动 5090 模型服务。

## 遗留：单独重启 D 的 Mooncake session 窗口

内网观察：只重启 D0 后立即发请求，可能收到
`KVTransferError ... remote mooncake session is not alive` / HTTP 500；随后日志出现
`Session recovered via probe; un-blacklisted`。

代码核查：原生 Mooncake sender 遇到 `failed_sessions` 先将该请求标为失败，后台探针成功后
解除 session 黑名单；恢复 session 不会自动重放已经失败的请求。
插件已有的 peer 注册前 probe 不构成重启后首请求必成功的保证。
仅凭当前报告不能确定全部首次过期触发条件，本次不修改 session 协议、不标为已修复。
当前使用内网已验证的规避方式：维护时 D 与 P 配套重启，完成实际请求验证后恢复流量；
仅 health 200 不足以证明 KV 传输 session 已恢复。

## 内网证据位置

以下为用户提供的内网路径，本仓库没有复制原始数据：

- `pd-deploy/migration/p0-p5.18/VALIDATION-REMOTE-ATTN-2026-10-28.md` 第 8 节。
- `pd-deploy/migration/p0-p5.18/validation-2026-10-28/new-commits/`。
