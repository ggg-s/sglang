# DSV4.1 Flash：原生 PD 的 layerwise 接入

此路径只用于同机单 P/D pair、纯文本、相同 TP、PP=DP=CP=1、Mooncake、
DSpark/其他投机解码关闭，不启用 KV cache sharding。使用原生 SMG 和 `sglang.launch_server`，没有 JSON
包装器。本次只修改插件、预检、测试和文档，没有修改 SMG/SGL 核心。

## 传输与接纳

插件按原生 `DeepSeekV4TokenToKVPool.get_contiguous_buf_infos()` 注册的物理
条目构建资源清单。共享 global KV、indexer payload 和 scale 按实际生产者归组，
不假设每个模型层拥有一对 K/V，也不重新打包 FP4 数据。模型配置中的
`kv_source_layer_ids`、`index_source_layer_ids`、compression ratio、页长度和
状态布局必须在两端匹配；不同池容量和不同物理页地址可以使用。

- 原有 middle chunk 使用原生发送逻辑，chunk 大小和 transfer batch 上限不变。
- 只有最终 token chunk、bootstrap 已完成且所有 TP rank 同意时才接管其全局 KV
  页。原生页索引采用 full page ID；末尾不完整页仍只预留一次。
- P 运行原生模型 forward，在资源生产层返回后记录 CUDA event，排入原生 Mooncake
  FIFO。选中条目由原生 `_send_kvcache_generic` 发送；插件补充 DSV4.1 的精确布局校验，
  不经过核心中仅支持 MHA 的 selected-buffer 检查，不替换核心实现。
- 有些原生 bounded replay 会在层调用外写 global KV；没有观察到生产者 hook 的
  条目在整个 forward 结束后发送。保持原生 bounded-replay 数学行为。
- 最终仍调用原生 `send_kv_chunk(last_chunk=True)`，发送空 global KV 尾部、原生 SWA、
  压缩续接状态和 metadata。ratio-2 的奇数前缀携带 pending half-pair，偶数前缀
  沿用原生空 carry 索引。**只传 global KV 不代表 D 可以开始 decode。**
- D 继续等待原生 receiver Success。没有 early metadata、Remote attention 或
  提前接纳。原生 sender 完成、所有 TP rank 同意后，才执行 P 延迟释放。

一个模型层的完成标记只表示其 global 资源传输结束，不代表所有共享消费者和
原生状态都已就绪。失败不使用请求的 released 标志推断写入完成；保留资源并报错。
本次不提供跨故障自动恢复协议。

## 首版边界

- `model_type=deepseek_v41`，原生非 unified FlashMLA 池，compression ratio 0/1/2。
  原生 SWA paged/request-window 状态由原生 final-chunk 接口处理。
- 两端均 `--language-model-only --disable-radix-cache`；P 关闭 scheduler overlap。
  不启用 LoRA、HiCache、PD multiplexing、mixed chunk 或 whole-prefix checksum。
- 保留 D 原生 full graph/overlap。只有实际参与 layerwise 的 P forward 走 eager，
  该次 forward 暂时绕过内嵌 decoder tail graph；普通 P forward 不改图配置。
  P 不使用 torch.compile，不能让编译图跳过 Python 完成通知。
- `HANDOFF_PRE_LAYERWISE=remote` 对 DSV4.1 拒绝启动；Remote sparse attention、
  DSV4.1 Remote 分段图和 D0→D1 迁移不在本次支持范围。
- 不使用 native receive staging、host staging、uniform-FP8 TRTLLM 或 unified KV。
  不能照搬 Qwen 示例的 `--page-size 1`；沿用原生 DSV4.1 支持的页布局。

## 原生启动方式

先使用选定 checkpoint、GPU、TP 和 backend 跑通插件关闭时的原生 PD。
DSV4.1 的 off 对照应取消 `HANDOFF_CTL_DIR`（使插件入口不注册），其余原生参数与
layerwise 对照一致；不要使用带 Qwen 手动迁移环境的 off 模式。
`P_NATIVE_ARGS` / `D_NATIVE_ARGS` 表示这套已跑通命令的原生参数数组，包含各自的
model、TP、GPU、端口、disaggregation role、bootstrap、attention backend，以及线上
optimistic prefill、chunk 和 batch 参数。保留这些设置，删除 DSpark/speculative 参数；
保留支持 PD 的 `dsv4` attention backend，不选择当前核心拒绝 PD 的 `trtllm` backend。

两端安装同一版本插件并使用新的同机共享目录，例如：

```bash
python -m pip install -e ./plugins/remote_attn --no-deps
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check \
  --source-only --sglang-root .

# P、D 各自执行，保持已有其他插件项。
case ",${SGLANG_PLUGINS:-}," in
  *,remote_attn,*) ;;
  *) export SGLANG_PLUGINS="${SGLANG_PLUGINS:+${SGLANG_PLUGINS},}remote_attn" ;;
esac
export HANDOFF_PD_DIR=/workspace/remote-attn/dsv41-run-001/journal
export HANDOFF_PRE_LAYERWISE=layerwise
export HANDOFF_PRE_LAYERWISE_HOST_STAGING=0
export HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=0
```

P 终端（`P_NATIVE_ARGS` 沿用原生线上参数，不能带 speculative 参数）：

```bash
export HANDOFF_CTL_DIR=/workspace/remote-attn/dsv41-run-001/control/prefill
python -m sglang.launch_server "${P_NATIVE_ARGS[@]}" \
  --language-model-only --disable-radix-cache --disable-overlap-schedule
```

D 终端：

```bash
export HANDOFF_CTL_DIR=/workspace/remote-attn/dsv41-run-001/control/decode
python -m sglang.launch_server "${D_NATIVE_ARGS[@]}" \
  --language-model-only --disable-radix-cache
```

P/D 必须在 prefill 期间完成目的端分配；首版不使用
`disaggregation_decode_allocation_policy=prefill_complete`。这不修改
`optimistic_prefill_attempts`：尚未 bootstrap 完成、正在重试或不适合流水的请求
仍走原生路径。默认自动分组对不足 3072 个 final-chunk token 的 batch 回退原生；
功能验证可在 P 设置 `HANDOFF_PRE_LAYERWISE_GROUP_SIZE=1` 强制按生产者分组。

SMG 使用原有 PD router 命令，不加载插件环境。先通过原生 `/health`、`/server_info`
确认 P/D 配置；这个 layerwise 路径不启动 RemoteHub，不以 `ready.<rank>.json`
作为路径证明。

## 路径确认与本地证据

一次实际参与适配的请求应有：

1. P 日志 `[dsv41-layerwise] batch=... requests=... group_size=... final_chunk_tokens=...`。
2. 共享 journal 下 `bootstrap_room/rank/offer.json` 包含 `dsv41` 资源清单。
3. generation 子目录出现 `resource-<producer>.json`，其 `buffer_indices` 是真实物理
   条目，`decode_ready=false`；这些不是 Remote-ready 证明。
4. `prefill.json` 包含 `model_type=deepseek_v41`、`native_final_states=true`；
   `released.json` 在原生 sender 完成及 TP 一致退休后出现。D 的 `done.json` 中
   `early_admission=false`，没有 `admitted.json` 或 Remote 调用。

不满足 final chunk、peer manifest、同 TP、单目的端或 bootstrap 条件时，接管前
回退原生，不推进原生发送游标。执行中发生错误则保留资源并报告，不能在写入中途回退。

本地 CPU 测试直接执行原生池的 wire layout、压缩 carry 索引、Scheduler chunk
状态规划、CommonKVSender 计数和 Mooncake byte/page planner，只替代 CUDA event、
传输 I/O 和服务进程边界。覆盖乱序物理页、FP4 payload/scale、完整 middle chunk +
非整页 final chunk、奇偶 carry、GPU event 未完成不得写入/释放、布局不匹配和
TP 不同意时游标不变、P eager / D native graph、错误时保留源分配。

新增 7 项专项测试，完整 105 项本地测试通过，包含真实双进程 Gloo；
源码预检覆盖 30 个 hook、76 个核心接口。既有 TP2 合并退休测试补齐 READY 通知的
确定性边界，避免合法的分批到达使合并断言偶发失败，未修改生产退休协议。

仅完成本地验证，未在 GPU 上加载模型、测数值一致性或测收益。GPU 验收需对
插件 off/layerwise 使用同一 revision 和原生 optimistic/chunk 配置，比较 token/
logprob、TTFT、P 完成到 D 接管间隙、吞吐和资源回收；再覆盖 bootstrap 延迟、
chunk 边界、取消、传输失败及 retraction。不得把 CPU 字节测试视为 CUDA 正确性验收。
