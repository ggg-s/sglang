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
- 只有最终 token chunk、bootstrap 已完成且所有 TP rank 同意时才分层接管其全局 KV
  页。P overlap 模式在合格 middle chunk 就建立请求租约，提前按原生整页入队，
  仍保留原生 chunk 大小；result handler 不重复发送已入队区间。原生页索引采用
  full page ID；末尾不完整页仍只预留一次。
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
- 两端均 `--language-model-only --disable-radix-cache`；P 支持原生 scheduler overlap。
  不启用 LoRA、HiCache、PD multiplexing、mixed chunk 或 whole-prefix checksum。
- 保留 D 原生 full graph/overlap。只有实际参与 layerwise 的 P forward 走 eager，
  overlap 模式通过本次 launch 的 plan 绕过内嵌 decoder tail graph，不临时改写
  模型的 tail 对象；普通 P forward 不改图配置。关闭 P overlap 的对照仍保留原串行路径。
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
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
case ",${SGLANG_PLUGINS:-}," in
  *,remote_attn,*) ;;
  *) export SGLANG_PLUGINS="${SGLANG_PLUGINS:+${SGLANG_PLUGINS},}remote_attn" ;;
esac
export HANDOFF_PD_DIR=/workspace/remote-attn/dsv41-run-001/journal
export HANDOFF_PRE_LAYERWISE=layerwise
export HANDOFF_PRE_LAYERWISE_HOST_STAGING=0
export HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=0
```

现场 Linux x86_64 镜像的 A/B 已确认：直接启动时必须优先加载上述系统
`libstdc++.so.6`。当 venv/conda 的旧 libstdc++ 与当前 torch/CUDA 绑定构建的 ABI
不匹配时，可能导致 `cudaHostRegister` 失败或错误绑定异常；仓库 `deploy/common.sh`
已有 preload，裸 `launch_server` 不会自动继承。这里将系统库置于已有 preload
之前；其他镜像按实际兼容库路径配置，`LD_LIBRARY_PATH` 保留已验证的 CUDA 配置。
这属于进程动态库环境适配，不改变 optimistic prefill、chunk、SEND_LOCK 或 FAST 参数。

P 终端（`P_NATIVE_ARGS` 沿用原生线上参数，删除其中的 speculative 参数和
`--disable-overlap-schedule`；不新增插件环境变量）：

```bash
export HANDOFF_CTL_DIR=/workspace/remote-attn/dsv41-run-001/control/prefill
python -m sglang.launch_server "${P_NATIVE_ARGS[@]}" \
  --language-model-only --disable-radix-cache
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
仍走原生路径。P overlap 的自动分组用实际 final-chunk token 总数与被接管请求最长
prompt 长度的较大值判断 3072 阈值，因此 32K prompt + 很短尾块仍能接上 layerwise；
组数仍参考实际 chunk 计算量。短 prompt 默认走原生，不建立额外 GPU/回收 plan。
关闭 P overlap 的对照沿用仅按 final-chunk token 判断的旧策略；
功能验证可在 P 设置 `HANDOFF_PRE_LAYERWISE_GROUP_SIZE=1` 强制按生产者分组。

SMG 使用原有 PD router 命令，不加载插件环境。先通过原生 `/health`、`/server_info`
确认 P/D 配置；这个 layerwise 路径不启动 RemoteHub，不以 `ready.<rank>.json`
作为路径证明。

## P overlap 的生命周期

- 每次 launch 独占 plan，保存 request/sender/cache-attempt/row、chunk 区间、资源组、
  forward event 和原生 GenerationBatchResult。通过 ContextVar 选择本次 producer hook，
  result handler 通过结果上的 plan 找回对应发送区间；不依赖共享 `controller.plan`。
- 同请求的多个 middle→final plan 共用请求租约。middle 在 forward event 后按原生
  sender FIFO 提前入队；final 仅预留尚未发送的页，仍由原生 final 接口发送 SWA、
  carry 和 metadata。`inflight_middle_chunks > 0` 不再单独导致长请求回退。
- 页索引在原生 forward stream 等待 schedule stream 后做 pinned D2H，索引 event
  随原生 `indices_event` 传给 worker；GPU/host 索引的引用保留至 IO 完成。
- 正常完成要同时满足：所有相关 launch 的 forward event、原生已提交采样/回拷的
  `copy_done`、完整入队账本清空及 native sender Success。未录制的 copy event 不算证明。
  入队账本覆盖接管前已有的原生 prefix/chunk，每个成功 on_complete 才消账，
  不能用 Failed 状态或仅记录在执行任务的计数器证明结束。
- GPU/IO 线程只更新本地事件/账本；scheduler 发就绪状态边沿通知。rank 0 经原生
  request broadcast 注入有序命令，所有 TP rank 才进入同形状最终 all_reduce，并一起释放。
  空闲或状态未变化时没有新增退休 collective；启动校验、批次接纳投票和原生
  continuous-input-polling 自身的 collective 保留。
- 取消、optimistic retry、retraction 和 reset 在原生 abort/备份/释放/换 attempt 前
  进入同一有序 drain，先消费先前 intent，再等每个 rank 的 GPU/IO 证明。取消时
  forward event 已覆盖所有源 KV 读者，延迟采样仅读 logits，无需强行提前采样，
  也不等待其尚未录制的 copy event；结果对象仍交原生队列处理。
  CUDA/传输异常、未知完成或 drain 超时均保留分配并报错，不自动复用页或故障回滚。

SMG、SGL scheduler loop、future map、采样和原生结果队列没有修改；D 的原生
overlap/full graph 路径保持不变。P radix cache 和 DSpark 仍关闭。

## 路径确认与本地证据

启用 P overlap 时，先确认 P 启动日志含 `[dsv41-p-overlap] enabled`。
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

当前共有 14 项 DSV4.1 专项测试，完整 112 项本地插件测试通过，包含真实双进程
Gloo；源码预检覆盖 34 个 hook、80 个核心接口。新增覆盖两批 final 交错、同请求
middle→final、队列任务继承、延迟采样、失败保留、重试/retraction/reset 栅栏、长
prompt 短尾块策略。TP2 用例验证不同 rank 的 generation 和 GPU 就绪时序，
最终正常/取消各一次共识，稳定轮询不产生额外退休投票。

RTX4060 小型真实 CUDA 验证及边界见 [验证记录](validation/dsv41-p-overlap-4060.md)。
未加载 DSV4.1 模型，尚未测 token/logprob 一致性或收益。模型 GPU 验收需对
插件 off/layerwise 使用同一 revision 和原生 optimistic/chunk 配置，比较 token/
logprob、TTFT、P 完成到 D 接管间隙、吞吐和资源回收；再覆盖 bootstrap 延迟、
chunk 边界、取消、传输失败及 retraction。小型 CUDA/CPU 测试不能替代模型验收。
