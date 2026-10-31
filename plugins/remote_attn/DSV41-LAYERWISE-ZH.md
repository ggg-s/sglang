# DSV4.1 Flash：原生 PD 的 layerwise 与 eager Remote 接入

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
- layerwise 下 D 继续等待原生 receiver Success。Remote 模式则在 indexer、SWA/carry、
  metadata 屏障后提前接纳，global 未完成时可 decode；详见专用 Remote 指南。
  P 还须等待远端 reader 完成及 TP 退休共识，才执行延迟释放。

一个模型层的完成标记只表示其 global 资源传输结束，不代表所有共享消费者和
原生状态都已就绪。失败不使用请求的 released 标志推断写入完成；保留资源并报错。
本次不提供跨故障自动恢复协议。

`HANDOFF_PRE_LAYERWISE=remote` 支持提前接纳、逐资源 Local 和 batch 1/2/4 分段 graph；
归属、启动和限制见 [Remote 接入指南](DSV41-REMOTE-EAGER-ZH.md)。

## P 端 host staging

P 设置 `HANDOFF_PRE_LAYERWISE_HOST_STAGING=1`，D 的该变量保持 0。
D 可独立启用下节的 receive host staging。插件在启动时按原生注册顺序绑定
compressed global KV、indexer FP4 payload/scale，逐项验证地址、连续性、设备、
总字节数和原生 page stride，然后预热原生 rank-owned `CorePinnedStager`。
复用 Qwen 的 bounded pinned slot、非阻塞 D2H stream、Mooncake 同步发送和失败隔离；
不 gather、不解量化、不重打包，也不按模型层分配 pinned buffer。

最后一个 chunk 的分组任务通过原生 `transfer_fn=partial(_transfer_data, stage=True)`
发送。原生 worker 先等生产层 event 和页索引 event，再执行 D2H；同一 slot 的锁覆盖
D2H 到 TE 返回，成功后才执行原有完成回调。SWA、carry 和 metadata 保留原生
final-chunk 顺序及 Success 屏障；layerwise 下 D 不提前 decode，Remote 使用专用提前状态屏障。`prefill.json.host_staging=true`
用于确认本次接管启用了该路径。

普通 middle chunk 保持已有原生策略。若原生线上已设置
`SGLANG_MOONCAKE_HOST_STAGING_ALL=1`，它们也复用同一 slot；没有该设置时普通
middle 仍直传，不擅自改成全量 host staging。chunk 大小和 transfer batch 上限不变，
optimistic prefill 不变，P FULL/SWA 前缀读租约与 overlap plan/接管证书照常使用。
P staging 启用失败直接拒绝启动，不静默退回直传。

传输失败会 quarantine 原生 pinned slot，完成回调不会提前执行。overlap 的本地
退休观察在检查 outstanding IO 前检测 poisoned slot，经已有有序切换命令触发最终
TP 共识；所有 rank 保留 KV/cache 读租约，不重用或释放失败 slot。稳定 chunk
不新增 TP collective；没有逐 chunk staging 投票。串行路径沿用已有 Source 失败保护，
最终释放仍要求 GPU、TE 和原生 sender Success 均完成。

4060 的真实 CUDA + Mooncake TCP 验证、缓存/overlap 回归和本地 TP2 结果见
[host staging 验证记录](validation/dsv41-p-host-staging-4060.md)。这些小型测试尚不代表
完整模型精度、生产吞吐或尾延迟收益。

## D 端 receive host staging

D 设置 `HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=1`；P 的此变量为 0，P send
staging 可独立开关。两端保持 `SGLANG_DISAGG_STAGING_BUFFER=0`：原生这个开关
面向成对 K/V，会在 MLA 启动时拒绝。插件在 native MLA manager 初始化后补绑定
原生 tensor 列表，在 D 创建同款 pinned `StagingAllocator`、GPU shadow 和
`DecodeStagingHandler`，再打开相应实例的原生完成门控；不临时改全局环境变量。

offer 的 `staging_wire=dsv41-flat-v2`、精确 DSV4.1 manifest、同 TP 和原生 ring
注册必须匹配。P 在最后一个 chunk 的 forward 前，用原生 `STAGING_REQ/RSP`
为各组申请区间，payload 按选中的物理条目及各自 native stride 紧凑排列。
分组写入保留原生 TE planner 和 batch 上限；没有把 indexer scale 误当作 V，
也不重新打包 FP4。普通 middle/cached-prefix chunk 保持原生同 TP 直传，
SWA/carry/metadata 继续原生 final-chunk 状态传输。

D 等原生 `CHUNK_READY` 后，在 scatter stream 上等待 req-to-token row event，
执行 pinned H2D，再按 FULL page ID 和各条目的 native stride 做字节 scatter。
原生 event 记录在全部写入后；只有 event 完成才能释放 ring allocation、推进
watermark。奇数尾页按 `ceil(prompt/page_size)` 校验，仍携带原生 carry。
receiver 网络 Success 不代表可 decode：原生 polling 还会门控 H2D/scatter 与
metadata，随后原有 TP MIN 一致提交。没有提前接纳，没有额外逐 token 同步；
D 全部本地后的 graph/overlap 保持原生路径。

取消/失败沿用原生 deferred abort ACK；P 写入安全证明完成后，native room release
先 drain scatter stream，再回收已提交和未提交的 ring 区间，最后释放 D KV。
GPU drain 失败则保留 allocation；manager teardown 在 active room/区间或 worker
尚未退出时拒绝回收。engine 强引用 ring/KV，直到正常 drain 和显式 deregister 均完成。
分配超限不回退直写等待 staging 的 D；P 经原有有序退休流程上报 native PD 失败。

ring 容量沿用 `SGLANG_DISAGG_STAGING_POOL_SIZE_MB`，需要同等大小的 pinned host
和 GPU shadow；单个分组必须能放入 ring。确认路径可检查 D 启动日志
`[dsv41-receive] pinned native ring enabled`、offer 中上述 wire 标记，以及本次
`prefill.json` 的 `recv_host_staging=true` / `native_staging=true`。这些标记不替代
H2D event 与原生 receiver/metadata 的完成证明。

详见 [D receive staging 验证记录](validation/dsv41-d-receive-staging-4060.md)。

## 首版边界

- `model_type=deepseek_v41`，原生非 unified FlashMLA 池，compression ratio 0/1/2。
  原生 SWA paged/request-window 状态由原生 final-chunk 接口处理。
- 两端均 `--language-model-only`，D 保留 `--disable-radix-cache`；P 支持原生
  UnifiedRadixCache 的设备内 FULL/SWA 前缀缓存和 scheduler overlap。
  不启用 LoRA、HiCache、PD multiplexing、mixed chunk 或 whole-prefix checksum。
- 保留 D 原生 full graph/overlap。只有实际参与 layerwise 的 P forward 走 eager，
  overlap 模式通过本次 launch 的 plan 绕过内嵌 decoder tail graph，不临时改写
  模型的 tail 对象；普通 P forward 不改图配置。关闭 P overlap 的对照仍保留原串行路径。
  P 不使用 torch.compile，不能让编译图跳过 Python 完成通知。
- `HANDOFF_PRE_LAYERWISE=remote` 支持 [eager sparse 提前接纳与逐资源 Local](DSV41-REMOTE-EAGER-ZH.md)：
  indexer、SWA/carry 和 metadata 先就绪，global KV 未完成时开始 decode。
  D Remote 串行，全部 Local 且 GPU/TP 退休后恢复原生 graph/overlap。
  D Remote batch 1/2/4 支持分段 graph，动态 prepare/sparse/RPC/merge 留图外；D0→D1 迁移尚未适配。
- 支持 P send 和 D receive host staging；不使用原生 MHA staging 开关、uniform-FP8 TRTLLM 或 unified KV。
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
# DSV4.1 使用插件 flat-entry 适配，原生此开关会拒绝 MLA。
export SGLANG_DISAGG_STAGING_BUFFER=0
```

现场 Linux x86_64 镜像的 A/B 已确认：直接启动时必须优先加载上述系统
`libstdc++.so.6`。当 venv/conda 的旧 libstdc++ 与当前 torch/CUDA 绑定构建的 ABI
不匹配时，可能导致 `cudaHostRegister` 失败或错误绑定异常；仓库 `deploy/common.sh`
已有 preload，裸 `launch_server` 不会自动继承。这里将系统库置于已有 preload
之前；其他镜像按实际兼容库路径配置，`LD_LIBRARY_PATH` 保留已验证的 CUDA 配置。
这属于进程动态库环境适配，不改变 optimistic prefill、chunk、SEND_LOCK 或 FAST 参数。

P 终端（`P_NATIVE_ARGS` 沿用原生线上参数，删除其中的 speculative 参数和
`--disable-overlap-schedule`；启用 P 缓存时也删除 `--disable-radix-cache`，不新增插件环境变量）：

```bash
export HANDOFF_CTL_DIR=/workspace/remote-attn/dsv41-run-001/control/prefill
export HANDOFF_PRE_LAYERWISE_HOST_STAGING=1
python -m sglang.launch_server "${P_NATIVE_ARGS[@]}" \
  --language-model-only
```

D 终端：

```bash
export HANDOFF_CTL_DIR=/workspace/remote-attn/dsv41-run-001/control/decode
export HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=1
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
  空闲或状态未变化时没有新增退休 collective；接管投票也改为下面的 TP 证书，
  原生 continuous-input-polling 自身的 collective 保留。
- 取消、optimistic retry、retraction 和 reset 在原生 abort/备份/释放/换 attempt 前
  进入同一有序 drain，先消费先前 intent，再等每个 rank 的 GPU/IO 证明。取消时
  forward event 已覆盖所有源 KV 读者，延迟采样仅读 logits，无需强行提前采样，
  也不等待其尚未录制的 copy event；结果对象仍交原生队列处理。
  CUDA/传输异常、未知完成或 drain 超时均保留分配并报错，不自动复用页或故障回滚。

SMG、SGL scheduler loop、future map、采样和原生结果队列没有修改；D 的原生
overlap/full graph 路径保持不变。P 可启用下面的前缀缓存；DSpark 仍关闭。

## P 前缀缓存

P 可删除原生 `--disable-radix-cache`，无需新增插件变量。仅支持原生
`UnifiedRadixCache` 的设备内 FULL 或 FULL+SWA 组件；session refs、Mamba/C128
等其他组件、HiCache、linker 和 unified KV 仍不开放。D 缓存保持关闭。

- FULL 页同时覆盖对应的压缩 global KV、indexer payload/scale 的物理页归属；
  原生树锁一起锁住 FULL/SWA 组件，SWA 锁回执在节点 split 后仍按原生 UUID 边界释放。
- overlap 在首次接管投票前获取独立缓存读租约，通过同一证书保存给请求的
  SourceLease。后续 middle→final 不重锁、不重投票；关闭 overlap 的原串行
  layerwise 路径也在 final 接管时使用同一缓存 guard。
- 读租约期间延迟该请求的 checkpoint/dedup、finish 和释放，避免正在发送的
  后缀被原生去重换页。已有前缀独立 pin，其他请求仍可命中它，不能将其淘汰。
- 不是把 SWA/carry 当成新缓存格式：SWA prefix match/re-prefill、request-window
  和压缩续接状态沿用原生 SGL；原生 final-chunk 仍负责将完整状态交给 D。
- 若首次接管前已有未完成的 native 发送，先缓存否定证书，等其 IO 账本清空
  触发资格边沿，再有序重确认，避免跨越未知的旧物理页引用历史。
- 正常完成须等原生 release 回调、GPU、IO 和 TP 共识；之后一次执行延迟的
  checkpoint/finish、原生释放及独立读 pin 释放。取消/retraction 先 drain，再
  交回原生状态变更；未知完成、失败回调继续保留资源。

读租约会延迟新增后缀的缓存发布/去重，可能增加瞬时占用，需在目标负载验证
显存及命中率。它不额外复制整个 KV，也不改变 optimistic/chunk/batch 参数。
路径证明：先用同前缀请求预热，再确认 `prefill.json` 中
`p_radix_read_lease=true`、`cached_prefix_tokens>0`，最终 `released.json` 的
`p_radix_lease_closed=true`。开关对照应保持 chunk、TP、graph、overlap 等参数一致。
验证记录见 [P 缓存验证](validation/dsv41-p-cache-4060.md)。

## TP 接管证书与资格变化

P overlap 不再为每个 chunk batch 重新执行接管 all_reduce。每个请求 attempt 的
证书只由 TP 一致的事件更新，和保护实际 GPU/IO 的 SourceLease 分开：

- 首次检查：按原生共享 batch 的新请求顺序合并一次 CPU all_reduce。接受与否
  都缓存。证书身份包含 rid、room、cache attempt、retraction count 和 TP 一致的
  birth ordinal，防止复用 rid/room 时旧通知误命中新请求。
- 稳定 middle→final：复用同一证书和已验证 manifest/peer 快照，继续建立每批
  独立 plan；没有逐 chunk 接管投票，也不逐 chunk 重建完整 manifest。
- 尚未 bootstrap 的否定结果不会逐 chunk 重投。bootstrap/offer/注册状态变化
  产生本地通知；rank 0 收齐 ready 后经原生请求广播发有序重确认命令。每 rank
  都按命令列出的固定顺序重新验证并投票，不能因本地不合格而跳过 collective。
- 通知有 epoch、资格版本和 revision；丢失时重传至收到原生广播 ACK，ACK 自身
  不产生新的 collective。相同资格版本/状态不反复投票。
- 注册表在服务开始前装上插件 guard。已接管期间固定 room 的 TransferInfo 和
  session 注册；新的 peer 注册及 room 更新先延迟，发送 worker 继续读取旧视图。
  新证书不能越过待应用的冲突注册更新。
- offer generation、session/布局、bootstrap 等资格失效时，任何 rank 都只发
  撤销通知，不独自切原生/graph。所有 rank 收到同一撤销命令后先投票，再完成
  已有 GPU/IO 的 TP drain，使用旧注册通知原生 abort，最后应用延迟注册。
  已经发送部分 KV 的 attempt 会终止，不能拿剩余 KV 拼到新的 destination。
- 接管和退休共用插件调度入口，最多一条有 collective 的异步命令在途。原生
  取消、retry/retraction/reset 先消费已有命令再 drain；没有租约的否定请求不会
  意外 drain 其他请求。资格版本不一致时全部 rank 共同拒绝，保留资源。

| 边界 | 新增接管同步 |
|---|---|
| 新 attempt 首次检查 | 一次，可合并同批多个新请求 |
| 稳定 middle→final / 未变化的否定证书 | 0 |
| 有序重确认或撤销 | 每条命令一次；撤销另有既有 TP drain 共识 |
| 空闲或等待 GPU/IO | 0 |

仍有本地状态检查、event query、入队账本和通知处理；这里的 0 指新增 TP 接管
collective，不代表零 CPU 开销，也不删除模型计算/原生 scheduler 的 collective。
SMG/SGL 核心、原生采样、future map 和 optimistic/chunk/batch 配置未修改。

## 路径确认与本地证据

启用 P overlap 时，先确认 P 启动日志含 `[dsv41-p-overlap] enabled` 和
`[dsv41-p-admission] enabled`。
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

当前共有 38 项 DSV4.1 专项测试，完整 136 项本地插件测试通过，包含真实双进程
Gloo；源码预检覆盖 40 个 hook、88 个核心接口。新增覆盖两批 final 交错、同请求
middle→final、队列任务继承、延迟采样、失败保留、重试/retraction/reset 栅栏、长
prompt 短尾块策略。TP2 用例验证不同 rank 的 generation 和 GPU 就绪时序，
最终正常/取消各一次共识，稳定轮询不产生额外退休投票。新增证书 TP2 用例
验证连续 63 个稳定 chunk 接管投票为 0，bootstrap 重确认、通知丢失重传、注册
撤销/退休/取消交错及单 rank 资格版本错误均保持一致的 collective 调用序列。

RTX4060 小型真实 CUDA 验证及边界见 [证书验证记录](validation/dsv41-p-certificates-4060.md)。
未加载 DSV4.1 模型，尚未测 token/logprob 一致性或收益。模型 GPU 验收需对
插件 off/layerwise 使用同一 revision 和原生 optimistic/chunk 配置，比较 token/
logprob、TTFT、P 完成到 D 接管间隙、吞吐和资源回收；再覆盖 bootstrap 延迟、
chunk 边界、取消、传输失败及 retraction。小型 CUDA/CPU 测试不能替代模型验收。
