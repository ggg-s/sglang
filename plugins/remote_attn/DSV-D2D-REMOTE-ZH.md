# DSV4.1 / Flash-0731 Remote D2D

本分支新增显式 D0→D1 后端迁移协议：D0 在完整 Decode 边界停算，D1 先接收
SWA、压缩器/indexer 状态和 indexer 历史，再接管 pending token。历史 global KV
后台复制期间，D1 继续用现有 sparse Remote Attention 读取 D0 的不可变前缀。
这不是等待全部 KV 复制完成后才恢复 Decode 的 copy-wait。

这是实验性后端控制接口。现有 P→D 最后 chunk 完整 forward 后的 Decode–KV
transfer overlap 不变；没有把最后 chunk 改回 layerwise forward。原有 Qwen3
迁移命令也不变。现有 `p4_gateway` 的旧 `drain_prepare/import_wave` 流程没有自动
切换到新 DSV D2D 协议，原生 HTTP/SSE 连接也不会自动改道；调用方须接管目标
请求的输出路由。不要把算子/协议验证当作整模、网关或 H20 RDMA 性能验收。

## 范围和前置条件

- D0/D1 都启用 `HANDOFF_PRE_LAYERWISE=remote`，模型、wire schema、TP 大小、
  page size、state layout 一致。两端使用不同 `HANDOFF_CTL_DIR`。
- 当前要求 Decode radix cache 关闭；P 的前缀缓存不受此限制。已有 P→D Remote
  窗口或前一次 D2D 尚未退休时，拒绝再次冻结该请求。
- 沿用现有纯文本、PP=DP=CP=1、无 DSpark/speculation、HiSparse、HiCache、LoRA
  的插件约束。当前不迁移 grammar、custom logit processor 或 logprob 请求。
- 保留原生 sampling 参数、完整 prompt/output token 历史及最后一个尚未写 KV 的
  sampled token。通过原生 `prepare_for_prebuilt/process_prebuilt` 和 FutureMap
  恢复 pending token，禁止迁移两端同时采样。
- 当前 bulk KV/state 使用独立连接上的最多 4 MiB TCP 帧；这是实际实现的传输
  方式，不是 native KV RDMA。Q/O/LSE 可通过 `HANDOFF_REMOTE_GPU_TRANSFER=1`
  使用现有原生 Mooncake GPU 通道。跨机地址沿用 `HANDOFF_PD_NETWORK=1` 和
  `HANDOFF_NETWORK_*`；控制文件通过现有受控通道送到各实例。

## 数据与 KV 所有权

1. `d2d_freeze` 在 scheduler 线程消耗一条尚未处理的原生 Decode overlap result，
   等待其 readback 并提交输出一次，然后移除请求。遇到 extend/prebuilt merge
   边界则返回错误，要求在下一原生 Decode 边界重试，不猜测其合并语义。
2. 冻结 D0 的 request handle/row、prefix length 和 FULL page map。SWA 只保留
   native live window；C2 pending half-pair、C4 page-scoped compressor/indexer
   state、C128 online row 或 raw pending-block ring 均按 native registration 的
   stride/index 寻址。整块边界没有 pending state 时不发送空洞行。
3. `d2d_import` 复用 native Decode prealloc，先检查 FULL/SWA/request-row 预算。
   同步复制 SWA、压缩状态和 indexer 字节，等待实际 GPU 完成后才允许提交。
   global KV 的后台线程此时可以尚未完成。
4. `d2d_commit` 先向每个对应源 rank 确认接管，再将 D1 请求接入原生 prebuilt
   Decode。source lock 将 commit 与 resume 串行化；commit 后拒绝恢复 D0。
5. 后台复制使用冻结 page map。immutable prefix 的最后一页只写
   `< floor(kv_len / ratio)` 的完整压缩行；D1 新完成的块、尾页 data/scale/padding
   不被覆盖。indexer 在 commit 前复制，Decode 开始后不再覆盖它。
6. 每个 global entry 的 copy event 只产生本地就绪事实；现有有序 TP retirement
   投票后才切 Local。最终退休还要求所有 Decode forward/sampling/readback 与
   RPC IO 完成。Remote 窗口沿用现有串行 Decode 策略，全部 Local 后恢复原生 overlap。
7. D1 发出 done 后，D0 tombstone 禁止新读。最后一个真实 CPU/GPU reader 退出，
   copy/commit 证明及各源 TP rank 的最终投票全部通过，才执行一次原生 KV free。
   D0 不能仅因 D1 已开始生成而释放历史 KV。

控制消息和 bulk KV 使用不同 TCP 连接，bulk 一次最多发出一个有界读取；不会
把全部历史 KV 先入队再发送 commit。实际带宽仍会影响迁移窗口，当前不承诺 TCP
bulk 能达到原生 RDMA 的速度。

## 控制命令

沿用现有 nonce + `cmd.0.json` → `result.<rank>.json` 文件控制协议。只写 rank 0
命令，由原生 request receiver 广播；等待所有 TP rank 返回，不要逐 rank 独立
发起变更。现有控制工具的 `send` 可以发送以下 payload。

```json
{"cmd":"d2d_freeze","rid":"source-request","migration_epoch":"move-001","destination":"D1","target_rid":"target-request"}
```

每个 D0 rank 返回自己的 `transfer`。将所有 rank 的 `transfer` 按 rank 排列后发给 D1：

```json
{"cmd":"d2d_import","migration_epoch":"move-001","destination":"D1","transfers":["此处放各 rank 的 transfer 对象"]}
```

TP1 可以用 `"transfer": {...}`。示例中的字符串必须替换为返回的对象，不要修改
manifest、route、token 或 epoch。一次 migration epoch 只能绑定一对请求，不能
用于重建已取消/失败/退休的分配。

```json
{"cmd":"d2d_commit","migration_epoch":"move-001"}
{"cmd":"d2d_status","migration_epoch":"move-001"}
```

commit 发 D1；status 可发两端。`target.admitted=true` 表示目标已接入 Decode，
`target.local=true` 表示退休完成；`source.released=true` 表示源端 read lease 退休。
`d2d_release` 只查询源端释放状态，不强制绕过任何 fence；成功迁移由 scheduler
自动回收源 KV。

提交前取消：在 D1 发 `d2d_abort`，遇到 `pending_reader=true` 则保持等待并重试，
不能认为取消已经完成。D1 worker 已退出、setup/copy stream 同步成功，且 TP 有序
退休后才释放其分配。再在 D0 发 `d2d_resume`，同样等待 `resumed=true`；源端必须先
撤销服务并排空实际 reader，才可恢复原 KV row。恢复不能释放 D0 的原始 KV。

提交结果不确定、commit 后目标失败、未知 GPU/transport completion 或 TP quarantine
均拒绝自动恢复 D0。quarantine 保留存储、callback 和注册，显式 abort/resume 不能
绕过它。原生 prealloc 中途异常可能留下未发布的 token map；此时保留 row，不按
`kv_committed_len` 猜测 free 范围。部分 free 已执行后报错也禁止重试，避免重复释放。
这些异常保留不是故障恢复能力，需要运维处理或重启实例。

## 验证

见 [4060 验证记录](validation/dsv-d2d-4060.md)。小池使用 source-loaded native
layout/state/metadata 与原生 prebuilt/checkpoint 代码；scheduler、allocator admission
和 native FlashMLA/compressor 的模型边界显式使用夹具。真实 GPU sparse kernel、
socket、registry reader、Mooncake GPU Q/O、copy event 和 TP2 Gloo 共识均实际执行。
尚需目标整模部署验证长序列输出一致性、原生流接续、H20 TP4 和 RDMA 吞吐。
