# P 缓存前缀统一接入 Remote progressive

Qwen3-8B/MHA 与 DSV4.1 共用 `prefix_stream`：在原生 cached-prefix 首次发送前，
把本次请求的缓存前缀、middle 与 final 区间纳入同一份计划。数据仍通过原生
Mooncake PD 地址规划、worker、staging ring / watermark / scatter 传输；SMG 不变。
原生 prefill/chunk 调度循环、模型计算、graph runner 与 Mooncake C++ 没有复制或修改。

## 启用与对照

沿用现有原生 P/D/SMG 命令。P 开启已支持的前缀缓存、
`SGLANG_PLUGINS=remote_attn`、`HANDOFF_PRE_LAYERWISE=remote` 后，满足资格的设备内
缓存命中默认接入；不新增必填变量。保留 optimistic prefill、chunk、传输 Batch、
graph 及已有 overlap 配置。Qwen P 的 overlap 范围仍沿用原有限制，D Remote 串行、
Local 原生 overlap；DSV 保留既有 P overlap 接管证书。

仅在 **P 启动环境**设置以下变量并重启，可做旧 Remote 路径对照：

```bash
export HANDOFF_PREFIX_REMOTE_PROGRESSIVE=0
```

不设置或设为 `1` 使用新路径。`0` 保留原生 cached-prefix 发送和旧 final-chunk
Remote，不是关闭整个插件。完全原生 PD 基线在 P/D 设置 `SGLANG_PLUGINS=__none__`，
使用原生 staging 配置启动新进程；不卸载插件，不使用 JSON 包装器。其他必要通用
插件存在时，应使用不包含 remote_attn 的白名单。

新路径要求 bootstrap 已完成、P 有设备缓存、尚未发送该请求的第一个区间、同 TP
且 peer 宣告 `range_stream=1`。文本/布局/页大小/缓存后端限制沿用各模型已有范围。
缓存未命中、能力未就绪或资格不一致，在任何发送接管前保留原有路径。DSV P overlap
必须已经取得该请求的接管租约；不跨未知旧 IO 追认新资格。激活后的 attempt、row、
sender、peer 注册发生变化则保留页并报错，不能中途切回普通发送造成重发或空洞。

## 数据与控制顺序

以32K输入、P命中24K、D未命中为例：

1. 复用原生 bootstrap、destination 页表与 sender cursor，先取得 P 缓存读 pin。
   首次范围/请求摘要在 TP 内确认一次；后续 middle/final 复用该生命周期。
2. cached-prefix 的物理发送拆成有界资源子批次。每个 manager 同时只提交一个
   已接管 payload；按4 MiB目标、原生正的 max-transfer-batch 和 receive ring 容量
   划分，最小单位是一页资源组。逻辑 chunk 参数不变；staging 的最小物理页/资源组
   必须能放入上述字节预算，direct 路径的单页组可能超过4 MiB。
3. Qwen 的独立 metadata 发送期间暂停继续提交 global；当前有界批次完成后即可
   发送 metadata。DSV 的 cached/middle/final indexer 优先于排队的 global，完整
   indexer 就绪后发送原生 SWA/carry/metadata，再持续推进 global。
4. P 仍需完成剩余 prefill 和交接 token。DSV 仍需完整 indexer 与 SWA/carry 屏障。
   这些条件满足后，D 可在24K缓存 global KV 尚未传完时开始 Remote decode。
5. 每层/entry 的全部缺失区间到齐并具备 GPU 证明后，通过有序 TP 命令切 Local；
   canonical final marker 只在全部已计划资源完整发送后提交。最终退休继续等待
   native completion、GPU/Remote reader 及缓存发布共识，再恢复 full graph/overlap。

新路径不减少最终 global KV/indexer 总传输量。第一版按完整层/entry 切 Local，
没有把任意已到达 chunk 直接变成新的局部 attention 边界；DSV 的初始 D 缓存命中
仍按已有实现局部读取。其他未接管原生请求的发送策略保持；共享链路及原生锁仍可能
影响时延，因此需要目标 GPU 测量收益。

## 完成证明、保护与取消

- 每个区间绑定 generation、物理资源集合、唯一编号及 suffix-relative `[lo,hi)`。
  D 原有命中 `[0,H)` 独立计入覆盖，实际写入位置加回 H。尾块完成不能替代前缀。
- staging 复用 native CHUNK_READY 与 scatter GPU event。attempt tag 直接通过
  native STAGING_REQ 到达，允许跨机 journal 镜像晚到；同编号重试幂等，换绑或重叠
  区间拒绝。DSV 末页继续只写完成的 global 行，保护 D 新行、carry 和 padding。
- Qwen direct PD 使用成功 TE 回执，再做 D 侧 GPU 可见性处理和 event 证明；仅有
  enqueue、P TE callback 或分配时的 committed 长度都不直接触发 Local。
- Qwen 新流式路径也按层注册有序退休资源；DSV 复用已有 entry 资源。rank-local
  完成只发通知，最终命令顺序与票形状共享，等待期间和 Local 稳态不新增逐步投票。
- 缓存 pin 覆盖待提交/在途的所有区间及 Remote 读者。P 在 native abort/free 之前
  排空可证明的 IO；直接 reset 活跃 stream 被拒绝。P overlap 继续复用原生 forward /
  sampler 证明和事件驱动退休。未知 DMA 或注册变化隔离页面，不能把失败当成完成。
  P 串行路径沿用原有来源回收投票，前缀错误通过同一票传播，避免单 rank 抛错后
  其他 rank 独自进入该 collective。

本次 SGL 核心只扩展 `MooncakeKVManager.request_layer_staging` 的两个可选参数：
`chunk_id_base=0`、`stream_generation=None`。默认编号及原有8帧消息保持；新范围
调用额外带 attempt tag。源和安装态预检会拒绝旧核心缺少这两个参数的构建。

## DSV4.1 延迟 checkpoint 的 SWA 窗口

P 读租约期间不执行会去重、重绑定 FULL 页的 checkpoint。每个已计算 chunk
保留页对齐的 SWA 尾窗（含一页回退余量），其余窗口外 SWA 仍可回收。暂存窗口由
`ReqKvInfo.swa_retained_ranges` 记录；原生池计数、abort 和最终行释放都计入这些页。
所有活跃 source 共用一个不超过 SWA 池四分之一的保留预算，超预算丢弃较旧窗口。
缓存收益仍取决于窗口是否保留；不保证任意分叉点都命中。quarantine 和发布失败的
所有者不会因其他请求的预算压力被回收。

只有 GPU、native IO、Remote reader 与有序退休证明满足后，才按窗口终点顺序
回放原生 checkpoint，再执行最终发布与行释放。这样保留中间分叉窗口，也避免
`decoder_swa_bounded_replay` 裁剪匹配 key 后落在唯一最终尾窗之前：例如 N=32772、
page=256、window=128，匹配终点是 32512，需要保留该位置之前的 SWA。
无需关闭 bounded replay，也不绕过原生 FULL/SWA 组件校验。

新插件依赖核心的暂存窗口计数与释放支持，预检会检查 `swa_retained_ranges` 字段；
部署时一起更新核心与插件。本地覆盖原样重发、24K 分叉、131K、多 source 预算、
overlap/在途 DMA/Remote 屏障、abort 与发布失败；H20 数值和性能仍需实机验收。

## 验收

P `prefill.json` 的 `range_stream=true`、`cached_prefix_tokens>0` 和 `range_pieces`
确认实际接管；D `done.json` 的 `range_stream=true` 配合各 rank 的区间与 TP 退休
证明确认闭环。没有 activation 的请求不能计入新路径收益。

建议三组：完全原生 PD、旧 Remote（P flag=0）、新 Remote（P flag=1）。先保持两组
P cache on、D cache off，预热同一 P；再另测双端命中。固定 RDMA/TCP、模型、TP、
graph、overlap、chunk、optimistic、并发和发送 Batch，比较 greedy output_ids、
TTFT/ITL/E2E、吞吐、KV高水位与回收。必须证明 **D首次decode早于缓存global全部完成**，
并确认没有靠拖慢后台传输制造窗口。

本次仅本地测试，尚无真实 CUDA/RDMA PD 或性能验收；见
[本地验证记录](validation/PREFIX-REMOTE-PROGRESSIVE-LOCAL.md)。
