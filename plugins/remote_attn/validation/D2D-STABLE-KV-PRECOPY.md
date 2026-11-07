# D2D：冻结前复制稳定 KV

2026-11-07；基线 `77a0fb614b`；分支 `codex/remote-attn-pre-layerwise`。

## 行为和边界

```text
PLAN → D1 RESERVE → D0 继续执行 / 有界预复制 → FREEZE
                      ↓                       ↓
                 D1 未准入预留空间       退休原 PD / 补尾部与可变状态
                                                ↓
                                      IMPORT → COMMIT → 目标执行
                                                ↓
                                   reader / GPU / TP 退休 → D0 free
```

- 只复制完整闭合页。Native committed_len/分配成功不能单独证明数据有效：
  Progressive PD 要求对应 entry 的原生 scatter/GPU 完成证明，或已 Local。
  未收到的 Remote entry、开放尾页，以及 SWA、carry、GDN 等可变状态不提前复制。
- 支持首 token 前的已完成 PD entry；源端尚不能 freeze 时仍可准备数据。
  正常 cut 等待期间持续推进；cut/drain 到达时固定一个稳定页快照，补完这个有界
  快照后冻结，不无限追逐持续增长的尾部。准备期间源结束则回滚，不能计为迁移。
- 每块不超过 4 MiB，PREPARE/COPY/APPLY 分别在 TP scheduler 执行。块间 D0 可继续
  调度。COPY 等待真实 forward stream，gather 后同步 GPU 读证明才返回。
  Mooncake DMA 访问独立私有 staging，不直接写 ticket/native row。
- COPY 的计划摘要、页映射、buffer 指针、attempt 和源回执，与 D1 已 APPLY 的
  回执在 freeze/import 两端校验。页映射或 attempt 改变则拒绝旧预复制结果。
  序号/nonce 和持久化 chunk id 约束重试；丢回复不能重复发起旧 DMA 或重复 scatter。
- 冻结后跳过已认证页，只传新尾部、开放页和全部可变状态。预复制不标记请求 Local，
  不释放 D0 原 KV，也不允许 D1 提前采样。最终 commit、旧 PD 退休及所有 reader /
  GPU / TP 完成后，仍通过既有 native free 回调释放 D0 一次。
- 取消先关闭源生产者，再处理目标私有槽。若最后一块已发送但未 APPLY，producer-stop
  仍不足以证明 peer-arrival，目标必须观察接收 footer 并完成自己的 GPU 读后才能
  注销槽。无法证明停止/到达时保留私有 backing；native reservation 在自身 GPU
  fence 后可回收，迟到写入不能触及复用后的 native KV。
- GPU gather/scatter 完成证明失败则保留相应 native owner。私有 transport 隔离会
  阻止该 worker 被误报为 fully idle。Source forward_stream 的声明已纳入源码预检。

目前 PREPARE/COPY/APPLY 仍是三个控制命令；本次改善的是冻结后的数据量，不承诺
短请求一定比自然排空更早释放，也未把文件控制、原 PD 退休或可变状态传输成本消除。

## 验证证据

[4060 机器结果及源文件摘要](d2d-precopy-4060.json)。RTX 4060 Laptop 8GB，真实
CUDA scatter/event、Mooncake TCP；模型采样和 native allocator 是明确 ABI fixture。
不是完整 32k/128 推理、跨机 RDMA 或 H20 性能结果。

| 并发 | 实际迁移 | 每请求源释放次数 | 冻结前字节 | 冻结后字节 | 完整快照字节 |
|---:|---:|---:|---:|---:|---:|
| 4 | 4/4 | 1 | 145152 | 279040 | 424192 |
| 8 | 8/8 | 1 | 290304 | 558080 | 848384 |

两档均验证首 token 前预复制、源继续生成、freeze 时仍部分 Remote、最终原生字节和
可变状态一致、成功路径私有槽全部回收。冻结后字节减少 **34.2%**。小池中从 freeze
命令 dispatch 到 native source release 分别约 308/639 ms，包含此探针的驱动和
旧 PD 退休；不能与 H20 的历史 2.4 s 直接相减，也不等同于真实模型 token 停顿。

取消 CUDA 探针实际让旧 Mooncake 写入在 native ticket 回收后到达：复用的 GPU KV
保持不变，旧私有槽被保留。CPU 定向六项保留：增量数据、映射变化、双端回执、
晚到写入、取消等待 peer-arrival、GPU 失败保留；均通过。既有丢回复恢复用例覆盖
submit/reserve/freeze/import 和三个预复制阶段，真实双进程 Gloo TP2 覆盖非对称稳定
页拒绝、每 rank 字节和退休，均通过。51 hooks / 146 core 接口、Ruff、空白检查通过。

按要求减少重复测试：删除重复参数轮次和单纯时间字段用例，最终仅复跑受影响检查。
较早全量运行发现两处旧固定 tick 上限断言，新阶段接入后已调整，并以相应恢复
用例复测通过；没有把早期全量结果冒充最终全量通过。

## 与自然排空使用同一起点

新增 `bench_p4_dsv_drain.py`；先完成同一 prompt 的预热及 KV 回收，测量请求均以
cut=max_new 禁止自动迁移。所有请求达到相同 trigger-token 后，**POST /v1/drain
之前**开始客户端单调时钟；自然与主动均从这里计到 source SAFE_TO_STOP，包括
主动路径的 plan/reserve/precopy 成本。该状态同时要求 native KV/租约释放和 source
fully idle。另记录 last-source→first-target 间隔、实际 owner 切换和序列连续性。

自然模式 `/v1/drain {"mode":"natural"}` 停止 D0 新准入，让现有请求自然结束；只接受
尚无迁移 epoch 的源请求，防止把已经预复制/迁移的流混入基线。两种模式均把新请求
路由到 D1。每次运行使用独立的新 gateway/source 状态，保持 P/D 配置、提示、采样、
预热、并发和触发 token 一致；建议交错运行两种模式，保存原始 JSON。

```bash
# 每种模式前恢复同样的新 gateway/source 状态；分别跑 c=4 和 c=8。
python test/manual/remote_attn/bench_p4_dsv_drain.py \
  --gateway http://GATEWAY:31080 --mode natural \
  --prompt-ids prompt32k.json --concurrency 4 \
  --max-new-tokens 128 --trigger-token 1 --output natural-c4.json

python test/manual/remote_attn/bench_p4_dsv_drain.py \
  --gateway http://GATEWAY:31080 --mode d2d \
  --prompt-ids prompt32k.json --concurrency 4 \
  --max-new-tokens 128 --trigger-token 1 --output precopy-c4.json
```

比较 `source_safe_to_stop_ms`，同时检查每个请求的 `migrated`、owner/seq 连续性和
`last_source_to_first_target_ms`。不能拿自然路径的 E2EL 与主动路径的 PLAN→COMMIT
混比。H20 内网不可访问，因此本次提供并校验了同起点的控制/所有权口径和实机复测
入口，**尚未验证 H20 模型上相对自然排空的实际收益**。
