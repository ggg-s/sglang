# DSV4.1 D receive host staging：本地与 RTX4060 验证

日期：2026-10-31。接入与原生启动方式见 [DSV4.1 文档](../DSV41-LAYERWISE-ZH.md)。
生产代码仅修改插件；SMG 和 SGL 核心无修改。

## 接入与生命周期

- 原生 `SGLANG_DISAGG_STAGING_BUFFER` 是 MHA/GQA 开关，会拒绝 MLA，保持为 0。
  D 使用已有插件变量 `HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=1`，在 native MLA
  manager 初始化后创建原生 pinned ring/GPU shadow 和 `DecodeStagingHandler`。
  不临时修改全局环境变量；P 的 send staging 可独立开关。
- offer 中新增 `dsv41-flat-v1` wire 标记。P 先验证同 TP、精确资源 manifest、原生 ring
  注册，再用原生 `request_layer_staging`、`PrefillStagingStrategy`、
  `_do_staging_transfer`、byte planner、`CHUNK_READY` 和 watermark 协议。
- 插件补 physical-entry 的分配字节数、发送地址和 H2D 后 scatter，保持 KV/indexer
  payload/scale 的原生 stride。没有 K/V 半分假设，不解量化或重打包 FP4。
  `ceil(prompt/page_size)` 允许非整页 final；ordinary middle、cached prefix 和
  SWA/carry/metadata 仍走原生顺序，chunk/transfer batch 上限不变。
- D 的原生 event 覆盖 row-ready wait、H2D 和所有 index scatter。event 完成后才 free
  allocation/推进 watermark。native receiver Success 仍需 scatter + metadata gate
  和原有 TP MIN，之后才进入 decode；无 early admission、无新增逐 token collective。
  P cache/overlap 证书及 D full graph/overlap 保留。
- D 取消/失败继续由 native queue 的 deferred abort ACK 和 room cleanup 管理；插件
  不创建 dense Target 的额外释放栅栏。native room release drain stream 后再释放 ring。
  GPU fence 失败则保留 allocation；engine 强引用 registered ring/KV。
  manager teardown 拒绝 active room/区间或未退出 worker；正常 drain 后显式 deregister。
- receive allocation 过大时原生永久失败，不退回等待 staging 的 D 直写地址。P 检查
  native PD 失败先于 pending-success-callback 快路，进入已有有序错误/退休协议，避免
  被永不触发的成功回调掩盖。稳定 chunk 不增加接管投票。

## 本地结果

完整插件 **130 项通过**，其中 **32 项 DSV4.1**。新增五项 receive 用例覆盖：

1. 真实 native allocator/handler/watermark + byte planner，12 条目混合 stride，
   最终 3 页到乱序目标页，字节完全一致；GPU event 未完成不得 free ring 或提交 D。
2. allocation overcommit/ring 回绕；重复 allocation 不重复分配，已消费的 id 不复活；
   拒绝重叠条目、变更 descriptor、越界 id、未知 wire 和超限分组。
3. 取消时 actual native room release 先 drain GPU，再释放已提交和未提交 allocation；
   late notification 不重建 room。
4. D 插件启动绑定、原生完成门控、offer 发布和 clean teardown；active room 拒绝 teardown，
   正常清空后 deregister 并撤销 engine ownership。
5. GPU drain 抛错时 allocation/event 保留，不把资源归还池。

CPU 的 CUDA/TE 边界使用真实内存字节替身，未用 mock 替换 native ring/planner/handler。
真实 TP2 Gloo 还执行 native `poll_and_all_reduce_with_staging`：rank 1 H2D 未完成、
metadata 未到达、全部完成、单 rank 失败依次得到 `[3], [3], [4], [0]`，两 rank 一致；
四次均是原生 1 元素 MIN，没有附加接收投票。原有接管证书、取消/retraction、缓存读租约
及稳定 middle→final 0 接管投票用例继续通过。

源码预检 **39 hooks / 88 core interfaces**；isort、ruff format/选定错误检查和
`git diff --check` 通过。

## RTX4060 结果

使用已授权 `ubuntu` 工作机（RTX4060 Laptop 8GB，PyTorch 2.13.0+cu130），独立
`/tmp/dsv41-d-receive-staging-20261031`，复用已有 venv，未安装包或修改部署。

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
PYTHONPATH=plugins/remote_attn PYTHONDONTWRITEBYTECODE=1 \
/workspace/remote-attn-cuda-20261028/venv/bin/python \
  test/manual/remote_attn/dsv41_receive_staging_cuda_probe.py \
  --sglang-root . --output receive-result.json
```

- 两个真实 Mooncake TCP endpoint，共用一张 GPU；原生 1 MiB pinned receive ring +
  GPU shadow，保留原生容量/分配算法，通过扩大 toy KV 条目实际触发回绕。
- native stride 为 `131072/131072/131072/32/32/32/8/8/8/131072/64/16`。
  P 直传→D host 和 P 64 KiB D2H slot→D host 两种模式均逐字节一致；未选页不变。
  两种模式均实际观察到 pending H2D/scatter event，完成后 ring 清空，watermark
  分别为 `(2,0)` 和 `(4,0)`。
- 实际 GPU copy/scatter 尚在队列中时取消，native release 等待 event；已提交和未提交
  allocation 均回收。随后覆盖目标页并 drain stream，未发生释放后写入。
  正常 manager teardown 之后 ring 确已从 TE deregister。
- 回归 P 原生 host staging：12 条目、257 字节 slot、32 host batches、字节一致，
  原生小状态直传、TE 失败隔离且不重试通过。
- 回归 `prefill_overlap_cuda_probe.py --prefix-cache`：四个 launch plan，normal/cancel/revoke
  三次回收、缓存读租约 drain、真实 pinned indices/event、稳定接管/空轮询 0 新投票通过。
  该独立 overlap 探针的 IO 是队列边界，不计作 Mooncake E2E。

完整 [结果 JSON](dsv41-d-receive-staging-4060-results.json) 包含 receive / P-send / overlap
三组结果。分别校验 **28 / 25 / 18** 份源码 SHA256，均与提交前本地文件一致。

## 验证边界

未加载 DSV4.1 checkpoint、未运行完整 SMG/P/D 服务或多 GPU NCCL；没有模型精度、
生产吞吐和尾延迟收益结论。receive 探针的控制消息在进程内交付给真实 native 协议方法，
数据通过真实 Mooncake TCP；未验证真实 ZMQ 控制网络丢包、跨进程 deferred abort ACK
或网络故障恢复。GPU 取消用例在对应 TE 写入返回后触发，证明 H2D/scatter drain；
完整网络写入与 abort ACK 的组合仍需服务级灰度。

模型灰度保持同一 revision 和 optimistic/chunk/batch 配置，对比 receive staging off/on
的 token/logprob、TTFT、P→D 接管间隙、吞吐、峰值 pinned/GPU 内存及取消/retraction 回收。
