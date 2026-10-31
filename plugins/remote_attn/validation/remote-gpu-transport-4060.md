# Remote GPU Q/O/LSE：本地、TP2 与4060验证

日期：2026-11-01。基线 `1c02750147`；本次只修改插件、测试和文档，SMG/SGL 核心没有新增修改。
[启用方式与生命周期协议](../REMOTE-GPU-TRANSPORT-ZH.md)，
[原始结果及源码 SHA256](remote-gpu-transport-4060-results.json)。

## 本地与 TP2

**170 项完整插件测试通过，包含59项 DSV4.1 测试与5项新增 GPU transport 合约测试。**

- 真实 RemoteHub 控制 socket、CPU Tensor 及受注册范围约束的 TE memcpy 测试边界：
  batch 1/2/4 Q/O/LSE 往返、稳定地址复用、返回 tensor 不别名接收 slot。
- `_send_tensor` 禁用时请求仍完成；MHA 使用真实插件 native executor 分支，外部
  attention kernel 替身下禁用 `Tensor.cpu`，保证 GPU 分支不悄悄回到 CPU payload。
- 确定性在途取消：reader gate 未完成时 registry tombstone 不触发源释放；GPU job 和
  结果写入完成后才归还 reader 引用，D transport IO 未完成时拒绝退休。
- 模拟未知 DMA：旧注册地址保留且不自动重试，新请求使用新地址；旧地址的迟到写入
  不会覆盖新请求结果。ACK、单调序号和最后读取 event 缺一不可复用。
- 真实 Gloo TP2：GPU transport 开关不一致在启动共识拒绝；一 rank 的 GPU payload
  consumer 未完成时不释放目标、不新增逐循环退休投票，完成后走原有有序最终共识。
  既有 P 证书、缓存、overlap、取消/retraction、逐资源 Local 和 graph 回归继续通过。

源码预检 **44 hooks / 108 core interfaces**；新增显式检查原生 engine 的注册、注销、
批量写入和 session 接口。isort、ruff 与 diff 检查通过。

## RTX4060

用户授权 ubuntu 工作机 RTX4060 Laptop 8GB，PyTorch 2.13.0+cu130；隔离目录
`/tmp/remote-gpu-payload-20261101`，使用已有 venv，没有安装包或修改部署。

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
PYTHONPATH=plugins/remote_attn PYTHONDONTWRITEBYTECODE=1 \
/workspace/remote-attn-cuda-20261028/venv/bin/python \
  test/manual/remote_attn/dsv41_segmented_graph_probe.py \
  --source-root . --gpu-transfer --mesh --output gpu-mesh-graphs.json
# 另测单 pair、--fused-projection --bf16-kv-source，以及 remote_gpu_lifetime_probe.py。
```

- 使用实际 Mooncake C++ TransferEngine，并从仓库源码加载 SGL 的
  `MooncakeTransferEngine.batch_register/batch_deregister/batch_transfer_sync/get_session_id`。
  四个 P/D engine 的注册 GPU slab 和每请求控制认证实际运行；没有替换数据传输函数。
- 单 pair：42次 Q 写入及42次 O/LSE 批量回写；两 P/两 D：44次各方向传输，包含 D0
  关闭后 D1 独立访问两个 P。多来源 native KV bytes 不同，避免误路由仍通过数值比较。
- 默认与融合/BF16组合：每组 batch 1/2/4、三个 progressive 阶段共18次 Remote graph
  replay、8次 Local full replay，graph/eager 最大绝对误差 **0**；live SWA/carry 检查通过。
  原有三套 Remote banks（各8段）、四种 native candidate variants 保留，捕获零 RPC。
- 在 GPU 路径禁用 RemoteHub `_send_tensor/_recv_q` 后测试仍完成，证明该控制连接
  不传 Q/O/LSE payload。传输 API 的输入/输出是注册的 GPU 地址。
- 实际 GPU 延迟 reader：GPU未完成时 Target.close 被拒绝、registry reader/cache pin
  保留；完成后顺利回传，回收 pin，下一请求安全复用同一 slot。
- RPC slabs 注销后，先前独立注册的 canary 仍能用同一个原生 engine 传输；未破坏其
  他内存注册。正常场景无隔离 slot，原有单 pair、progressive/cache 保护及 Qwen 共享图探针通过。

第一轮实测发现 GPU job 缺少旧 sparse executor 依赖的 owner epoch 字段；补齐后重跑。
本地测试还明确了正常 teardown 顺序：先关闭 D client、完成双端注销，再关闭 source hub。

## 验证范围

4060上物理传输为 **Mooncake TCP 后端**。该测试证明插件通过原生 Mooncake 读写 GPU
缓冲区、RemoteHub TCP 只传控制消息；不代表 NVLink、IB/RoCE/GDR 硬件已验收，也不代表
底层 Mooncake TCP 实现没有内部 host staging。已核对安装版本的 CUDA visibility API 名称；
RDMA writes-ordering/flush 的硬件行为仍需目标机器验证。

native graph backend/primitives、QKV/state 方法、native-byte sparse GPU reader、GPU
事件、Mooncake 数据搬运与控制 RPC 真实执行。三层模型的 linear/norm/store kernels、
indexer/carry planner、原生 PD admission 为小型夹具；MHA kernel 在 CPU合约测试中是外部边界。
没有加载完整 checkpoint、运行完整 SMG/P/D 或多 GPU NCCL，也未证明 TTFT/ITL/吞吐收益。
