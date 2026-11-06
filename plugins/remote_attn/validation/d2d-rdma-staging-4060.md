# D2D Mooncake engine / GPU staging backend 验证

日期：2026-11-06。分支：`codex/remote-attn-pre-layerwise`，基线
`5884814306731be4485f3fa7ef3666e8fbe1979e`，本次修改尚未提交。

状态：本地契约测试与 RTX 4060 CUDA/TCP 验证通过；**真实 RDMA 双机验收未完成**。
配置和语义见 [D2D RDMA 文档](../D2D-RDMA-ZH.md)。

## 验证范围

| 项目 | 用例数 | 结果 |
| --- | ---: | --- |
| 既有 remote-attn CPU / socket / TP2 Gloo 契约 | 361 | PASS |
| engine 身份、能力交集、严格模式、注册隔离、visibility 顺序、原生初始化协议记录 | 15 | PASS |
| Qwen 原生 Mooncake TCP：FP16/BF16、同步/异步、group/slot/range 组合、取消、RemoteDrop、状态丢失 | 8 | PASS |
| DSV4 / DSV4.1 原生 Mooncake TCP：同组 entry 独立 Local、源端延迟回收 | 2 | PASS |
| 真实 CUDA GPU staging、明确 fake NIC/visibility 边界 | 7 | PASS |

共 376 项 CPU 契约与 17 项 GPU 环境用例。RTX 4060 Laptop GPU；现有 CUDA13
验证 venv。GPU 用例使用小型原生 pool / scheduler ABI fixture，不加载完整模型权重。
最后一组 7 项具体包括：

- Qwen：原生 GPU KV、CPU snapshot 输入、写入后完成状态丢失。
- DSV4 和 DSV4.1：逐 entry 渐进 Local，最后 entry 未就绪时源端不能释放。
- DSV4 和 DSV4.1：bulk 写入后状态丢失，source/target GPU staging 仍注册且保留；
  target abort / source resume 被拒绝，原生请求 KV 不释放，不虚报 Local。

`rdma_staging_cuda_probe.py` 的 fake NIC 使用独立 CUDA stream 上的
`cudaMemcpyAsync(DeviceToDevice)`，同步其完成后才返回。它检查注册区间和 CUDA
pointer 类型，并能在实际拷贝后注入完成结果丢失。GPU tensor、CUDA event、gather、
copy、scatter、socket 控制协议、entry retirement 和异常隔离均走真实实现。
**MR/HCA 注册、GPUDirect 外部写可见性、跨机 DMA 与 NIC 完成队列未由此验证。**

## 关键检查

- Qwen 获取原生 PD 的 rank-local engine binding；同一原生 wrapper/raw 对应同一
  engine 身份。TCP fallback 是独立、共享的已知 TCP engine。
- 双方能力取交集，优先 GPU RDMA；strict RDMA 不兼容时拒绝，未知协议不冒充 TCP。
  使用源加载的真实 `MooncakeTransferEngine.initialize` 验证协议记录与强制覆盖。
- GPU 注册/注销探针不确定时保留 backing，并关闭该 engine/device 的 GPU 能力。
- 接收端 visibility 在 CUDA copy 前执行；返回的 GPU 数据不别名可复用 staging。
- 未改变原生 reserve-before-freeze、分页尾部裁剪、FULL/SWA 预留、TP 一致性和 P4
  SSE/重启恢复路径；原有 361 项回归保持通过。
- source/target native completion 不明时禁止回退重传、提前 unregister、释放或复用。

## 复现

在有完整 SGLang 测试依赖的 checkout 中：

```bash
export PYTHONPATH=python:plugins/remote_attn:test/manual/remote_attn
python test/registered/unit/plugins/test_remote_attn.py
python test/registered/unit/plugins/test_d2d_mooncake_backend.py

python test/manual/remote_attn/rdma_staging_cuda_probe.py \
  --source-root "$PWD" --output /tmp/d2d-rdma/gpu-staging.json
python test/manual/remote_attn/range_layer_cuda_probe.py \
  --output /tmp/d2d-rdma/qwen-tcp.json
HANDOFF_D2D_GROUP_SIZE=3 HANDOFF_D2D_TOKEN_RANGE=17 \
  python test/manual/remote_attn/group_layer_cuda_probe.py \
  --source-root "$PWD" --suite dsv-local --output /tmp/d2d-rdma/dsv-tcp-local.json
```

本次 GPU 快照：`/tmp/d2d-rdma-4060.cNGi6Q`。原始结果和 99 个实际使用源码文件
的 SHA-256 记录在 [JSON 报告](d2d-rdma-staging-4060.json)。本地 CPU 测试使用
隔离 Torch 环境和最小 SGLang 顶层测试导入边界；传输/分配契约实现未替换。

## 待真实 RDMA 环境验收

两台具备受支持 GPU/HCA/驱动的机器：验证 GPU registration、RDMA visibility、
多 NIC / 多 QP 的完整 batch 完成、混合能力 TCP fallback、网络故障和取消中的
迟到写。还需实测迁移吞吐、首次 entry Local-ready 和 Decode P99。
本报告不宣称真实 RDMA 正确性验收、完整模型推理等价性或性能提升。
