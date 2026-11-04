# DSV4.1 Remote 热路径优化：4060 验证

日期：2026-11-04。设备：NVIDIA GeForce RTX 4060 Laptop GPU（8 GiB）；
PyTorch 2.13.0+cu130。独立测试目录，无服务部署变更。

## 结果与范围

- 插件 CPU/真实 TP2 Gloo 回归：261 项通过。覆盖真实 hybrid checkpoint/插入、
  SWA 预算与窗口保留、overlap 释放顺序、TP quarantine、引用和 allocator 收据；
  新增 7 项覆盖逐层 Local、页表增量、5→2 RPC 合批、提交时 IO 持有、未知 DMA
  poison、前缀流水线乱序/失败和公平 send lock。
- 新配置进入 TP 启动签名；不同 rank 的 FAST/在途上限不一致均拒绝。
- source-only preflight：49 hooks、125 core interfaces；`git diff --check` 通过。
  静态错误检查使用 Ruff E9/F63/F7/F82。
- [数值与 RPC 记录](dsv41-fast-4060.json)：V4、V41、V41_NEW、V41_FP4、
  V41_FP4_NEW × BF16/FP16 query × 2/16/33/64 heads，共 40 项通过。
  native 字节布局 fixture 与独立 Torch reader/partial 对照，FP32 partial/LSE
  容差为 atol=3e-4、rtol=3e-4；不是逐位相同。测试还覆盖 padding、重复 top-k、
  空集合/sink、非法 slot、非有限 query/sink、D cached prefix、ratio-2 odd tail。
- 真实 TCP control/socket 与原生 Mooncake engine 的 GPU buffer 通道均验证了
  5 行→2 RPC、Q producer event 不依赖 D local、结果正确、逐层恢复原生 callback、
  source refs 归零。GPU 通道无 socket tensor payload，使用实际注册 GPU buffers。
- TCP tensor 路径观察到 P Remote GPU 计算与 D local GPU 工作重叠。
  Mooncake 在该设备使用 **TCP**；观察到 Q 提前发送，但 P 计算未与 D local 重叠。
  不将其描述为 H20 RDMA 重叠或生产吞吐收益。
- [GPU 生命周期记录](dsv41-fast-lifetime-4060.json)：在途 GPU reader 的取消保留
  source pin，过早 Target.close 被拒绝；完成后 pin 归零、slab 可复用、无 quarantine
  槽残留；RPC teardown 后原生注册仍可传输。
- [P overlap/cache 记录](dsv41-fast-overlap-4060.json)：normal/cancel/revoke 三种
  路径均完成，读取先 drain、释放后无写入。CUDA TP1，IO callback/缓存锁边界为
  fixture；真实 hybrid checkpoint 与 TP2 quarantine 由 CPU 套件覆盖。

V4、32 heads、512 个选中行的单次算子微基准：旧 reader+partial **1.218 ms**，
融合路径（含错误标志检查）**0.300 ms**，约 **4.06×**。
这是小池算子测试，受 GPU 时钟与运行状态影响；没有加载 DSV4.1 权重，未测整模
TTFT/吞吐/文本精度，没有验证 H20 RDMA 或跨机 TP4 GPU。
原生 Local 恢复检查使用 ABI fixture 的 callback，非完整 FlashMLA 整模验证。

## 复现

在仓库根目录、具备 CUDA/Torch/Triton/Mooncake 的环境中：

```bash
export PYTHONPATH=plugins/remote_attn
python test/manual/remote_attn/dsv41_fast_cuda_probe.py --source-root . --output /tmp/fast.json
python test/manual/remote_attn/remote_gpu_lifetime_probe.py --source-root . --output /tmp/lifetime.json
python test/manual/remote_attn/prefill_overlap_cuda_probe.py --prefix-cache --output /tmp/overlap.json
python -m sglang_remote_attn.check --source-only
```

本地 CPU 使用 source-loading harness 运行 `TestRemoteAttn` 全部 261 项，绕过 Mac
缺少的 CUDA-only 导入，没有跳过该类测试。GPU 三份 JSON 内的源码 SHA256 已与
提交前工作树逐项核对一致；结果来自上述真实 kernel/传输探针。

生产配置和回退开关见 [Remote 说明](../DSV41-REMOTE-EAGER-ZH.md#remote-热路径优化2026-11-04)。
