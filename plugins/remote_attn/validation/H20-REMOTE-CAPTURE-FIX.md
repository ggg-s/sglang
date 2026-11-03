# H20 Remote 捕获与 Marlin workspace 修复

2026-11-03，基于 `3f4234d823`，目标分支 `codex/remote-attn-pre-layerwise`。

## 修改

Remote bank 显式使用 `capture_error_mode="thread_local"`。参数从后端工厂传入
`BreakableCudaGraphBackend`，再传给每个 `BreakableCUDAGraphCapture` segment。
原生 full graph、其他 breakable 调用者及 primitive 的默认值仍为 `global`。
`thread_local` 仍检查捕获线程自己的不安全 CUDA 调用，并非 `relaxed`；模式语义
见 [PyTorch CUDAGraph 文档](https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html)。

Marlin 的 `marlin_make_workspace` 原来每次调用都查询设备并新建 reduction locks，
仅重复 warmup 无法消除下一次调用的分配。Remote bank 现在启用按图、按调用顺序
独立的 `CaptureWorkspacePool`：

- 首次 warmup 在 capture 外分配；后续 warmup 和 capture 复用对应调用的地址。
- Marlin workspace 大小为 SM 数乘 `max_blocks_per_sm`，MoE 使用 4；该上界不依赖
  batch/token 数，无须删除 bs=1 或仅针对某个 bucket 放宽规则。
- 不同调用、bucket、图 bank 不共享 reduction locks。每次调用的 `zero_` 入图，
  每次 replay 都先清零，防止前次锁状态污染后续执行。
- capture 中出现未 warmup 的调用、不同的请求或不同调用数时立即拒绝，不临时分配。
  context 通过 `ContextVar` 隔离线程，异常时恢复，workspace 由 backend 保留到 cleanup。
- 没有开启该选项的原生调用者保留原来的逐调用分配行为。

保留 Remote buckets **1/2/4/8/16** 及已有 padding、native candidate variants。
插件预检检查新构造参数和 workspace API，拒绝混用旧 core。预检的字面量默认值
比较改为值和类型比较，避免字符串默认值因对象地址不同而误报。

## 验证

CPU 插件契约 **220 项通过**，含真实双进程 Gloo；workspace 契约 **3 项通过**，
原生 breakable 输出契约 **7 项通过**。
新增覆盖 warmup/capture 无额外分配或设备查询、调用和图之间地址独立、锁清零、
调用轨迹变化拒绝、线程隔离及默认值类型检查。预检最终为 **48 hooks / 118 core
interfaces**。本机 CPU 启动器仅用 `unittest.TestCase` 替代通用测试包装，模型与
CUDA/TE 边界仍使用原有 fixture；并非 CPU 上运行完整 serving 栈。

RTX4060 Laptop，PyTorch 2.13.0+cu130，独立目录
`/tmp/remote-h20-capture-fix.FzXkW1`；复用原验证环境，未修改或停止既有服务。

| 探针 | 结果 |
| --- | --- |
| BF16，真实 Marlin workspace allocator | 42 次 Remote、6 次 full replay，数值检查通过 |
| FP16，同上 | 42 次 Remote、6 次 full replay，数值检查通过 |
| workspace 地址 | 每组 5 buckets × 3 调用 = 15 个独立分配，均在 warmup 完成 |
| capture 约束 | Remote capture 中分配或设备查询会使探针失败，两组均未触发 |
| 跨线程模式 | 捕获期间另一线程真实 `cudaMalloc(4096)` 成功，捕获后 replay 正确 |
| DSV V4，串行 prepare | 78 次 Remote、8 次 full、582 次 RPC；最大绝对误差 0 |

两组 workspace 探针包含 bs=1/2/4/8/16、非整桶大小以及重复切换，主动写脏 scratch
以检查每次 replay 的清零。DSV 探针保留四种 native candidate variants、live
metadata、SWA/carry、padding 和 sampler 输出检查。

原始结果附源码 SHA256：
[BF16](h20-capture-fix-4060-bfloat16.json)、
[FP16](h20-capture-fix-4060-float16.json)、
[DSV V4](h20-capture-fix-4060-dsv41-v4.json)。

```bash
python test/registered/unit/plugins/test_remote_attn.py
python test/registered/unit/model_executor/runner_backend/test_capture_workspace.py
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only --sglang-root .
python test/manual/remote_attn/segmented_graph_probe.py --dtype bfloat16 --logits-output --marlin-workspaces
python test/manual/remote_attn/segmented_graph_probe.py --dtype float16 --logits-output --marlin-workspaces
PYTHONPATH=plugins/remote_attn python test/manual/remote_attn/dsv41_segmented_graph_probe.py --source-root . --output /tmp/dsv41-v4.json --kv-layout v4 --serial-prepare
```

## 验证边界

本次在 RTX4060 运行真实 CUDA capture/replay、原生 workspace allocator 和跨线程
CUDA 分配；模型、MoE 数学、attention/FlashMLA 与调度仍有小型 fixture 边界。
未运行 H20 整模 MXFP4 Marlin MoE/FlashMLA、多卡 NCCL 或 c=16 性能负载，不能据此
宣布 H20 remote 服务已完成验收。用户报告的 H20 `thread_local` 成功实验是修复
依据，不冒充本次重新测得的整模结果。

V4 index staging 末条不足完整 item 也已修复：补齐原生低压缩率 index 池的逻辑
guard 页分配，保留全部 staging 边界校验，见 [完整 item 对齐修复](DSV41-STAGING-ITEM-ALIGNMENT.md)。
三处已定位代码阻塞均已处理；H20 整模和服务性能仍需在目标部署上验收。
