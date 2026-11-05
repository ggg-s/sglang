# D2D bulk v2：两个 P1 修复与 4060 复测

日期：2026-11-05。分支 `codex/remote-attn-pre-layerwise`，基线
`cc0cc820ade5a4ec206a2a977cfa216c24d24b46` 加当前未提交修改。

结论：两个 P1 已修复。完整 CPU 合同回归 301/301；4060 CUDA 正向迁移 8/8；
两个故障回归各 3/3 通过。修复前的 [问题复现记录](dsv-d2d-bulk-v2-4060-review.md)
保留，修复后 [完整 JSON/源码 SHA256](dsv-d2d-bulk-v2-4060-fixed.json) 独立保存。

## 修改

- `TargetClient.call` 增加本地 `retry_on_disconnect` 选项，D2D bulk 读取显式关闭
  隐式重试。首次不确定断连直接交给 Reader 隔离，第二次请求的“未提交”响应不再
  可能被用来清理首次写入的内存。Attention RPC 的默认幂等重试行为不变。
- engine-owned `Sender.copy_owner` 在提交 D2H 前强引用 GPU 输入 tensor 和生产
  event，仅在 `CudaPinnedSlot.copy()` 的完成 event 同步成功后清除。若提交或
  完成证明失败，输入与 Host 缓冲区一起保留，禁止其他 stream 提前复用显存。
- 没有增加 GPU 全局同步或改变 wire schema。原有逐 entry event、scatter 尾页
  保护、TP 有序 Local 和源 KV 退休条件保持不变。

## 验证结果

| 检查 | 结果 |
|---|---|
| 完整 CPU 合同回归（含 TP2） | 301/301 |
| 新增 CPU：断连禁止重试及保留接收区、D2H 输入异常保留、成功后释放 | 3/3 |
| CUDA：C2/V4、C2/V41 FP4、C4/V4、C128/V4，各两种 Q/O 通道 | 8/8 |
| CUDA：断连/取消与未完成 bulk 写入交错 | 3/3 |
| CUDA：真实异步 D2H 提交后丢失完成证明、尝试复用原地址 | 3/3 |
| Ruff 静态检查、格式检查和 `git diff --check` | 通过 |

断连用例均为 `reads=1`、`rpc_retries=0`，接收 slot 保持注册且隔离，prepared
abort 被拒绝，源端 resume 等待 reader。放行原始 Mooncake 写入后正常完成，不会
向失效地址写入。

D2H 用例均为输入仍有强引用、源地址未被复用、Host 中原始数据为 17；不再出现
修复前被替换数据 231 覆盖的情况。异常 sender 仍隔离，不能通过 close 绕过。

正向用例继续验证状态字节、Remote 提前接管、尾页/padding、GPU forward 延迟
释放、源端只释放一次和恢复 Local。C128 的数值比较沿用相对容差，完整误差见 JSON。

## 环境与复现

RTX 4060 Laptop 8188 MiB，驱动 592.82；实际加载 torch `2.13.0+cu130`、
Triton `3.7.1`，Mooncake CUDA13 wheel `0.3.13.post1`，实际链路为 Mooncake TCP。
D0/D1 使用同卡独立小池和真实 CUDA/TE/socket；scheduler 和分配使用 native ABI
夹具，不是完整模型、HTTP/SSE 或 RDMA 验收。

在隔离源码目录 `/tmp/d2d-bulk-p1-fixed-4060.2Ly52M` 中运行：

```bash
PYTHONPATH=python:plugins/remote_attn python test/manual/remote_attn/dsv_d2d_cuda_probe.py \
  --source-root "$PWD" --output positive-all-final.json

PYTHONPATH=python:plugins/remote_attn python test/manual/remote_attn/dsv_d2d_bulk_cuda_review.py \
  --source-root "$PWD" --case retry-cancel --output retry-cancel.json

PYTHONPATH=python:plugins/remote_attn python test/manual/remote_attn/dsv_d2d_bulk_cuda_review.py \
  --source-root "$PWD" --case d2h-input --output d2h-input.json
```

故障探针现在输出 `safety_pass`/`status`，并在写入报告后断言安全条件；不满足即
以非零状态退出。每个故障用例分别用新进程执行三次。

原始日志归档：本地 `/private/tmp/d2d-p1-fix/results.tar.gz`；CPU 日志：
`/private/tmp/d2d-p1-fix/full-cpu.log`。测试后 GPU 使用量恢复为 0 MiB。
