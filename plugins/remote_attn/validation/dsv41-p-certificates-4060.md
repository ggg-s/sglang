# DSV4.1 P TP 证书：本地与 RTX4060 验证

验证日期：2026-10-31。接入说明见 [DSV4.1 文档](../DSV41-LAYERWISE-ZH.md)。
只修改插件、预检、测试和文档，未修改 SMG/SGL 核心；原生 optimistic prefill、
chunk、batch 和 D overlap 配置保留。

## 本地 CPU 和真实 TP2

完整插件单文件回归 117 项通过，含 19 项 DSV4.1 专项。保留真实原生池、
Scheduler chunk/state planner、CommonKVSender 和 Mooncake page/byte planner；
模型、GPU 和外部传输分别在边界替代。

新增单机用例验证：稳定 middle→final 仅首次接管投票；否定证书只在 bootstrap
就绪后重确认；注册和 room 更新直到 GPU/IO drain、旧目标 abort 后才可见；
否定请求 reset 不 drain 其他请求；复用 rid/room 时旧通知不能命中新 birth。

真实两个 Gloo 进程使用实际证书、UDS 通知和插件请求处理/退休协议。native
request broadcast 所用 CPU 组用真实 Gloo 广播传递控制消息；该专项中 batch、
model forward、GPU event 和 IO callback 是 fixture，不是完整 SGL 服务。

逐次调用 `dist.all_reduce` 的向量长度，两端完全一致：

```text
[2, 2, 2, 2, 5, 5, 2, 5, 2, 2]
 A  B  R  C  TC TA RB TB  D  X
```

A/B/C/D 为新 attempt 的首次检查，R 为 B 的 bootstrap 重确认，TC 为 C 正常
退休，TA 为 A 的原生取消 drain，RB/TB 为 B 的撤销及 drain，X 为一个 rank 的
资格版本错误（所有 rank 投同形状向量后共同拒绝）。向量长度 2 是一个请求的
资格协议位和资格位；长度 5 是既有单请求退休协议。

- A 的连续 **63 个稳定 chunk：0 次新增接管 all_reduce**。
- B 的否定结果重复使用无投票；仅 rank 0 bootstrap ready 时也没有激活投票。
- rank 1 第一条通知模拟发送后丢失；未收到 ACK 时重传，最终正常激活且不反复投票。
- 注册更新先冻结旧视图；退休命令在途时不能另发资格命令。
- 原生取消在同轮请求处理内先消费 C 的退休，再 drain A，随后有序撤销 B。
  释放顺序均为 `[103, 101, 102]`，没有一端提前复用页。
- 16 次空轮询增加 0 次新投票；资格版本错误没有 rank 提前返回或继续模型 forward。

逐 rank 结果见 [TP2 结果](dsv41-p-certificates-tp2-results.json)。原始本机日志与结果
保留在 `/private/tmp/remote-attn-certificate-evidence-20261031`。
源码预检通过 34 个 hook、83 个核心接口；isort、ruff format/静态错误检查和
`git diff --check` 通过。

## RTX4060 真实 CUDA

通过用户授权的 `ubuntu` SSH 别名使用工作机 RTX4060 Laptop 8GB，复用已有
PyTorch 2.13.0+cu130 / CUDA 13.0 venv。独立目录：
`/tmp/dsv41-p-certificates-20261031`；未安装包或修改已有部署。

脚本 [prefill_overlap_cuda_probe.py](../../../test/manual/remote_attn/prefill_overlap_cuda_probe.py)
使用真实 CUDA stream/event、pinned D2H 索引和插件证书/GPU/IO/退休逻辑：

- 页索引 event 观察到 pending，完成后的 pinned host 索引逐元素一致。
- 4 个 launch plan，middle→final 复用首个接管决定；稳定接管投票计数为 0。
- 共 3 次首次检查、1 次 bootstrap 重确认、1 次资格撤销。
- 注册替换保持旧视图，待实际 GPU forward 和线程队列源读取完成后，按旧视图
  abort 再应用新注册；normal/cancel/revoke 各释放一次，共 3 次退休共识。
- 4 次 IO 完成回调；取消/撤销无需提前执行延迟采样，也不等待未录制 copy_done。
- 释放后 sentinel 保持正确，100 次空轮询不增加退休投票。

GPU 是 TP1，因此其协议计数不能证明多 rank 通信；TP2 顺序由上述 Gloo 用例验证。
模型计算、原生 envelope/abort metadata 和 IO 是小张量/线程队列 fixture，没有
运行 Mooncake TE、SMG、完整 SGL scheduler 或加载 DSV4.1 checkpoint。
未测多卡 NCCL、模型 token/logprob 正确性、TTFT 或吞吐收益。

结果与插件/probe SHA256 见 [CUDA 结果](dsv41-p-certificates-4060-results.json)。
复现（已有支持 CUDA 的 PyTorch，从 checkout 根目录）：

```bash
PYTHONPATH=plugins/remote_attn python \
  test/manual/remote_attn/prefill_overlap_cuda_probe.py \
  --output /tmp/dsv41-p-certificates-results.json
```
