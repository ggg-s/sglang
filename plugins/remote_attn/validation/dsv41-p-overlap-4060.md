# DSV4.1 P overlap：本地与 RTX4060 验证

本页保留 `b891fd39d5` 的 P overlap 历史基线；最新 TP 证书验证见
[新记录](dsv41-p-certificates-4060.md)。当前 probe 已扩展为证书用例，复现旧基线请 checkout 上述提交。

验证日期：2026-10-31。只修改插件、预检、文档和测试，没有修改 SMG/SGL 核心。
实现说明与原生启动命令见 [接入文档](../DSV41-LAYERWISE-ZH.md)。

## 本地 CPU

`test/registered/unit/plugins/test_remote_attn.py` 共 112 项通过，其中 DSV4.1 专项
14 项。使用本地 CPU PyTorch，源码加载原生 pool/Scheduler/CommonKVSender/
Mooncake 的 planner；CUDA event、传输和模型进程是边界 fixture。

新增覆盖：

- 同请求 middle→final，提前入队后的原生 result handler 不重复发送；末尾非整页、
  奇数 carry 和原生 sender 的累计页数正确。
- 两个 final 批次先后 launch、结果延后处理，producer hook、索引、发送队列和
  result plan 分别归属各自请求；嵌套 tail 图仅按当前 plan 绕过，不改模型对象。
- 接管前已排队的 native prefix/chunk 被继承；GPU ready、sender Success 均不能
  单独替代 IO 队列完成证明；延迟采样未提交时不查询空的 copy_done。
- 取消、optimistic retry、retraction/reset 在原生状态变更前 drain；行/attempt 改变
  时拒绝释放。GPU/入队异常保留租约；allocator 回调部分失败后不重复调用。
- 32K prompt + 17-token final tail 保留自动 layerwise 策略。

`test/manual/remote_attn/tp_control_worker.py` 使用真实双进程 Gloo、不同 rank 的
transport generation、UDS 边沿通知及原生 request receiver 的广播边界。新增 P
租约用例使用相同有序协议：慢 rank 未就绪时没有退休 all_reduce，正常完成和
取消各一次最终共识（向量长度均为 5），各 rank 一致释放 `[1, 2]`；32 次空轮询
增加 0 次退休投票。既有 D overlap 用例同时回归。

无 GPU 的源码预检通过：34 个 hook、80 个核心接口。isort、ruff format、选定
静态错误检查及 `git diff --check` 通过。

## 工作机 RTX4060

主机为 SSH 配置 `ubuntu`（经 `windows` 跳转的 WSL）。独立目录：
`/tmp/dsv41-p-overlap-20261031`；复用已有
`/workspace/remote-attn-cuda-20261028/venv/bin/python`，未安装包或改动已有部署。
GPU 为 NVIDIA GeForce RTX4060 Laptop 8GB，PyTorch 2.13.0+cu130 / CUDA 13.0。

脚本 [prefill_overlap_cuda_probe.py](../../../test/manual/remote_attn/prefill_overlap_cuda_probe.py)
使用实际插件代码、CUDA stream/event、pinned D2H 页索引和 TP1 退休：

- 索引 event 确实观察到 pending，完成后 pinned host 页索引逐元素一致。
- 3 个 launch plan，3 次 IO 完成回调；normal/cancel 各释放一次，最终共识共 2 次。
- GPU 完成但队列未完成时不得回收；取消无需提前执行延迟采样或等待未录制的
  copy_done，必须等待模型 forward event 和全部源读取回调。
- 释放后写 sentinel，确认没有晚到的源读取/回调；100 次空轮询增加 0 次退休投票。

结果与全部插件 Python 文件、probe 的 SHA256 保存在
[dsv41-p-overlap-4060-results.json](dsv41-p-overlap-4060-results.json)。

复现（checkout 根目录，已有支持 CUDA 的 PyTorch）：

```bash
PYTHONPATH=plugins/remote_attn python \
  test/manual/remote_attn/prefill_overlap_cuda_probe.py \
  --output /tmp/dsv41-p-overlap-results.json
```

此 GPU 用例的模型计算/原生 envelope 是小张量与线程队列 fixture，没有加载
DSV4.1 checkpoint、Mooncake TE、SMG 或完整 SGL 调度服务；不是多卡 NCCL 验证。
未测模型 token/logprob 一致性、TTFT 或吞吐收益。完整模型 PD 的生产验收仍须按
接入文档执行，不能以这些小型用例宣布生产可用。
