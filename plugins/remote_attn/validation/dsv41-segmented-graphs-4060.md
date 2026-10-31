# DSV4.1 D Remote 分段 CUDA graph：本地与4060验证

日期：2026-10-31。生产改动仅在插件；SMG/SGL 核心无新增修改。
启动要求和捕获范围见 [Remote 指南](../DSV41-REMOTE-EAGER-ZH.md#d-remote-分段-cuda-graph)。
[完整 CUDA 结果及 SHA256](dsv41-segmented-graphs-4060-results.json)。

## 实现与安全边界

- 复用 Qwen 的 native full bank + native BreakableCudaGraphBackend、eager_on_graph、
  private Remote pool、bucket output bridges、完成 event 和 TP 生命周期证书。
- DSV 仅额外捕获 batch 1/2/4；Remote 不 padding，3/8 等回退 eager。Local 原生 full
  bank 的 buckets/padding 保留，不在每个请求、每个 token 或资源切换时重新捕获。
- metadata raw→full 升级在图外；MQALayer 原生 serial/multi-stream/low-ratio multi-stream
  prepare 整段在图外，保留 native compressor/indexer、fused SWA 写入、carry 与侧流 joins。
  **QKV 投影也暂留图外**。attention 后投影、norm、MLP/MoE 等其余稳定前向在图内。
  本阶段未将 prepare 内的投影进一步细分。
- native backend.forward 是另一个 break：动态 sparse/top-k 读取、Remote RPC、partial
  attention/LSE merge，以及 request-window commit 都在图外。Local full capture 不改变。
- 回放闭包不接收/保存 capture-time ForwardBatch；metadata、prepare、attention 都使用
  当前 active batch。K/V 只保留一个 tensor 参数，再重建 native `k is v`；eager 结果的
  shape/dtype/device type 与捕获 bridge 一致。一个 token 内冻结并核对 row/attempt/Local mask。
- L 层实际边界须为 metadata + 每层 prepare/attention，L 层通常 2L+2 个 segment。
  实际捕获边界纳入启动 graph manifest；启动开关、bucket、模型/压缩配置及原生方法 ABI
  在额外 capture barriers 前进行 TP 共识。
- graph 判定复用 TP-shared targets：Remote、部分 Local、拷完但未退休都保持 Remote bank；
  GPU sampling/readback、native transfer/scatter 和 TP 最终退休后，才恢复 full。
  原有 scheduler Remote 串行/Local overlap 边界保留，不新增逐 decode all_reduce。
- 异常不重跑同一个 token。prepare/metadata 异常可能缺失 native side-stream join，
  标记 graph bank unsafe、保留 storage；主 stream event 不能独自证明这种情况下的释放安全。

## 本地

**156 项完整插件测试通过，其中 55 项 DSV4.1。** 新增六个 derived-property/生命周期测试：

1. 仅三个 Remote buckets；捕获零 RPC；每轮 live metadata、prepare 与 attention 使用
   当前 batch，dummy batch 不泄漏；三类 prepare 返回结构分别覆盖 tuple 和 Q tensor。
2. exact Remote batch 1/2/4，其他形状 eager；无 progressive lease 的 post-PD compatibility
   窗口保持 eager；Local 恢复 native predicate，32 轮 Local 不增加同步。
3. token 中途资源 mask 改变、结果 bridge shape 不匹配立即拒绝；回放结束清理 live 引用。
4. native prepare 失败后拒绝把未 join 的侧流当成已完成，禁止释放 graph bank。
5. native ABI 不兼容或 metadata/prepare/attention 边界缺失不能静默通过捕获。
6. 四个原生 candidate variants 的 full graph 保留；同 batch native keys 仅复用一套
   Remote 捕获；metadata/prepare/attention 按 live label 执行，作用域退出恢复原值。

既有真实双进程 Gloo TP2 用例扩展：

- 同配置接受；开关、bucket、native ABI 或 candidate 标签任一 rank 不同，两端均在额外 capture 前拒绝。
- 两端 Remote、部分 Local、all-copied-not-retired 对 batch 1/2/3/4/8 的决策均为
  `[true,true,false,true,false]`；retired 后交回原生 predicate。
- graph 判定不新增逐 decode all_reduce。此前慢 rank、取消 release 差异、GPU 未完成
  拒绝/重试、TP 一致资源退休等回归继续通过。

源码预检 **41 hooks / 101 core interfaces**；新增 native metadata/backend.forward 与
MQALayer 三个 prepare 方法的显式接口检查。isort、ruff format/选定错误和 diff 检查通过。

## RTX4060

用户授权的 ubuntu 工作机 RTX4060 Laptop 8GB，PyTorch 2.13.0+cu130。
隔离目录 `/tmp/dsv41-segments-20261031`，复用既有 venv，不安装包或修改部署。

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
PYTHONPATH=plugins/remote_attn PYTHONDONTWRITEBYTECODE=1 \
/workspace/remote-attn-cuda-20261028/venv/bin/python \
  test/manual/remote_attn/dsv41_segmented_graph_probe.py \
  --source-root . --output graph-result.json
```

DSV graph 探针结果：

- 原生 full/breakable backend 和 breakable primitives 使用当前仓库 source-loaded 定义；
  capture-mode get/set 函数使用 source-loaded 原生定义。
  CUDA capture/replay、**sgl_kernel 原生 weak tensor views**、output copies、stream event
  和多流 fork/join 均真实执行，不使用 strong-ref 替身。
- 三层小模型、四个 candidate labels × 原生 full buckets 1/2/4/8 共 **16 个 full keys**；
  Remote 仅三个 bucket，每 bucket **8 个实际 CUDA segment**，不乘以 variant 数。
  18 次 Remote 分段回放、
  **42 次真实 RemoteHub/TargetClient sparse RPC**，P status fixture 保持 Transferring。
- 四个 candidate labels 均在回放中覆盖，并在回放后恢复原生 scope。
  live metadata 和 top-k 在回放间改变；模拟资源逐项 Local 后停止对应 RPC。捕获阶段
  **零 RPC**。动态回调不在 CUDA capture 中执行，prepare 包含真实 CUDA side streams/joins。
- 分段回放与同一 fixture 的 eager 数值流程最大绝对误差 **0.0**。这证明该小型图回放流程
  的数值一致性，不代表完整 DSV checkpoint 精度。
- Remote 3/8 拒绝图回放；最终 Local 后 **8 次 native full graph replay**，不再调用 RPC。

兼容回归：

- Qwen 小型分段图探针：三层、每 bucket 4 segments，12 次 Remote / 4 次 full replay，
  22 次 toy RPC 边界调用、capture RPC 0，使用真实 weak views，验证共用 bank 框架未退化。
- 既有 DSV progressive 探针：8 次真实 RPC，逐资源 Local、pinned H2D/末页保护、四种 native
  cache 编码、GPU reader 取消 fence 和 P cache pin 回收继续通过。与独立 union reference
  最大绝对误差 **3.814697265625e-06**。

三组分别校验 **35 / 5 / 28** 份源码 SHA256，与提交前本地源码一致。

## 尚未验证

没有加载完整 DSV4.1 或 Qwen checkpoint、运行 SMG/P/D 服务或多 GPU NCCL。
4060 无法验证完整模型所需的 SM100 FlashMLA/indexer/compressor kernel。新 graph 探针的
三层模型、MQA prepare、metadata/indexer/carry 规划及 pool-manager 边界为小型 fixture；
真实执行部分是 native graph backend/primitives、CUDA side streams/bridges，以及基于
source-loaded native byte layout/pool accessor 的 sparse GPU reader 和 RemoteHub TCP。
Qwen 兼容探针的 RPC 是 toy 边界。既有 progressive 探针的 native status/top-k 也有 fixture。

未证明完整模型 token/logprob 一致性、实际 MLP/MoE 图回放、NCCL 行为、TTFT/ITL/吞吐收益
或跨故障恢复。QKV prepare 仍 eager；Remote sparse attention 仍为 Torch 参考计算，
batch RPC/kernel 优化不在本次范围。独立 graph pool 增加捕获时间与显存，应计入整模预算。
