# Hybrid Remote：P 端原生 prefix cache

日期：2026-11-07。修改基线：`fde7c621c8`，分支 `codex/remote-attn-pre-layerwise`。

## 修复与配置

原 `Controller.validate()` 将 `disable_radix_cache` 作为 P/D 的共同条件，
使 P 无法开启 prefix cache。现按角色校验：

- P 支持内置、device-only `(FULL, MAMBA)` UnifiedRadixCache。
- D0/D1 仍要求关闭 radix；D2D 请求行、KV 与 GDN slot 保持独立所有权。
- HiCache、外部 storage/linker、session 和第三方 cache backend 在启动时明确拒绝。

Hybrid P→D 走原生最终 PD，不在 P 提前提供 Remote 读服务。因此沿用原生
checkpoint/dedup、Mamba CoW、cached-prefix early send 与最终 PD 退休；不安装
只适用于 dense FULL 的 source-cache guard。D0 后续 D2D freeze 的 Remote lease
与 P 缓存是两个独立生命周期。

**P 不传 `--disable-radix-cache`；D0/D1 添加该参数。** 保持 page-size=1、
`--disable-overlap-schedule`、普通 FP32 GDN SSM 与 BF16/FP16 NHD KV。
该组合默认 `auto` 解析为 `no_buffer`，也可显式指定
`--mamba-radix-cache-strategy no_buffer`。

P 命中只减少计算，不减少 D 必须接收的 KV：D 未缓存前缀时，要发送全部 prompt
KV 以及完成后缀计算的工作 slot 状态。缓存 checkpoint slot 和工作 slot 不能混用。

## 验证结果

RTX 4060 Laptop 8GB，Torch 2.13.0+cu130。使用四层缩小的原生 Qwen3.5/GDN
配置、原生 HybridReqToTokenPool / HybridLinearKVPool / UnifiedRadixCache，
不是手工构造缓存命中结果。

`qwen_hybrid_prefix_cache_cuda_probe.py` 验证：

1. 第一个请求计算并通过原生 PD checkpoint/退休，将 FULL KV 与 GDN checkpoint
   留在树中；后续请求通过真实 `Req.init_next_round_input()`，先后命中 8、10 token。
2. 命中使用原生 Mamba match、独立工作 slot、deferred CoW 收集和 CUDA state copy。
   实际 Conv/GDN 续算与从零状态运行相同 native kernels 的冷路径**逐位一致**；
   续算不改写树内 checkpoint。
3. 原生 `maybe_send_cached_prefix_chunk` 发送命中前缀，最终 `_send_kv_chunk`
   发送剩余 KV 和 MAMBA 状态；累计完整覆盖 prompt，无重复和遗漏。
   最终 MAMBA payload 指向请求的真实物理工作 slot。
4. 原生 `process_disagg_prefill_transfer_queue` 在 Transferring 时保留 KV、请求行
   与工作 slot；只有 Success 后由 RequestRetirer/native release 释放。
5. 6 次真实 Mooncake TCP 单次提交，将所选 KV / Conv / SSM 经 pinned host staging
   传到接收缓冲并回到 CUDA，字节相同。

CPU `test_remote_main_contracts.py` **22 项通过**，包括 P/D 角色校验、不支持的
缓存策略拒绝、原生配置、GDN 分配与 D2D 所有权。新增 P 缓存启动用例在原代码上
失败，复现 `radix/overlap/MTP/LoRA disabled`，修复后通过。
启动夹具现在使用实际 RuntimeContext 的 scoped 配置，替代手工拼接配置 accessor。

既有 `qwen_hybrid_cuda_probe.py` 的 **11 项 CUDA 生命周期回归通过**：
GDN Conv/SSM 续算误差 0，Remote GQA BF16 误差 0.0009765625，Local 误差 0；
源/目标 GDN slot 1/2，18 次原生 Mooncake 提交。保留 producer stream 依赖、
新 suffix、状态先于 admission、取消、迟到写及完成未知的资源隔离断言。

机器结果与实现哈希：[qwen-hybrid-prefill-cache-4060.json](qwen-hybrid-prefill-cache-4060.json)。
原始日志：`/private/tmp/hybrid-prefix-results/results.tar.gz`。

## 边界与复现

模型权重执行、采样、metadata 与 sender 接口集成是显式测试边界。prefix-cache
探针的 cache/pool、Req 匹配、CoW、Conv/GDN kernels、原生 PD 页/状态选择、
发送队列退休与 Mooncake 字节传输均真实执行；没有启动完整模型 HTTP 服务。
这里的 GDN 对照不等于整个模型的 prefill/decode logprob 一致性验收。

没有 H20 内网入口，未测完整 27B、跨机 RDMA 或端到端 TTFT/吞吐收益。
当前验证覆盖默认 `no_buffer`；不将该结果扩展为 extra_buffer/HiCache 的验收结论。

```bash
PYTHONPATH=python:plugins/remote_attn \
python test/manual/remote_attn/qwen_hybrid_prefix_cache_cuda_probe.py \
  --output /tmp/hybrid-prefix-cache.json

PYTHONPATH=python:plugins/remote_attn \
python test/registered/unit/plugins/test_remote_main_contracts.py
```
