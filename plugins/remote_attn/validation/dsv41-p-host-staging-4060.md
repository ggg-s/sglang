# DSV4.1 P host staging：本地与 RTX4060 验证

日期：2026-10-31。接入见 [原生 P/D 启动文档](../DSV41-LAYERWISE-ZH.md)。
生产代码仅修改 `sglang_remote_attn` 插件，SMG 与 SGL 核心无修改。
P FULL/SWA 前缀缓存、P/D scheduler overlap、optimistic prefill、chunk 和 transfer batch 配置保留。

## 实现

- 在插件启动校验之后，为 MLA 路径补原生 `set_kv_buffer_tensors` 绑定。按原生
  global 注册顺序处理 compressed KV、indexer payload/scale，用原生 FULL page stride，
  不用 tensor 第一行大小代替 indexer 的压缩页跨度。
- 复用 Qwen 的 `CorePinnedStager` / `PinnedKVStager`。rank 共享一个原生 bounded
  pinned slot；启动时 reserve/prewarm。分组任务使用原生
  `transfer_fn=partial(manager._transfer_data, stage=True)`，其余发送规划/批次上限不变。
- 原生 worker 等生产者和页索引 event 后才做 D2H，slot 锁持续覆盖 D2H 和 TE；
  成功回调仍由原生 worker 发出。SWA、carry、aux 和 receiver Success 保持原生 final 路径。
- 普通 middle chunk 按原生策略发送；已有 `SGLANG_MOONCAKE_HOST_STAGING_ALL=1`
  时复用同一 slot。仅插件 HOST_STAGING=1 不强制改写普通 chunk 策略。
- poisoned slot 在既有 overlap 本地退休观察中变成错误通知，随后由有序命令触发 TP
  最终共识。错误时保留源 KV、缓存读租约和 registered slot，不直接释放或重试；
  稳定 chunk/轮询不新增 TP collective。接管证书和取消/retraction 协议保持原路径。

## 本地

完整插件回归 **125 项通过**，含 **27 项 DSV4.1** 测试。新增覆盖：

1. 真实原生池、页规划器和 staging 实现，仅 CUDA copy/TE 使用 CPU 字节边界：
   12 个条目、异构 native stride、乱序物理页、middle→final、奇数 carry，逐字节一致。
2. 23 字节 slot 强制跨 batch/条目切分，原生 transfer batch 上限保持为 2；
   SWA/carry 保留 native state planner，未误当成 global staging 数据。
3. P 原生 host-staging-all、缓存和 overlap 开启可同时校验；D host staging、
   receive staging、错误 stride/长度和任意自定义 transfer callback 均拒绝。
4. 原生 TE 返回失败后 slot quarantine、保留强引用、不 close、不直传重试；
   即使 pending callback 未完成，也能观察到 poisoned slot。
5. 真实双进程 TP2 Gloo：单 rank slot 失败发出 ordered abort，两端同序执行一次
   5 元素最终投票，均拒绝释放并保留 pending IO/KV；失败前 32 轮稳定观察无 all_reduce。
   原有稳定 63 chunk 接管投票为 0、缓存命中、撤销、取消和 retraction 用例继续通过。

源码预检：34 hooks / 83 core interfaces。isort、ruff format、ruff 选定错误检查、
`git diff --check` 通过。未增加或修改核心接口。

## RTX4060

机器：用户授权的 `ubuntu` 工作机，RTX4060 Laptop 8GB，PyTorch 2.13.0+cu130。
使用独立 `/tmp/dsv41-p-host-staging-20261031` 和已有只读 venv，未安装包或修改部署。

真实 staging 探针：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
PYTHONPATH=plugins/remote_attn PYTHONDONTWRITEBYTECODE=1 \
/workspace/remote-attn-cuda-20261028/venv/bin/python \
  test/manual/remote_attn/dsv41_host_staging_cuda_probe.py \
  --sglang-root . --output staging-result.json
```

- 同一 GPU 上两个真实 Mooncake TCP endpoint，实际 native byte planner、pinned
  allocation、CUDA 非阻塞 copy stream、D2H 和 TE 写入。
- 12 条目 native stride 为 `256/256/256/32/32/32/8/8/8/256/64/16`；
  GPU producer event 确实经历 pending，完成后数据逐字节一致，未选中页保持零值。
- 257 字节 slot 强制产生 **32 个 host batch**，所有 TE source 均位于该 pinned slot，
  每批字节数不超上限；多组及普通 native staging 复用同一 slot。
- 原生 small state 直接传输成功。注入 TE rc=-1 后 slot 被隔离，第二次发送未调用 TE，
  没有直接 GPU fallback。错误由注入产生，并非真实网络故障恢复测试。

另运行 `prefill_overlap_cuda_probe.py --prefix-cache`：真实 CUDA indices/event、
四个 launch plan、缓存读租约和 ordered TP1 retirement 通过，normal/cancel/revoke
三个回收均成功，无释放后写入；该探针的 IO 使用队列边界，未把它称为 Mooncake E2E。

完整结果和源码 SHA256 见 [JSON](dsv41-p-host-staging-4060-results.json)。
staging 的 24 份源码、overlap 的 17 份源码均与提交前本地文件逐项校验一致。

## 边界

此验证未加载完整 DSV4.1 模型、未运行完整 SMG/P/D 服务、未执行多 GPU NCCL。
未证明模型 token/logprob 精度、生产并发性能或 host staging 收益。
真实模型灰度仍需相同 revision 下比较 staging off/on 的 token/logprob、TTFT、
吞吐、P→D 接管间隙、取消/retraction 和资源回收；optimistic/chunk 配置保持一致。
