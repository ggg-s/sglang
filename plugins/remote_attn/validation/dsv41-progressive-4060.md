# DSV4.1 提前接纳 / 逐资源 Local 验证

日期：2026-10-31。实现与原生启动见 [接入说明](../DSV41-REMOTE-EAGER-ZH.md)。
生产改动均在插件内，SMG/SGL 核心无新增修改。结果文件见
[四组 CUDA 输出与源码哈希](dsv41-progressive-4060-results.json)。

## 本地与真实 TP2

完整插件回归 **150 项通过，包含 49 项 DSV4.1**。保留最新 drain 修复及其三项回归；
本次新增 11 项 DSV 用例，以及既有
真实 TP2 subprocess 用例中的 progressive 扩展，覆盖：

- 原生 final envelope 的 state/aux 参数复用；indexer 乱序 TE 完成按集合计数，未齐全
  不发 ready；global 放在 state barrier 后，canonical final 不重发 D-owned 状态/metadata。
- 只有 P state/metadata + D index/scales GPU scatter + metadata room + 同代身份齐备才
  early commit；global 未 Success 也可接纳，receiver 和 metadata slot 保留。
- P HTTP response-only snapshot 等各 TP shard 的同代 barrier；不改变 live Req、sender
  Transferring 或分配所有权，不新增 response collective，不重复发终态。
- 连续 eager sparse decode 读取缺失 global；共享 consumer 依据 producer 逐资源 Local；
  混合请求不调用会读取未就绪 global 的 native whole-batch kernel，SWA store 不重复。
- V41/V41_NEW/V41_FP4/V41_FP4_NEW 的 data/scale 地址、页号重排、chunk 起点、32/33/34/39
  输入边界。末页 D 新行、odd carry 与 padding 保留；完整 immutable 页才允许整页 copy。
- GPU proof 未完成时拒绝最终退休，取消只排队 deferred free；native scatter、canonical
  completion 和最终 TP 命令通过后 metadata/receiver/native KV 各释放一次。
- 原生 normal scheduler 没有 overlap copy_done 时，记录实际执行 stream event；
  原生 overlap 路径继续使用 sampling/publication/readback 的既有 completion proof。
- direct reset/retraction 也先经过对称资源/lifetime drain，再修改 attempt/row；DMA
  失败 poison abort ACK、保留页面，不发布 sparse-ready、不杀共享 transfer worker。
- 单 physical entry 超过 native receive ring 容量时，接管前回退 native PD，cursor、页预留
  和 staging request 均不改变。稳定 Local 没有额外同步。

真实双进程 **Gloo TP2** 增加以下场景：

1. rank 0 的资源先复制完成，rank 1 未完成；32 次稳定观察无 all_reduce、无 Local 切换。
2. 慢 rank 完成后，同一有序命令、一次 5 元素最终 MIN，两端同时切资源 0；再独立切资源 1。
3. canonical completion 后，两端取消 release 状态不一致，稳定观察不发退休投票或 free。
4. release 一致但 GPU 尚未完成，最终投票拒绝；GPU 完成的状态边沿触发重试，最终各释放一次。
   总向量顺序 **[5, 5, 5, 5]**：两个资源、一次 GPU 未齐重试、一次最终 lifetime 退休。
5. 资源 ABI（kind/ratio/producer/layout/dtype/stride 等，不含指针）纳入既有启动一致性检查；
   两端 ABI 不同一致拒绝。原有 graph/cache/staging/mode 一致性检查继续通过。

合并后测试文件的格式化经 AST 等价检查，不改变测试逻辑。
源码预检：**41 hooks / 94 core interfaces**。isort、ruff format/选定错误检查及
`git diff --check` 通过。不存在新增生产核心源码修改。

## RTX4060 CUDA

已授权 `ubuntu` 工作机：RTX4060 Laptop 8GB、PyTorch 2.13.0+cu130、CUDA 13。
使用独立 `/tmp/dsv41-progressive-20261031`，复用既有 venv，无安装或部署变更。

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
PYTHONPATH=plugins/remote_attn PYTHONDONTWRITEBYTECODE=1 \
/workspace/remote-attn-cuda-20261028/venv/bin/python \
  test/manual/remote_attn/dsv41_remote_cuda_probe.py --progressive \
  --sglang-root . --output progressive-result.json
```

实跑结果：

- 将 D 的完整 global 前缀页清零，保留 D 新行/odd carry；P native status fixture 保持
  **Transferring**。真实 RemoteHub/TargetClient TCP + registry + CUDA sparse reads 连续解码，
  前两轮各 3 次 RPC；c1 scatter/最终命令后下一轮仅 c2 两次；c2 Local 后 RPC 不再增加。
  **总计 8 RPC**，两个资源各一次 5 元素 TP1 最终投票。
- 两种资源的真实 pinned H2D、CUDA scatter event 完成前不允许 Local；event 完成通知本身
  也不切换，只有有序命令/最终投票才切换。末页 D 新行/data/scale/padding canary 不变。
  末页使用 GPU 页号向量索引，避免标量 CUDA tensor 索引的潜在 host 同步；探针在同一
  scatter stream 预热后观测实际 pending event，避免 allocator 首次分配同步干扰。
- 四种 native 布局的 GPU slot 读取与独立 CPU quant reference 逐 BF16 bit 一致。
  attention 与独立 union-softmax oracle 的最大绝对误差 **3.814697265625e-06**。
  top-k 不修改、重复项保留、共享 producer/consumer、P KV/SWA 不变、pending half-pair 拒读。
- sparse barrier 前拒读；barrier 后允许 native Transferring 期间 RPC。
  与此区分，兼容 post-PD 对照仍在 native Success 前拒读、完成首轮 3 RPC 后恢复 Local。
- 关闭目标连接时 P 仍有真实 GPU reader 等待 CUDA event：tombstone 立即阻止新读，
  GPU 完成/ref 归零之前 FULL/SWA pin 保留，之后才释放；两种 pin 最终均归零。
- 原生 receive staging 回归：真实 Mooncake TCP、1 MiB pinned ring、12 条异构 native
  stride，P staging on/off 字节一致，实际 pending H2D、ring wrap、水位、取消 drain、
  已发/未发 allocation 释放及 deregister 后无写入均通过。
- P prefix cache/overlap CUDA 回归：4 plans、native indices event、normal/cancel/revoke、
  未提交 sampling 情况和 GPU read drain；稳定 middle→final 接管投票 0、空轮询投票 0。

四组分别校验 **27 / 26 / 31 / 21** 份源码 SHA256，与最终本地源码一致。
P overlap 探针的 hash key 使用 basename，其余使用仓库相对路径；JSON 保留各探针原始键。

## 范围与限制

4060 **没有加载完整 DSV4.1 checkpoint 或运行 SMG/P/D 服务**，也不能验证模型所需的
SM100 FlashMLA/indexer/compressor kernels。GPU 运行的是 source-loaded 原生 layout、pool
accessor、metadata、独立 quant reference，以及真实 sparse RPC、CUDA math/scatter/event、
registry 和 native Mooncake receive ring；top-k、canonical status、state/metadata 接纳输入
为小型 fixture。P overlap 的 IO 为队列边界 fixture。

原生 SWA/carry planner、metadata 提交、native receiver 生命周期与 TP2 命令在本地测试中
验证；没有把 fixture 称为完整 GPU compressor 接续或服务 E2E。完整 graph 恢复测试为
原生 predicate/ownership 门控，没有在 4060 捕获完整模型 graph。未跑多 GPU NCCL、
完整 token/logprob 精度或生产负载，未证明 TTFT/吞吐收益和跨故障恢复能力。
