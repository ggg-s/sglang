# Qwen3 / DSV4.1 Remote 分段图：batch 8/16

日期：2026-11-01；基线 `68c2c99f7f`。按用户要求仅做本地测试，未上传源码或执行 GPU 验证。

## 接入行为

- 共用插件策略扩为精确 batch **1/2/4/8/16**。只为原生 capture 列表中已有的桶额外
  捕获 Remote bank；保留原有 1/2/4 最小要求，不修改 max-running 或原生桶列表。
  因此 8/16 需要在原生配置中存在；小实例仍可只配置 1/2/4。
- Qwen 每个可用桶一套 Remote bank；DSV 的四种 native candidate variants 仍复用
  同 batch 的一套 Remote bank，不增加为每 variant 一套。启动共识加入允许的 Remote
  bucket 策略，在任何额外捕获前拒绝各 TP rank 插件策略不一致。
- RPC、merge 及 DSV 动态 sparse/state 边界不变；未覆盖或未捕获的 Remote shape 仍
  eager，不 padding。全部 Local 且租约按既有协议退休后，使用原生 full graph 的全部
  keys/variants，包括 32 等更大的原生桶；不重新捕获，不新增逐 token collective。
- Mooncake MHA 继续复用已有最多4行一帧的拆分循环；DSV 保持逐行 sparse 调用，
  未扩大数据面 slot 或改变取消、缓存保护、GPU proof 和 TP 退休协议。

运行时只改 `segmented_graph.py`。SMG、SGL 核心和模型计算没有修改。8/16 额外 Remote
捕获会增加启动时间及 graph 显存，具体成本与收益需要目标 GPU 测量。

## 本地测试

**173项插件测试 + 7项共享 BCG 输出测试通过，共180项**；插件测试含61项 DSV4.1
及真实 Gloo TP2 检查。源码预检为44 hooks / 108 core interfaces；格式、静态和 diff 检查通过。

- 扩容前，更新后的 Qwen 和 DSV 用例都因 Remote keys 仍只有 1/2/4 失败；修改策略后通过。
- Qwen：捕获原生 1/2/4/8/16/32，Remote 仅前五桶；各桶消费 live 请求，捕获零 RPC。
  3/12/32 Remote 拒绝图执行，Local 对六个原生桶恢复 full，零 Remote RPC/event。
  另验证只提供 1/2/4 时不会伪造 8/16 图。
- Qwen 共用的原生 BCG 输出方法：真实 `LogitsProcessorOutput` 使用16行缓冲，
  16→8→4→1→2→16→8→1 回放，验证数值、形状和 sampler 字段隔离。
- DSV：原生 full/BCG 输出方法与原生 decode 输出处理走真实源码；Remote 共8次，
  序列16/8/4/1/2/16/8/4，Local 共9次，序列16/8/4/1/2/16/8/32/32。
  logits/hidden states、外壳独立、稳定 tensor 地址及9个 sampler 字段隔离通过。
  四种 candidate labels 在8/16同样覆盖，只建5套 Remote bank，保留24个 full keys。
- 新增启动回归证明最小桶要求不变、缺少2仍被拒绝，以及 Remote 策略不一致在创建
  backend 前失败。真实 Gloo TP2 覆盖策略 mismatch、8/16 routing、Local 恢复及零逐步投票。

CPU 测试中的 graph 执行是边界替身；没有真实 CUDA capture/replay、完整 checkpoint、
FlashMLA、NCCL 或性能验收。两套 CUDA 探针已更新到五个 Remote 桶，DSV 小缓存扩至
32行并准备16个 Remote targets；此次没有运行这些 CUDA 探针。

线上配置与核对方式见 [单 pair](../SINGLE-PAIR-ZH.md) 和
[DSV Remote](../DSV41-REMOTE-EAGER-ZH.md)。历史4060结果保留原有源码版本与验证范围。
