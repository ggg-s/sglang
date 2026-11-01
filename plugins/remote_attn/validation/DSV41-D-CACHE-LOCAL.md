# DSV4.1 D cache 本地验证

日期：2026-11-01；基线 `7e7c8a1ec1`。仅本地测试，未上传源码或执行 GPU。
运行时只改插件，SMG、SGL 核心没有新增修改。原生启用方式及资源归属见
[接入说明](../DSV41-D-CACHE-ZH.md)。

## 行为变化

- DSV D 启用原生 decode radix opt-in 后，复用公共 Guard，允许设备内 FULL 或
  FULL/SWA UnifiedRadixCache。额外等待/活动 pin 只锁 FULL。H 必须页对齐，native
  matching 保留完整私有 SWA 活动窗口；HiCache、session/linker/其他组件仍拒绝。
- P final sender、总页数、ring 容量校验及 D scatter 都考虑非零 H，传输范围仅为
  缺失尾部；partial global 页仍保护 decode 新行与 carry。indexer/scales 完成、
  native SWA/carry/metadata 与相同 H 的 sideband 共同构成早接纳屏障。
- request-window 的 D SWA metadata 调用原生 page_indices，按本次 D 请求行地址
  生成；carry payload 不变。普通 paged SWA 保留原生增量 PD 行为。
- Remote sparse 直接读取 D 已缓存的选中位置，其余缺失位置继续通过 P；保留
  native top-k 顺序和重复项。逐 entry Local 不发布缓存，最终 lifetime 退休才
  调用原生 checkpoint/release，并在执行结果 TP 共识后恢复 Local full/overlap。
- 取消、未知 DMA 完成、retraction、cache 部分失败继续复用公共 Guard 的 sticky
  ABORT、排空、attempt 推进及隔离策略；无新增逐 decode / 稳定 chunk collective。

## 本地结果

**192 项插件测试 + 7 项共享 BCG 输出测试，共 199 项通过**。插件测试含70项 DSV
用例，其中9项 D cache 用例；真实 Gloo TP2 增加 cached progressive 场景。
源码预检46 hooks / 114 core interfaces；格式、静态和 diff 检查通过。

- 原生 UnifiedTreeCore、FULL/SWA component lock/receipt、split 和 unlock：D 的
  额外 pin 保住8 token FULL 前缀，旧 SWA 保护量为0；split 后 ABORT 不插入，锁计数
  最终归零。pending admission 8次重试不重新 lookup / collective，metadata 后移交
  正常 request lock；非对齐 H=7 在 metadata 前拒绝且不泄漏 pin。
- 原生 `_swa_tail_len`、`_match_prefix_and_lock`、`match_kv_cache` 配合真实 Radix
  树：17 token prompt、8 token window、8 token page，完整16 token tree hit 被原生
  限制为 H=8，保留9 token fresh tail。
- H=8、33 token 的原生 sender：final absolute start16 对应 suffix cursor1，总尾部
  4页，final发送3页；H 与 offer 不一致不能接管。换回基线 candidates，新增用例
  因其仍拒绝非零 H 而失败；当前实现通过。
- native ring 与实际 CPU byte scatter：suffix-relative 页3加回H后写物理末页，
  缓存 global/index 页不变；N=33 的 ratio2 未完成末行保持原值，index payload只
  写缺失页。修改 H 后拒绝 scatter。已有 interleaved/split scales、padding 和
  partial-page 回归同时通过。
- 原生 RequestWindow.row_loc/page_indices：slot2、24-row capacity、positions
  24/32/40 得到 `[6,7,8]` 页，carry 对象原样保留，不从 FULL 地址推断 SWA。
- sparse math 读真实 V4.1 FP4/FP8 byte layout：污染 P 的已命中前缀，D 的输出仍
  与完整可见 KV 的 union attention 一致；RPC只收到 `[15,15]`，保留重复 top-k，
  cached位置0不再远程读取。
- early commit → GPU pending → TP retirement：有缓存的正常发布和 ABORT 两种
  interleaving 都保留 metadata、FULL pin 及请求行，直到 native canonical/scatter
  和 GPU proof 完成；cache执行结果票只出现一次，完成后解除pin。实际初始化路径
  证明 DSV native radix opt-in 会安装公共 Guard 及有序 request receiver，包括
  scheduler overlap 关闭的对照。
- graph eligibility 与上述活动 lifetime 联动：Remote 1/2/4/8/16 保持分段资格，
  最终 cache commit 后切 Local full；期间及 Local轮询禁止额外 all_reduce。
  原有真实 LogitsProcessorOutput、batch切换和 sampler 字段隔离 CPU回归通过。
- 真实 Gloo TP2 cached progressive：先让单 rank resource ready，不能提前切 Local；
  按有序命令切两个 entry，取消意图到达时不同步回收，GPU/数据面完成后才最终退休。
  两 rank 的投票形状一致为 `[5,5,5,5,1]`，最后1为缓存执行结果票；32次稳定轮询
  不投票，最终无pin、无ABORT尾部发布。公共层已有单 rank发布失败隔离、stale
  intent、取消/retraction、未知DMA与 allocator去重/回收用例继续通过。

完整依赖环境可运行：

```bash
PYTHONPATH=python:plugins/remote_attn python test/registered/unit/plugins/test_remote_attn.py
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only --sglang-root .
```

本机轻量 CPU harness source-load 原生逻辑以避开 CUDA/Triton 导入；native GPU
event、TE、graph capture/replay 是边界替身。DSV hybrid checkpoint 执行端用接口
替身验证门控，真实 component receipts 已执行，公共 Radix checkpoint/dedup/free
使用原生 CPU 实现。没有完整 DSV UnifiedRadixCache 模型推理、真实 CUDA/NCCL/PD
或性能验收；目标 GPU 的共享前缀、取消、retraction、淘汰压力及输出一致性仍需验证。
