# DSV4.1 P 前缀缓存：本地与 RTX4060 验证

日期：2026-10-31。作为逐项对齐 Qwen3-8B 的第一项，P 开放设备内 FULL/SWA
前缀缓存，D 缓存保持关闭。未修改 SMG/SGL 核心；optimistic prefill、chunk、
batch 和原生 graph/overlap 配置保留。接入见 [DSV4.1 文档](../DSV41-LAYERWISE-ZH.md)。

## 本地结果

完整插件回归 122 项通过，含 24 项 DSV4.1 专项。新增覆盖：

1. 源码加载原生 UnifiedRadixCache/UnifiedTreeCore 的 lock/unlock、FULL/SWA
   acquire/release 和节点 split：独立读 pin 在 native request unlock 和 split
   后仍保护两组件，最终按原生回执释放；component protected/evictable 账本守恒。
   分配器、LRU 和服务进程是边界 fixture，未启动完整缓存后端。
2. 真实原生 DSV4.1 chunk/state/page planner 结合读租约：middle→final 不重复
   获取读 pin，GPU/入队账本/TP 未完成时不 checkpoint；回收后只执行一次
   checkpoint/finish，串行 final 路径也保留原生 SWA/carry completion。
3. 首次接管前已有 native IO 时不借用未知页历史；账本完成事件触发有序重确认。
4. 取消时尚无 native release 回调，先完成 GPU/IO/TP drain，再关闭独立读 pin；
   原生 request pin 保留到 native reset/free。D 启动仍拒绝开启缓存。

真实双进程 Gloo 扩展了原有证书交错用例，分别带不同 rank 的前缀长度运行：
接管、重确认、正常退休、取消和撤销的 collective 序列均保持
`[2,2,2,2,5,5,2,5,2,2]`，与缓存关闭版本一致；63 个稳定 chunk 无新增接管投票。
已成功退休的 A/B/C 读租约全部关闭；最后的资格版本错误用例共同拒绝，保留
未证明安全的 provisional pin，而不是宣称故障时正常释放。
Gloo 中缓存 API 是边界 fixture，原生混合锁回执由上述单机专项覆盖。

结果见 [TP2 证据](dsv41-p-cache-tp2-results.json)。预检 34 个 hook、83 个核心接口，
isort、ruff format/静态错误及 `git diff --check` 通过。

## RTX4060

用户授权的 `ubuntu` 工作机，独立目录 `/tmp/dsv41-p-cache-20261031`，复用
已有 PyTorch 2.13.0+cu130 / CUDA 13.0 环境，未安装包或修改已有部署。

```bash
PYTHONPATH=plugins/remote_attn python \
  test/manual/remote_attn/prefill_overlap_cuda_probe.py --prefix-cache \
  --output /tmp/dsv41-p-cache-results.json
```

使用真实 CUDA stream/event、pinned D2H 索引及插件 CacheGuard/SourceLease/
TP1 退休，验证 normal/cancel/revoke 三条路径都在 GPU/源读取完成后回收读租约；
传输期间 checkpoint 被延迟，最终 checkpoint 一次、finish 三次、读 pin 归零。
稳定 middle→final 的新增接管投票计数为 0，4 次 IO 回调，3 次退休共识；
完成后 sentinel 保持正确，空轮询不增加共识。

缓存后端是锁/checkpoint 边界 fixture，不是 GPU 上的完整 SGL/Rust 混合缓存；
IO 是线程队列，不是 Mooncake TE。原生混合锁/分裂由 CPU 专项测试验证。
结果与源码 SHA256 见 [CUDA 证据](dsv41-p-cache-4060-results.json)。
尚未加载完整 DSV4.1 模型，未完成缓存命中前后 token/logprob 一致性、多卡 NCCL、
TTFT、吞吐和显存收益验收；不能将本记录视为生产模型已验收。
