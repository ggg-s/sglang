# Remote 最后 chunk：完整 forward 后发送

日期：2026-11-04。适用于 `HANDOFF_PRE_LAYERWISE=remote`，无需新开关。

Qwen 最后 chunk 使用一次原生 forward，保留 forward mode 和 split_index，不传入
split_forward_count。所有该 chunk 的发送任务在 forward 返回后提交，共享同一个
CUDA 完成 event。原有层组/区间/有界批次、metadata 提前发送、staging、逐层 Local
和缓存引用生命周期继续复用。历史前缀可继续与最后 chunk 计算重叠。
独立 layerwise 模式仍执行 split-prefill，按层组交错计算/发送。

DSV4.1 remote 不安装逐层 forward hook，串行路径保留原生 bounded-replay tail
CUDA graph；完整 forward event 覆盖该图的写入。既有 P overlap 的完整 forward
证明与 indexer/SWA/carry/metadata 屏障继续生效。PP=1 完整模型检查保留。
TP 启动签名加入新版计算边界，拒绝同 TP 新旧协议混用。

## 验证

- CPU/source-loading 的 TestRemoteAttn 全部 **264 项通过**，含真实 TP2 Gloo、
  checkpoint/插入、SWA 窗口与预算、在途 IO、overlap、quarantine 和退休回归。
- 新测试检查完整 forward 前没有任何 final-group 提交、共享完成 event、staging
  编号及 callback 保留、forward 失败时不发送部分 KV、独立 layerwise 继续交错执行。
- DSV startup 回归确认没有逐层 hook，尾窗图保持且仅发布一个 flush 完成证明。
- 4060 Laptop GPU，Torch 2.13.0+cu130：[原始记录](remote-final-forward-4060.json)。
  Qwen ABI fixture 在 BF16/FP16 下各只调用一次完整 forward；提交时 CUDA event
  尚未完成，独立 consumer stream 等 event 后能正确读取全部层数据。
  DSV ABI fixture 执行真实 CUDA tail graph，最终 event 覆盖 graph 写入。
- 49 hooks / 125 core interfaces 的 source-only 检查、Ruff E9/F63/F7/F82、
  git diff --check 通过。JSON 源码 SHA256 与提交前工作树逐项一致。

这是计算边界、真实 CUDA event/graph 和生命周期回归；未加载 Qwen/DSV 整模，
没有新增 Mooncake/RDMA 或 H20 TTFT 性能结论。传输协议未在本次修改中改变。

```bash
PYTHONPATH=plugins/remote_attn python test/manual/remote_attn/remote_final_forward_cuda_probe.py \
  --source-root . --output /tmp/final-forward.json
```
