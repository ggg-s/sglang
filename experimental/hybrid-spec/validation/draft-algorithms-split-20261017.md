# EAGLE3 / STANDALONE split 接入

本次基于 `7ee73928b35779ae9391969c6227b04f1bce2b45`，将 EAGLE/MTP、EAGLE3、
STANDALONE 的 split 与 split_no_skip 统一到 GPU 动态行号压缩和 gate=0…K。
默认仍为 full，不修改原生算法身份和 target verify/commit。

## 实现

- 移除 split 只允许 `EAGLE` 和两种 MTP 架构的拦截；仍要求 greedy、topk=1、
  单机 TP、device 历史及受支持的模型配置。
- EAGLE3/reduced-vocab 草稿的每次 local argmax 都先经过原生 hot-token 映射，
  再进行 gate、SA 选择及 verify-chain 写入。prefix 与 compact tail 使用相同目标词表。
  split 的 dense chain 不再因存在 token map 而退回 token-list 路径。
- STANDALONE 不依赖 target hidden state；compact tail 的 seed 可以为 None，
  不对其分配 staging 或执行 index_select。EAGLE3 则按实际 prefix 输出形状收集，
  不假设 seed 的拼接宽度与后续 recurrent hidden width 相同。
- 所有路径共享 gate 和 GPU 行号 kernel：G=0/1 不运行前缀 inner forward，
  G≥2 运行 G−1 次；G=K 没有 tail；G>K（例如 K=2/G=3）在配置阶段报错。
- Triton 多步 attention 和原生 dsv4 使用现有的独立 metadata/Graph pool；
  V4 的逐步写槽、Qwen3-MoE fused KV-store 的连续写地址约束保留。
- split API 升为 v5，worker、runner 和插件必须同时更新；通用 API v2、SA 布局 v3 不变。

仍拒绝其他 draft attention（例如 FlashInfer）、translated KV、KV canary、
memory saver、draft-extend DSA seed；其他 SWA/线性 attention 的模型范围未扩张。
取消 batch≤4 不保证任意批次的显存或性能，每个前缀仍捕获 1…B 大小的 tail。

## 4060 验证

完整回归：**505 passed，2 skipped，227.19 秒**；其中 CUDA 267 项，
CPU/host/Gloo 238 项。两项跳过为需要 2/8 卡的 NCCL 用例。结束后显存 0 MiB。

在工作机隔离 worktree 中先验证补丁，再提交推送。测试原始证据见
[4060 日志和源码哈希](draft-algorithms-split-4060/README.md)。

CUDA compact 矩阵包括 Qwen 风格 MTP、V4 风格 MTP、EAGLE3（有/无 token map）、
STANDALONE；每种覆盖 split/split_no_skip、B=1/8/16、K=1/2/3/8、G=0/1/2/3/K。
同一捕获 Graph 的多次 replay 变化命中行、raw batch 和 padding；除独立预期链外，
每轮还与相同候选和 gate 的 full worker 完整草稿链逐行比较，验证跳过行没有 tail 计算。

EAGLE3/STANDALONE 的 138 个 compact 用例额外使用仓库原生的
TritonMultiStepDraftBackend 元数据方法、build_inner_fb_view 和
`generate_draft_decode_kv_indices` CUDA/Triton kernel；紧凑批的 ragged KV 读取
与按请求表直接求和的独立 dense 参考比较。原生 topk、动态行号 kernel、CUDA Graph
capture/replay、gather/scatter 和私有池在 GPU 上实际执行。

模型、child attention 执行以及完整 serving 的导入边界仍用轻量替身；
不声称跑过真实权重、完整 EAGLE3/STANDALONE 服务、模型精度、吞吐或 TP8 NCCL。
这些仍需在具备模型和完整 SGLang 环境的机器验收。

## 运行

在已可运行该 checkout 的原生 EAGLE3 或 STANDALONE 命令上：

```bash
export PYTHONPATH="$PWD/experimental/hybrid-spec/src:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export HYBRID_SPEC_CONFIG="$PWD/experimental/hybrid-spec/configs/device-draft-split.json"
# 保留 --speculative-algorithm EAGLE3 或 STANDALONE
# --speculative-draft-attention-backend triton
# --speculative-eagle-topk 1
# --speculative-num-steps 8 --speculative-num-draft-tokens 9
# --max-running-requests 8 --cuda-graph-max-bs-decode 8
```

模板采用 K=8/G=2/B=8。更改 K 时同步更改 JSON 与两个 speculative steps/token 参数，
并保持 G≤K。DSV4 仍使用原有 dsv4 模板和 backend，不改为 triton。

完整依赖环境直接运行 `python -m pytest experimental/hybrid-spec/tests -q -rs`。
4060 的 lean_cuda_plugin 只用于隔离测试导入，不加载到真实服务。
做模型对照时固定原生/full/split 的参数，交替顺序重复运行；观察混合批次下的
`mixed_ct`、`tail_request_ct`、接受长度和输出正确性，性能结果独立记录。
