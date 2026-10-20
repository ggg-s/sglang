# 混合 tail 修复与 gate 小规模对照：4090，2026-10-20

## 根因与修复

真实模型复现时，原始批次长度为 `[792,655,408,295]`，来源 mask 为
`[true,true,true,false]`，只有原始第 3 行需要 neural tail。
两条路径的首 token 都为 2617，位置都是 295，写入槽位都是 4207；
但 no-skip 的下一候选为 4612，compact 为 37498，首步 logits 已明显不同。

额外记录确认：compact 行号 `[3]` 被暂存到 eager registry 的第 0 行，
而 registry 与外层 draft graph 共用 `seq_lens/req_pool_indices` 存储。
外层长度因此变成 `[295,655,408,295]`、请求行号变成 `[3,1,2,3]`，
原始 positions 却仍为 `[792,655,408,295]`。再次 replay no-skip 时，
Triton KV indices 的打包长度与 indptr 的位置口径不一致，neural 行读错 KV。
这是共享输入暂存覆盖导致的错误，不是靠调整 gate 或容忍 argmax 误差解决。

修复位于 `python/sglang/srt/model_executor/runner/eager_runner.py`：
预规划 metadata 的 decode 直接使用调用方的输入地址，不再调用
`load_batch()` 复制到共享 eager registry。需要规划或允许重规划且已过期
的普通 eager decode 仍暂存输入、初始化 metadata；PDMUX 行为不变。
没有全局打开 `SGLANG_EAGER_INPUT_NO_COPY`，也没有绕过 ModelRunner forward。

## 回归证据

- 新增 `test_preplanned_decode_preserves_outer_graph_request_mapping`。
  旧代码稳定失败于外层长度被改写；修复后完整文件 **10 passed、3 subtests passed**。
  fresh/stale 分支同时检查暂存与 metadata 重新规划仍生效。
- 同样的 8 条真实模型请求（4 条串行、4 条同时），审计 **40 个固定状态、
  64 个请求行**。每状态 split/no-skip 各回放三次，候选及来源 mask 全部一致，
  每次审计还检查外层长度和请求映射保持不变。
- 其中 C=4：2 个整批 SA 状态、4 个 3-SA/1-neural 混合状态、
  2 个 1-SA/3-neural 混合状态。C=1：21 个 SA、11 个 neural 状态。
- 该审计含额外重放与诊断拷贝，**不用于吞吐结论**。

原始证据：`pre-fix-unit.log`、`post-fix-unit.log`、`audit.jsonl`、
`fixed-audit-summary.json`；修复前两次诊断见压缩的 mismatch JSON。
脚本中的 a66968470a 是基准 commit，运行时还包括本次 EagerRunner 修复及上一轮
续写歧义 gate 补丁，不能将本次结果归到未修改的该 commit。

## 性能对照口径

- RTX 4090 48GB；现有 Qwen3-8B target / Qwen3-0.6B draft，FP16、TP=1、
  STANDALONE、Triton attention、K=8、gate=1、split、static verify。
- 两臂都使用修复后的 tail，只切换 `sa_require_unique_continuation`。
  动态 verify 关闭，模型/数据/部署参数完全相同，诊断插件关闭。
- 同一组公开 code-edit 数据的前 **24 条请求**，并发 **4**，每条固定
  **192 个输出 token**（greedy、ignore_eos），每臂 **2 轮**。
  共 96 条计时请求、18,432 个输出 token；不扫参数或并发。
- 每臂先单独预热；每轮 flush 后再读 before、执行负载、读 after。
  计时不包括启动、预热、flush 或 server-info 读回，包含 prefill、decode 和
  客户端批次间 HTTP 开销，是这段固定工作负载的端到端 output tok/s。
- 接受长度为各请求 `completion_tokens` 总和除以 `spec_verify_ct` 总和，
  包含 bonus。另存排除 bonus 的 correct-drafts/verify。
  诊断只使用 before/after 增量，逐轮检查计数器非负。
- `analyze_ab.py` 从逐请求原始响应与 before/after 生成汇总，同时核对输出文本。

## 对照结果

| 策略 | 接受长度（含 bonus） | 端到端 output tok/s | 两轮 throughput |
|---|---:|---:|---:|
| 旧 gate=1 | 6.8166 | 486.85 | 482.80 / 490.96 |
| 新 gate=1 + 续写歧义过滤 | 6.8674 | 486.92 | 485.92 / 487.92 |
| 变化 | **+0.745%** | **+0.015%** | +0.645% / −0.618% |

两轮的接受计数各自完全重复：每轮旧 gate 676 次请求级 verify，新 gate
671 次，输出均为 4,608 token。两臂的 24 条输出文本逐条一致，各臂重复轮次
的文本也完全一致。两臂合计 18,432 个计时输出 token。

每轮来源计数（两轮相同）：

| 指标 | 旧 gate | 新 gate |
|---|---:|---:|
| SA 请求行 / 总 verify 请求行 | 411 / 676 | 402 / 671 |
| SA 选中行的正确草稿均值，不含 bonus | 7.404 | 7.567 |
| neural 请求行 | 265 | 269 |
| 全跳过 / neural tail 调用次数 | 50 / 173 | 44 / 177 |
| 混合 batch 次数 | 101 | 107 |

新 gate 的候选筛选方向有微小正向信号：选中 SA 的平均正确草稿数上升，
总请求级 verify 次数减少。但它同时减少整批跳过，增加 neural tail 和混合
batch 调用。端到端吞吐几乎相同，两轮变化方向相反，**没有可辨的吞吐收益**。
这些计数描述各策略自己的轨迹，不能把两个计数之差当成逐候选拒绝数，也没有
单独测量新 SA 查询检查耗时来做时间归因。

结论：**保留 tail 正确性修复，续写歧义 gate 继续默认关闭，不追加大实验。**
此结论限于本次 Qwen3 / 4090 小样本，不外推到 DSV4 / H100。

机器可读汇总见 `analysis.json`。原始响应、日志、配置和实际执行脚本见
`evidence/` 或 `evidence.tar.gz`。归档 SHA-256：
`027f8f68fb953dc2023bff4cf2f442266a278027cfc9beae3e24902e4de2cff4`。
`evidence/production-files.sha256` 的五个生产文件与本地当前源码逐一匹配。
所有服务已退出，稳定后 GPU 占用 **0 MiB**。

仓库保存压缩的完整原始证据；`evidence/` 为本地解压目录，不重复提交。
在本目录重新生成统计：

```sh
mkdir -p evidence
tar -xzf evidence.tar.gz -C evidence
python analyze_ab.py
```
