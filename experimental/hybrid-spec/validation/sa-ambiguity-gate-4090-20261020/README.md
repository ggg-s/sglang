# SA 续写歧义 gate：短回归，2026-10-20

实现 `sa_require_unique_continuation=true`，默认关闭；仅接受 device GPU SA、
完整 K-token 候选。动态 verify 保持关闭，neural gate 宽度不变。

## 判断依据

原来的 gate=1 只要求 SA/neural 的首候选相同。例如历史中同一个上下文
分别出现续写 `[A,B,C]` 与 `[A,X,Y]`，首 token 一致无法区分这两条续写。
新开关检查匹配的 SAM 状态：后续 K 步中出现多个出边则拒绝本轮 SA；
全局检索池里同样长的有效匹配也必须给出相同的 K-token 候选。本地历史
在匹配长度平局时仍参与冲突检查，不能因原来的本地优先规则绕过。

更短的上下文不否决更长的匹配。不可用、代际失效及没有完整候选的 donor
不参与比较；来源内已观察到的短尾分歧仍算冲突，EOF 本身不作为不同 token。
歧义发生时整行使用原生 neural，保持固定验证长度，不选择更短上下文规避拒绝。
原来的匹配长度、可用续写长度和来源顺序仍决定候选；没有按该规则预测正确概率。

检查在现有 SA 查询 kernel 内执行，增加状态遍历与候选比较，不增加模型
forward、CPU 回读或 TP collective。每个参与比较的来源至多增加 K 步遍历；
这不代表查询零成本。

## 验证

- 本地 `test_core.py`：**109 passed**，包含不支持配置的拒绝检查。
- 远端 RTX 4090：`test_native.py`、`test_shared_parallel_cuda.py`，
  **14 passed in 85.31s**，含本轮 CUDA 扩展编译时间。
- 共享 C++ 状态机对每次 append 后的最长重复后缀运行独立文本穷举 oracle：
  检查所有已观察到的续写是否与一个完整 K-token 续写相容，覆盖 K=1…4、
  克隆状态、内联及额外分叉。
- CUDA 覆盖第二候选分叉、来源平局、不同线程及跨步扫描、来源代际失效、
  更长上下文优先、Graph padding、历史重建和重复 Graph 回放。
- 原有关闭开关的跨来源文本 oracle 回归通过。
- 未启动模型服务，未下载模型或安装依赖。结束时 GPU 占用 **0 MiB**。

远端命令（已激活现有 SGLang 环境，目录独立）：

```sh
PYTHONPATH=src timeout --signal=INT 240 python -m pytest -q \
  tests/test_native.py tests/test_shared_parallel_cuda.py \
  --disable-warnings --maxfail=1
```

原始输出见 `short-regression.log`；`tested-files.sha256` 是远端实际测试文件
的 SHA-256，与当前实现及上述两个测试文件核对一致。本地 C++ 工具链首次
运行受临时目录沙箱限制，故 native oracle 在远端执行并通过。

## 限制与启用

```json
{
  "history_mode": "device",
  "backend": "gpu_sa",
  "partial_drafts": "fallback",
  "num_neural_gate_tokens": 1,
  "sa_require_unique_continuation": true,
  "dynamic_verify": false
}
```

这是一段配置增量，需要合并到已有完整配置。
无歧义不等于正确：单次出现的历史、共同复制同一个错误都可能通过。
有歧义也不证明 neural 一定更好。本轮只验证过滤语义和 CUDA Graph 执行，
**没有真实模型接受长度或端到端性能对照，因此不改变默认策略**。
先前 split 混合批的候选差异未由本改动定位或修复，不可据此宣称混合 tail 等价。

后续已定位并修复共享 eager 输入暂存覆盖外层请求映射的问题，并完成新旧 gate
小规模模型对照；见 [tail 修复及对照结果](../tail-fix-gate-ab-4090-20261020/README.md)。
接受长度约 +0.75%，吞吐无可辨收益，继续默认关闭本开关。
