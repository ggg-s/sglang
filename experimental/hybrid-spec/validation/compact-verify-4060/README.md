# Compact verify — RTX 4060 组件验证

最终完整回归：**813 passed、2 skipped，264.24 秒**；两项跳过分别为 NCCL 2 卡/8 卡测试。
结束后 RTX 4060 显存为 **0 MiB**。本地 CPU 回归另有 259 项通过。

最终结果、环境与源码 SHA256 见 `metadata.json`，完整输出见 `regression.txt`。
仅最终完整运行计入通过数量；探索性失败和中断运行不计入。

基线为 `049db08e4d`，工作机独立 worktree：
`/workspace/compact-verify-20261018/repo`。先 fetch 已发布分支，再应用本轮补丁。
本地源码、配置和测试与工作机逐文件校验；提交前核对全部 54 项 SHA256。

覆盖：

- SA=8，neural=1/2/3/8；gate=0…8，包括 gate=3/neural=2、gate=8/neural=1。
- 实际 worker draft_forward、原生 GPU 动态行号和 split graph：MTP、EAGLE3、
  映射词表、STANDALONE、hidden_states=None，以及官方 Arctic suffix 来源。
- 混合 batch、全 SA、全 neural、重排、缩批、Graph padding，batch=1/2/4/8/16。
- 原生 target 打包 kernel 的 anchor、候选、position、KV 写入列，以及保留原完整 KV plan。
- 固定地址 metadata 刷新、同一档位内长度/顺序变化、不同 token 档位切换，
  4/9/13 等非整窗口大小；模拟 8-token 对齐的真实单 GPU 图回放。
- 原生 Triton attention 与独立 PyTorch softmax attention 数值对照，含 prefix KV、
  ragged query、空 padding 行和反复切换旧图。Graph 外缓冲区由测试显式持有。
- 原生 greedy verify 和 commit kernel：拒绝位置、bonus、全局 accept index、
  seq_lens 更新、draft-extend select_index；补零尾部即使碰巧匹配 target，也不能越过有效链。
- split 的 host neural 行数复用于 verify 选档；不能复用时读回单个总数。
  诊断区分逻辑 token、实际 target 行数和未压缩窗口，padded 对照计入实际图 padding。
- 保留已有 SA 生命周期、共享池、overlap/chunked、CPU/Gloo 和单卡 CUDA 回归。

复现（工作机仓库根目录）：

```bash
PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH \
PYTHONPATH=/workspace/arctic-suffix-20261017/ArcticInference:/workspace/compact-verify-20261018:experimental/hybrid-spec/src \
CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2 \
TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo \
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests \
  -q --tb=short -rs --junitxml=../regression-v6.xml
```

轻量导入辅助沿用[已保存快照](../dsv4-lease-4060/lean_cuda_plugin.py.txt)，
仅用于合成模型测试；真实服务不要加载。新 compact verify 用例在测试内隔离完整
serving 依赖；CUDA Graph、打包/映射、接受/提交以及 attention kernel 为原生实际执行。

这不是完整 SGLang 服务或 DSV4 权重验证。神经层使用合成模型；DSV4 child metadata
由形状/地址模拟检查，实际 DSV4 压缩 KV 和 TP8 NCCL 未在这张单卡上执行。
8 对齐测试证明单 GPU 的档位/打包行为，不能当作多卡 SP/NCCL 验收。
真实模型 greedy 输出、DSV4 TP8、长上下文压缩边界、取消/retract 与性能由内网补测。

内网配置与对照方法见[接入说明](../../README.md)。保持 SA8/neural3/gate1 相同策略，
比较 padded 与 compact，另外保留原 SA8/neural8 基线；覆盖 c=1/4、gate=3/neural=2、
全命中/无命中/混合 batch、上下文 127/128/129 边界和停止截断，并交替顺序重复测量。
