# Arctic 官方后端 RTX 4060 验证

结果：完整回归 **663 passed、2 skipped（337.74 秒）**，再通过 46 项新增
Arctic+dsv4（19.56 秒），合计 709 个不重复通过用例。CUDA 455 项、CPU/host/Gloo
254 项。测试结束显存 0 MiB，无 pytest 进程。

隔离 worktree：`/workspace/arctic-suffix-20261017/repo`，基于本分支
`53fdad7f27aa888dd5b440dc284f4ea2725120e4`。先 clone/pull 已发布代码，再上传本次
补丁验证。Arctic 从官方仓库 clone，固定
`9d4643881608178f0f58a33d1b61919ca246997f`，仅按官方 CMake 编译 suffix 模块，
没有修改上游源码，也没有安装/替换 vLLM 或 Torch。

- `metadata.json`：环境、每轮命令/结果、最终本仓库与上游源码 SHA256。
- `regression.txt`：完整回归原始输出。
- `dsv4.txt`：补充的 Arctic+dsv4 split 矩阵原始输出。
- pytest 轻量导入辅助沿用 [已保存快照](../dsv4-lease-4060/lean_cuda_plugin.py.txt)。

环境：Python 3.10.18、Torch 2.8.0+cu128、Triton 3.4.0、nvcc 12.0、
GCC 12（SA CUDA 扩展）、GCC 13（官方 Arctic CPU 扩展）、nanobind 2.9.2。
验证使用真实 `SuffixDecodingCache`、C++ tree、CPU 历史/H2D 与实际 CUDA Graph；
模型和 child attention 的执行仍为合成计算。EAGLE3/STANDALONE compact 用例
保留原生 Triton 多步 KV metadata/index kernel 验证。

完整命令（工作机仓库根目录）：

```bash
PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH \
PYTHONPATH=/workspace/arctic-suffix-20261017/ArcticInference:/workspace/arctic-suffix-20261017:experimental/hybrid-spec/src \
CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2 \
TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo \
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests -q --tb=short -rs
```

首次完整回归后，仅增加 `arctic_dsv4` 参数组合，单独运行该新增矩阵。
共享 gate 的 `_synchronize`/`select_step` 与基线 DeviceOverride 的 AST 对照相同，
仅将行数取得方式从 `len(self.slots)` 改为 `self.tokens.shape[0]`，供 staged CPU
候选复用。Gloo TP1/2/4/8 额外覆盖 staged gate；NCCL 2/8 卡在单卡机器上跳过。

这是运行时功能回归，不是完整真实权重/serving、TP8 NCCL、模型质量或吞吐验收。
真实服务不要加载 lean_cuda_plugin。安装与使用见 [接入指南](../arctic-suffix.md)。
