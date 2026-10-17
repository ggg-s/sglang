# 动态行号 / gate：RTX 4060 证据

测试源码以 `993fd124c5` 为基线，在工作机隔离 checkout 应用当前源码补丁，
完成验证后才提交。14 个修改 Python 文件（含新 kernel）的精确 SHA256 保存于
[metadata.json](metadata.json)，提交前逐项与本地源码核对一致。

[最终原始输出](regression.txt)：**363 passed、2 skipped，44.76 秒**。
CPU/Gloo 234 项，单卡 CUDA 129 项；跳过的两项为 NCCL 2 卡与 8 卡。
RTX 4060 Laptop 8GB、WSL2、Python 3.10.18、Torch 2.8.0+cu128、Triton 3.4.0、
nvcc 12.0、GCC/G++ 12。nvcc 与 Torch CUDA runtime 版本不同，记录为这次实际
通过的独立功能测试环境，不据此替代内网完整 serving 软件栈。

92 项 compact Graph 用例覆盖 B=1/8 的 K/G 矩阵及 B=16/G=3；执行实际 worker
函数体、原生 topk Triton kernel、当前 GPU 行号扫描与生产 gather/capture/replay/
scatter。模型/attention 计算是合成边界；额外导入辅助仅替换未执行的 full Graph
delegate 标记类型与测试本来就要替换的 attention 容器导入。
其余 37 项 CUDA 不使用这个 fixture 的替换分支，包含 10 项行号扫描用例。

辅助文件与上一轮的[原样快照](../dsv4-lease-4060/lean_cuda_plugin.py.txt)完全相同，
SHA256 也记入元数据；仅其所在的独立目录改变。它从 `repo/python` 直接加载
原生 topk 和 ForwardContext，actual worker body 由测试读取同一 checkout。
普通内网完整环境不要使用该辅助文件。

原始工作机命令（venv 复用上一轮的独立环境）：

```bash
cd /workspace/dsv4-dynamic-20261017/repo
export PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH
export PYTHONPATH=/workspace/dsv4-dynamic-20261017:experimental/hybrid-spec/src
export CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests -q --tb=short -rs
```

辅助文件位于 `/workspace/dsv4-dynamic-20261017/lean_cuda_plugin.py`，
原始日志为该目录的 `regression.log`。新位置复现时保持辅助文件与 `repo/` 同级。
验证后测试进程退出，GPU 显存 0 MiB，没有启动模型服务。

这轮未运行 DSV4 权重、真实 dsv4 attention、TP8 NCCL 或端到端吞吐/正确性评测；
[内网验收步骤](../dynamic-rows-gate-20261017.md)继续覆盖这些项目。
