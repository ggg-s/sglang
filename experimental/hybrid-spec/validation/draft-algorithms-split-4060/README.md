# RTX 4060 split 三算法回归证据

结果：**505 passed，2 skipped，227.19 秒**。CUDA 267 项（含 230 个 compact
用例），CPU/host/Gloo 238 项。测试结束显存 0 MiB，无 pytest 进程。

工作机：`ai-workstation`，NVIDIA GeForce RTX 4060 Laptop GPU 8 GiB，WSL2。
隔离目录 `/workspace/dsv4-algorithms-20261017/repo` 从已推送的 `7ee73928b3` 创建，
上传本次源码补丁后验证，验证通过并核对源码哈希后才提交推送。

- [metadata.json](metadata.json)：软件栈、命令、测试结果、测试源码 SHA256。
- [regression.txt](regression.txt)：完整回归的原始 pytest 输出。
- 轻量 pytest 导入辅助沿用 [前轮快照](../dsv4-lease-4060/lean_cuda_plugin.py.txt)，
  其 SHA256 同时记录在 metadata.json；只影响 compact CUDA 用例的 serving 导入。

完整命令（工作机仓库根目录）：

```bash
PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH \
PYTHONPATH=/workspace/dsv4-algorithms-20261017:experimental/hybrid-spec/src \
CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2 \
TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo \
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests -q --tb=short -rs
```

230 个 compact CUDA 用例使用精确的仓库 worker 函数体、原生 topk、稳定行号 kernel、
实际 prefix/tail Graph 和 full 对照。138 个 EAGLE3/STANDALONE 用例额外加载
原生 Triton 多步 metadata 方法、build_inner_fb_view 与 KV index kernel，
将紧凑 ragged KV 读取与独立 dense 参考比较。
模型和 child attention 执行仍是合成计算；完整 SGLang 导入边界由辅助脚本隔离。
其余 CUDA 测试不使用这个导入替身，包含真实 GPU SA 查询、反馈与生命周期。

不是完整权重/serving 的模型验收，不包含 TP8 NCCL、模型质量或吞吐结果。
2/8 GPU NCCL 测试在单卡机器上跳过；Gloo TP1/2/4/8 仅验证 CPU 共识。
新配置模板在本地通过 HybridConfig.from_file 校验。内网真实模型直接运行测试，
不要启用这里的 lean_cuda_plugin。
