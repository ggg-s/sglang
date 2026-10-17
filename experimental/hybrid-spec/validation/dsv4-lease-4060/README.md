# RTX 4060 回归证据（2026-10-17）

测试对象：`a3a0d112b6ff02d48a57855afb7d68e3b964bf30`。
工作机从 `https://github.com/ggg-s/sglang.git` 的
`codex/hybrid-suffix-speculation` 分支 clone/pull，测试源码无未提交修改。

- [原始最终输出](regression.log)：218 passed、2 skipped，24.24 秒。
- [元数据](metadata.json)：精确命令、环境、版本、源码与辅助文件 SHA256。
- [轻量导入辅助原样快照](lean_cuda_plugin.py.txt)：仅用于 8 项合成模型 compact
  CUDA 测试；其他测试不走该 fixture 的替换分支。

CPU/Gloo 181 项、真实单卡 CUDA 37 项；NCCL 2 卡和 8 卡两项因只有一张卡而跳过。
8 项 compact 测试的模型/attention 计算本身是测试中的合成计算；辅助 fixture
另外替换未执行的 `FullCudaGraphBackend` 标记类与 attention 容器导入边界，
绕开完整 serving 依赖。原生 topk Triton kernel、ForwardContext、插件的
prefix/tail capture、GPU gather/scatter、私有 Graph pool、所有 mask 与 padding
检查仍实际执行。其余 29 项 CUDA 用例没有这些额外替换，包含 4 项新增 lease
回收及 Graph 查询恢复测试。

不代表真实 DSV4 attention、DSV4 模型数值正确性、TP8 NCCL 或性能已验收。
nvcc 12.0 与 PyTorch 的 CUDA 12.8 runtime 版本不同，记录为本次实际通过的
独立 kernel 测试环境，不据此替代内网的完整软件栈。

原始工作机目录结构：

```text
/workspace/dsv4-lease-validation-20261017/
  repo/                     # GitHub clone
  venv310/                  # 独立测试环境
  lean_cuda_plugin.py       # 本目录快照的原始位置
  metadata.json
  regression-a3a0d11.log
```

在该目录结构中复现：

```bash
cd /workspace/dsv4-lease-validation-20261017/repo
export PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH
export PYTHONPATH=/workspace/dsv4-lease-validation-20261017:experimental/hybrid-spec/src
export CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests -q --tb=short
```

辅助文件保留为 `.py.txt` 原样快照；复制到原始目录并命名为 `lean_cuda_plugin.py` 后使用。
辅助文件按原始位置解析 `repo/python`；在其他位置复现时，应保持这一同级布局。
完整 SGLang 环境的内网验收不加载此辅助文件，命令见
[验收步骤](../dsv4-lease-lifecycle-20261017.md)。

验证结束后检查：测试进程已退出，GPU 显存 0 MiB；没有启动模型服务。
