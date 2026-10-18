# 在线动态 compact verify — RTX 4060 验证

最终完整回归：**853 passed、2 skipped，277.63 秒**。跳过项为 NCCL 2 卡和
8 卡测试；结束后 GPU 显存 0 MiB。仅最终 v8 运行计入通过数量。

`metadata.json` 保存环境、基线 `9b88ffa222`、补丁 SHA256 和 25 项源码散列；
`regression.txt` 为完整输出。工作机独立目录：
`/workspace/dynamic-verify-fast-20261018/repo`。

覆盖 target 反馈标签、截断尾部、首个拒绝、neural/local/shared 来源、冷桶与探索、
完整周期异步计时、flush/idle/prefill、逐轮 GPU/CPU 选择与计数一致性，以及
候选位置超过 256 时的回退。新增六项初始化回归覆盖 draft 子类跳过 target 构造器的捕获流程，以及 attention backend 创建后的能力检查。混合批次、重排/缩批、gate=3/neural=2、
gate=8/neural=1、CUDA Graph token 档位切换沿用原生打包、验收、提交及 attention
算子执行；神经模型层为合成层。官方 Arctic suffix 和真实 Gloo 1/2/4/8 进程组也在回归内。

复现：

```bash
PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH \
PYTHONPATH=/workspace/arctic-suffix-20261017/ArcticInference:/workspace/dynamic-verify-fast-20261018:experimental/hybrid-spec/src \
CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2 \
TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo \
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests \
  -q --tb=short -rs --junitxml=../regression-v8.xml
```

轻量导入辅助沿用[已保存版本](../dsv4-lease-4060/lean_cuda_plugin.py.txt)，仅用于
合成模型测试。此次实际运行 Python 3.10.18、Torch 2.8.0+cu128、Triton 3.4.0。

`overhead.py` / `overhead.json` 是融合实现的 TP1 合成选择+反馈微测：batch=1/4/16
中位数约 232/323/226 µs，包含最终总数读回，不含 target 模型或 TP 通信。
这是 v5 专项之后的初步测量；后续版本调整了配置校验和 serving 初始化流程，微测涉及的选择/反馈算子未变化。
不能据此推算服务吞吐、H100 或 TP8 收益。

完整 SGLang 服务、真实模型 greedy 输出、DSV4 压缩 KV 与 TP8 NCCL 不在本次组件
验收范围。4090 服务验证单独记录；DSV4 TP8 仍需内网运行。
配置与测量口径见[动态验证说明](../../DYNAMIC_VERIFY.md)。
