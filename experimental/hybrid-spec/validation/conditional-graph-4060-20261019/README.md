# RTX 4060 conditional graph 验证

**TP1 功能通过，调度微测未显示稳定加速。** 测试源码来自
`55444165099a4248e6ea534c0c51171ff06bcb85` 的完整 Git archive；
13 个改动文件已在工作站逐项核对 SHA256。本轮没有修改生产代码。

环境：`ai-workstation` / RTX 4060 Laptop 8 GiB / WSL2，驱动 592.82，
Python 3.12.3、Torch 2.14.1+cu130、CUDA 13.0、Triton 3.8.0。
独立目录 `/workspace/conditional-graph-20261019/repo`，新建 `venv`。
原有 Torch 2.8 环境保留；它没有条件节点绑定，另跑 CPU 回归确认默认路径兼容。

## 最终结果

|组别|通过数|耗时|证据|
|---|---:|---:|---|
|CUDA compact / 动态行压缩|280|103.02 s|[最终日志](cuda-regression-v2.log)、[JUnit](cuda-regression-v2.xml)|
|组件 / 动态 verify / 原生打包验收|352|88.43 s，含首次原生扩展编译|[最终日志](component-regression-v2.log)、[JUnit](component-regression-v2.xml)|
|STANDALONE conditional 专项|4|11.72 s|[日志](focused.log)，是上述 280 项的子集|
|旧 Torch 2.8 CPU 回归|114|1.42 s|[日志](legacy-torch28.log)，与新环境组件用例有重叠|

两组最终回归共 **632 项通过，0 失败、0 跳过**；专项和旧环境数量不再加进总数。
CUDA 组包含 **20 项 conditional 用例**，覆盖 Triton、dsv4、EAGLE3、
hot-token EAGLE3、STANDALONE 五类合成模型，以及 B=1/8、gate、
不同来源长度、零尾段、mixed mask、padding 和反复重放。
模型与 child attention 计算为合成实现，保留原生 topk、KV metadata / index
和真实 CUDA Graph / 行压缩 / gather / scatter；不代表完整权重服务验证。

独立 [dispatch_probe.py](dispatch_probe.py) 不使用 serving 导入替身，直接使用
生产 dispatch 和条件图，尾段包含 **FP16 256×256 GEMM**。
B=1/4/16/32 分别检查每种 tail 行数及两种行排列，包含缩批和 padding，
两种模式共 **460 轮**输出与独立参考一致；GPU 分支直方图和 mixed 计数一致。
CPU profiler 每 10 轮的 `aten::item`：普通 split **10 次**，conditional **0 次**。
原始结果见 [dispatch-probe.json](dispatch-probe.json)。

结束后 GPU **0 MiB、0%**，没有遗留测试进程。
[metadata.json](metadata.json) 保存结果、环境及源码/辅助脚本哈希；
[requirements.txt](requirements.txt) 保存隔离环境依赖。

## 调度微测

同一探针，预热 20 轮，3 次 × 100 轮取中位数，**两种模式每轮末尾均同步**。
B=1 的测量输入全部命中 SA，其余 B 的一半行进入 neural tail。

|B|普通 split µs/轮|conditional µs/轮|
|---|---:|---:|
|1|82.84|174.95|
|4|108.18|101.67|
|16|122.08|133.91|
|32|118.47|140.05|

这些是小 GEMM 加调度的单次短微测，包含条件节点、统计和最终同步成本，
受启动顺序、频率及 host 噪声影响。**消除 `.item()` 不等于必然提速**；
不能把这些数值换算成真实模型吞吐，也不能外推 TP2。
保持开关默认关闭，实际服务是否开启仍需模型负载 A/B。

后续 [开销消融](../conditional-graph-overhead-4060-20261019/README.md) 用 GPU
event 与预计算 predicate 对照确认 IF 树为主要新增成本。上表 B=1 的
174.95 µs 未在交错重测中复现，不能视为该 batch 固有的大幅回退。

## 初次失败与处理

- 首次 CUDA 大矩阵有 2 项普通 split 失败、278 项通过，20 项 conditional
  均通过。异常为测试反复替换模块后，Triton `CompiledKernel.__del__`
  在后续捕图期间卸载 module，导致 capture invalidated；
  [初次日志](cuda-regression.log.gz) 保留错误及 unraisable warning。
  第二次加载 [capture_gc_plugin.py](capture_gc_plugin.py)，在测试之间回收，
  测试执行期间暂缓自动循环 GC，280 项通过。该辅助只影响测试生命周期，
  不替换 CUDA 算子，也没有修改生产捕图或 Triton。
- 首次组件回归有 26 项失败、326 项通过，起因是新环境缺 Ninja，随后加载不到
  尚未构建的 `.so`；见 [初次日志](component-regression.log.gz)。
  隔离安装 Ninja、CUDA nvcc/crt/nvvm 13.0.88、CCCL 13.0.85，
  使用新建的 `cuda13` 目录、gcc/g++-12、`MAX_JOBS=2`、架构 8.9 后，
  原生扩展成功编译，352 项通过。

旧工具链 `/usr/bin/nvcc` 为 CUDA 12.0；新环境的 `CUDA_HOME` 指向
`/workspace/conditional-graph-20261019/cuda13`，其 bin/include/nvvm 链接到
新 venv 的 `nvidia/cu13`，lib64/libcudart.so 链接到对应 libcudart.so.13。
[安装日志](compiler-install.log) 和 [运行脚本](run-component-v2.sh) 保留配置。

合成模型导入辅助 [lean_cuda_plugin.py.txt](lean_cuda_plugin.py.txt) 与前轮相同，
只用于隔离本来就由合成计算替代的 serving 导入边界，不用于真实服务。

## 复现与边界

工作站最终运行命令分别保存在 [CUDA 脚本](run-cuda-v2.sh) 和
[组件脚本](run-component-v2.sh)。独立探针可运行：

```bash
cd /workspace/conditional-graph-20261019/repo
PYTHONPATH=experimental/hybrid-spec/src \
  ../venv/bin/python ../dispatch_probe.py
```

未覆盖 TP2/NCCL 条件分支、Arctic、完整模型权重服务、生成质量和服务吞吐。
单卡通过不能替代多卡 collective 分支一致性的实机验收。
