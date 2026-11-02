# H20/Hopper Remote：V4 KV 适配与 4060 CUDA 正确性验证

日期：2026-11-02。分支：`codex/remote-attn-pre-layerwise`。
完整结果、测试范围和逐文件源码 SHA256 见 [JSON](dsv41-hopper-v4-4060-results.json)。

## 实现

- 保留原生 `KVLayout.V4`，不修改 KV pool 的架构选择或解除 V41 内核限制。
  Remote reader 按原生页格式解码 448 维 e4m3 NoPE、64 维 BF16 RoPE、七个
  UE8M0 scale；忽略第八个 scale padding 字节及页尾 padding，跟随原生 V4 的
  FP32 FTZ 语义。只读取选中 slot，保留顺序和重复项；异常 shape/stride、越界
  slot 和非有限 payload 拒绝执行。V41 FP8/FP4 路径保留。
- 共享启动校验允许原生 V4/V41 SWA，拒绝 FP4 SWA 与 uniform-FP8。D 先完成 TP
  实际 layout 一致性检查，再进入后端创建；不支持的布局不会进入捕获。
- Hopper 状态更新复用核心现有分派：C1 非融合写入、C2 独立 pair-pooling/ring
  kernel、非 SM100 index-store。没有开启 Blackwell fused compressor 或改写
  compressor 权重布局。fused-compress/gate 策略加入已有图启动 ABI 共识。
- 复用 native projection/state 边界，QKV 投影入图，metadata、SWA、carry、
  indexer/top-k、sparse RPC 与 merge 留图外；按 live batch 裁掉 bucket padding。
  收尾 scatter 的 V4 data/scale 几何已覆盖部分页保护回归，不覆盖 D 的新增行。

## CPU

完整插件契约回归 **219 项通过**（42.172 s），含 TP2 Gloo。测试启动器只将通用
`CustomTestCase` 替换为 `unittest.TestCase`，避免导入无关 serving 依赖；使用真实
torch、原生源码加载方法和已有 CUDA/TE 边界替身。

新增覆盖 V4 native reader oracle、非连续物理页、重复/空/非法 slot、BF16 pool
view、signed zero、scale 0/255、NoPE/RoPE 非有限值及 padding；V4 sparse union
与独立原生反量化 + softmax 比较；混合 batch、奇数 C2 前缀、producer 共享、
progressive 转 Local、部分页 scatter、Hopper compressor 分派和串行状态边界。
格式、isort、仓库 pre-commit 选定的 ruff 检查及 `git diff --check` 通过。

## RTX4060

用户授权的 `ubuntu` 工作机，RTX4060 Laptop 8GB，PyTorch 2.13.0+cu130。
独立目录 `/tmp/dsv41-hopper-20261102.XvXJXL`，复用
`/workspace/remote-attn-cuda-20261028/venv/bin/python`，无依赖安装或服务部署修改。

原生内核探针 `dsv41_hopper_cuda_probe.py`：

- 实际运行原生 Triton V4 quantize/store/dequant；page size 1/4/64/128。
  写入后整个 cache（含未写行与 padding）逐字节匹配独立 torch writer。
  plugin sparse reader 同时逐 BF16 位匹配原生 CUDA dequant 与独立 torch oracle。
- 实际运行 C2 pair-pooling kernel 连续 12 步，使用有 stride 的 interleaved
  KV/score ring，覆盖奇偶步、ring wraparound、请求 0 的虚拟行与备用行保护。
  pooling 结果逐位一致，carry/ring 写入和 slot/group-position 完全一致。

V4 progressive 探针 `dsv41_remote_cuda_probe.py --kv-layout v4 --progressive`：

- 真实 RemoteHub TCP、CUDA sparse read/math、pinned H2D scatter 与 ordered
  per-resource retirement；8 次 remote call，最大绝对误差 **0**。
- 未通过 sparse barrier、读取未完成 half-pair 均拒绝；global 尚未完整到达时
  D 已计算，末页 D 新行保留，P KV/SWA 不变；取消时保留进行中的 GPU reader。
- 原生 top-k 不变，共享 consumer 正确；资源转 Local 后对应 RPC 停止。

V4 串行 prepare 分段图 `dsv41_segmented_graph_probe.py --kv-layout v4 --serial-prepare`：

| QKV 投影融合 | KV 来源 | Remote replay | RPC | capture RPC | 最大绝对误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| 关 | FP32 | 78 | 582 | 0 | 0 |
| 关 | BF16 | 78 | 582 | 0 | 0 |
| 开 | FP32 | 78 | 582 | 0 | 0 |
| 开 | BF16 | 78 | 582 | 0 | 0 |

每组使用真实 native full/breakable CUDA bank，Remote buckets 1/2/4/8/16，
24 个原生 full shapes、四种 candidate variant；包含 batch 3/5/6/7/13/14/15 的
padding，live metadata/top-k/SWA/carry 检查、8 次 full replay、恢复 Local、
原生输出包装及 sampler 字段检查。replay 不重复调用 host QKV projection。

六份 CUDA 结果中记录的 **49 份源码**与本地逐项 SHA256 一致。

## 重现与边界

命令见 [运行说明](../DSV41-REMOTE-EAGER-ZH.md)。在有对应 CUDA 依赖的环境中，
使用 `PYTHONPATH=plugins/remote_attn`，先运行 V4/C2 原生内核探针，再运行 V4
progressive 和分段图探针；图矩阵额外组合 `--fused-projection` 与
`--bf16-kv-source`。本机使用 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`。

原生内核探针验证了真实 V4 Triton storage/dequant 和 C2 kernel。RPC/图探针的
checkpoint、linear/norm/store 及 indexer/carry 规划仍有小型 fixture 边界。
未运行 H20 完整 checkpoint、FlashMLA 整模 attention、SMG/P/D、多卡 NCCL 或
生产负载，不据此断言 H20 服务已完整验收或 remote 有性能收益。
