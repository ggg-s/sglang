# SA 72 字节自适应布局验收

日期：2026-10-21。保留精确频次与默认 split/gate=1，详细布局见
[FREQUENCY.md](../../FREQUENCY.md)。此前 v5 频次验收见
[历史报告](../sa-frequency-20261021/README.md)。没有运行真实模型服务。

## 结果

按配置的 `max_history_tokens` 自动选择；捕获 Graph 后不切换地址或布局。

| 容量 H | 状态存储 | 索引预算/token/槽位 |
|---|---|---|
| ≤32767 | 9×uint16 + 1×int32，22 字节/状态 | 约 72 字节 |
| ≥32768 | 10×int32，40 字节/状态 | 约 108 字节 |

32 槽 × 16K 历史的索引约从 54 降至 36 MiB。预算不含 Graph 私有池、
查询缓冲与分配器对齐；GPU 测试核对实际张量字节数与完整估算函数一致。
短 prompt 导入长容量槽位时，CPU 直接构建相同的 32 位布局；不增加 GPU
解压/转换或计数 D2H。错误布局导入会在复制前拒绝。

九个字段以值加一编码，保留 -1 哨兵和 unsigned 高位状态编号；secondary
字段完整保留 32 位，拆成两个半字读写，避免 22 字节步长的未对齐访问。
历史和边标签仍是 int32；没有限制 token 词表，也没有频次饱和或近似。

扩展名升级为 `hybrid_spec_gpu_sa_v6`，部署须同步更新并重启服务，首次重新
JIT 编译。动态成本表增加 `gpu_sa_storage_revision=6` 指纹，旧表需要重新校准。

## 验证

4060 环境：RTX 4060 Laptop 8 GiB，PyTorch 2.14.1+cu130、CUDA 13.0、GCC 12。
用户授权上传后在独立目录 `/workspace/sa-compact-20261021` 执行；87 个 SA
源码、测试、配置和 benchmark 文件与本地最终摘要完全一致。

| 检查 | 结果 |
|---|---|
| 最终 4060 专项与成本表补验 | **46 passed / 0 skipped**，其中 32 项 CUDA、14 项 CPU，exit=0 |
| 扩大包级回归 | **814 passed / 254 skipped / 3 deselected**，exit=0 |
| 其中 split/SA test_cuda | **351 passed / 248 skipped** |
| 其中 switch Graph | **14 passed** |
| 其中 CPU Gloo TP | **5 passed / 5 skipped**；单卡未验收 NCCL |
| 最终本地包级回归，不含 TP 文件 | **359 passed / 703 skipped** |
| 原生精确频次 oracle + ASan/UBSan | 通过，exit=0 |
| Ruff E9/F、58 个 Python 文件格式、diff whitespace | 通过 |

通过数存在重叠，不能相加。扩大回归先执行，随后加入最终成本表版本校验及
大于 65535 的计数用例；最终专项在全部最终源码上重新执行 frequency、native、
shared_parallel_cuda 和成本表兼容测试。详细分模块结果见 `results.json`。

关键覆盖：

- 两种布局都逐 token 对照独立子串计数 oracle，覆盖穷举、clone、重复历史、
  精确候选与只读查询，host 小容量导入后继续追加。
- 32767/32768 容量分界、最大可编码状态编号与 -1 哨兵、完整有符号 secondary。
- 原生和 CUDA 均构造超过 32768 个实际状态；两种布局的所有逻辑字段及
  Graph 更新结果一致，防止把 uint16 高位误读为负数。
- CUDA 长历史经 Graph 消费反馈后，实际出现次数达到 **65539**，仍保持精确。
- 词表 token 大于 65535、GPU 接受反馈、重复 replay、padding、代际、完成来源、
  本地/共享 donor，以及 20/64/257 来源并行查询。
- 旧版/缺少布局版本的动态成本表会被拒绝。

三个排除项沿用此前已确认的远端旧生产依赖问题，与此次压缩改动无关：
`test_draft_extend_graph_state_survives_prefill_merge_and_filter[1-True]`、
`[3-True]`、`test_cuda_target_dispatch_runs_only_selected_physical_tier`。
这些旧依赖的当前仓库版本仍未在远端验收，不能视为通过；原因和差异文件见
[此前报告](../sa-frequency-20261021/README.md)。本轮用正确的 `tests/` node ID
明确排除，未修改这些测试的断言。

扩大回归沿用已记录的轻量 serving 导入隔离及 capture GC 插件，只替换合成
测试原本不用的服务边界；真实 CUDA、Triton、行号及 Graph 操作仍执行。
首次 v6 编译发现分派包装位置错误，修复后重新编译与验证；原始失败日志保留。

## CUDA 组件成本对比

后续[分阶段计时与异常调查](../sa-phase-20261021/README.md)增加了CUPTI kernel
直接计时，确认外围event会混入提交/调度间隙。以下旧数值保留为原始记录；比较
更新、查询本身的计算耗时请以新报告为准。64种token在旧门槛下实际会提前回退。

同一份 v6 精确频次算法，对比显式 32 位布局与自动布局。Graph 中消费 9 个
token 并查询 8 个候选，每点 30 个计时样本，恢复 GPU 状态在计时区间外。
这是 warm-cache 组件测试，不含模型、TP、H2D、反馈生产或真实请求调度成本。
没有固定 GPU 时钟，不用于推算 TPOT 或吞吐。

14 个初始形状及 6 个复测形状的语义 checksum 全部一致。16K、batch=4 时，
压缩索引减少 2359368 字节，符合每状态节省 18 字节的计算。

部分初始组件中位数（μs）：

| 容量 | 字母表 | batch | 32 位 | 自动布局 |
|---|---|---|---|---|
| 16384 | 2 | 1 | 169.984 | 142.336 |
| 16384 | 2 | 4 | 168.960 | 143.360 |
| 16384 | 64 | 1 | 72.704 | 64.512 |
| 32768 | 2 | 4 | 201.728 | 202.752 |

32768 的两组实际都使用 32 位布局，作为噪声对照。初始 16K/单一 token/batch=4
出现过 2.945 倍耗时异常；保留原始结果，额外三次复测的自动/32 位耗时比分别为
1.048、0.961、0.949，未稳定复现，没有给异常点作确定归因，也不据此宣称模型加速。
原始样本见 `layout-costs.json`、`layout-repeat-costs.json`。

## 证据与范围

`source-manifest.json` 保存远端源码摘要和环境；`manifest.json` 保存本地结果及
证据文件摘要。`cuda-evidence.tar.gz` 包含运行脚本、测试插件、XML、日志、
编译失败记录及组件计时脚本；`local-evidence.tar.gz` 保存本地 XML 和 sanitizer
日志。真实模型接受率、吞吐、多卡 NCCL 和上述三项旧依赖检查仍未验收。
