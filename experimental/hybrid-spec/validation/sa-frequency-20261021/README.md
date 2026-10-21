# SA 在线频次与 split/gate=1 默认值

日期：2026-10-21。基线提交 `84b4960ed5`，实现与策略说明见
[FREQUENCY.md](../../FREQUENCY.md)。没有执行模型吞吐或质量实验。

## 实现

- CPU/CUDA 共用 suffix-link 动态树维护精确出现次数，clone 继承后重新挂接；
  无定期全历史重扫。query 只读，避免共享 donor 查询间的数据竞争。
- 默认逐步选择最高频续写，同频选择较小 token ID。保留 first 复制策略；
  不做跨 donor 的频次合并，候选仍经 gate 与 target 验证。
- 普通配置默认 device/split/gate=1。DSPARK 和原 sa-spec 参考配置显式保留
  支持范围内的特殊策略。worker split API 仍是 v5，SA 扩展缓存名升级到
  `hybrid_spec_gpu_sa_v5`，索引预算约 108 字节/token/槽位。

## 本地验证

环境：macOS arm64，Python 3.11、PyTorch 2.14.1、Apple Clang。
依赖安装在独立临时 venv。此环境没有 NVIDIA CUDA。

| 检查 | 结果 |
|---|---|
| 最终包级回归，不含单独运行的 TP 文件 | **353 passed / 697 skipped** |
| TP 文件，真实 CPU Gloo 1/2/4/8 进程与子组 | **5 passed / 5 skipped** |
| 最终 native + frequency + core 专项 | **116 passed / 10 CUDA skipped**，包含在包级范围内，不另行相加 |
| 频次 C++ 独立 oracle，ASan/UBSan | 通过，无越界或未定义行为报告 |
| Ruff 错误/未使用导入、格式、diff whitespace | 通过 |

新增原生测试在每次追加后扫描历史，核对所有状态出现次数、每步贪心续写，
并确认只读查询没有改写状态。覆盖二值穷举、随机 clone、重复文本、最长及最短
状态代表串、较小 host 存储导入大容量后继续追加。

首次本地尝试缺少 Torch；已在临时 venv 安装。原生编译的临时文件路径与 Gloo
loopback 曾受沙箱限制，保留 TMPDIR 后编译通过，Gloo 在允许 loopback 的本地
执行中通过。默认改变后，原来验证 full/gate=0 的 fixture 现显式声明这些参数；
split gate 测试继续独立运行。跳过项包括 CUDA/NCCL 和未安装的 Arctic 扩展，
不能算作这些能力通过。

## 原生 CPU 组件微测

`benchmark_sa_history.py --baseline-ref HEAD --lengths 1024 16384 32768
--vocabs 1 2 64 --repeats 3`，追加长度沿用工具默认值。九个点的自动机语义
checksum 与基线一致。原始样本在 `native-costs.json`。

计数维护增加 CPU 成本。例如 32768-token / 二值历史，旧版追加中位数约
0.0315 μs/token，新版约 0.5061 μs/token；64-token 字母表约 0.0743 →
0.1241 μs/token。无 CPU 绑核、每点三轮，存在大小核/系统调度波动；这只是
原生状态维护的有界组件测量，不能用于推算 CUDA 查询、模型 TPOT 或吞吐。
没有把频次新增描述成零开销，也没有据此宣称策略的端到端净收益。

## RTX 4060 CUDA 验证

获得用户源码上传许可后，在独立目录 `/workspace/sa-frequency-20261021` 验证。
环境为 RTX 4060 Laptop 8 GiB、PyTorch 2.14.1+cu130、CUDA 13.0、GCC 12。
新版 `hybrid_spec_gpu_sa_v5` 完成实际 JIT 编译，SA 源码、测试、模板、微测
共 86 个文件的 SHA-256 均与本地最终文件一致，见 `cuda-source-manifest.json`。

| 检查 | 结果 |
|---|---|
| frequency + native + shared_parallel_cuda 专项 | **33 passed，0 skipped**，其中 26 项执行 CUDA、7 项 CPU；exit=0 |
| 扩大包级回归 | **803 passed / 3 failed / 254 skipped**；exit=1，不是全绿 |
| 其中 test_cuda | **351 passed / 248 skipped**，包括 split 紧凑尾段组件矩阵 |
| 其中 test_switch_graph | **14 passed** |
| 其中 test_dspark | **28 passed** |
| 其中 test_tp | **5 passed / 5 skipped**，通过的是 CPU Gloo；单卡没有验证 NCCL |

专项是扩大回归的重叠子集，不能把通过数相加。专项覆盖 first/frequency、
eager/Graph、本地/共享 donor、GPU 接受路径反馈、排除拒绝 token、重复 replay、
padding、代际失效、20/64/257 来源并行查询和发布顺序变化。

新增两个负例首先在未修复实现上均失败，证实短候选或分支歧义回退时，
consume-query 已写入的候选没有清零。修复后两个负例及专项全部通过。
旧测试另有三类适配：生命周期测试显式保留 full，旧 donor 排序测试显式保留
first，结构比较只读取 10 字段布局的前 5 个语义字段。新版频次的行为由独立
文本计数 oracle 覆盖，没有把这些预期改成忽略错误。

扩大回归中的三个失败与远端已有生产依赖版本差异一致：

- `test_draft_extend_graph_state_survives_prefill_merge_and_filter[1-True]`
  和 `[3-True]`：远端旧 draft-extend runner 没有当前 STANDALONE 的
  `hidden_states=None` 处理，随后 filter 发生 IndexError。
- `test_cuda_target_dispatch_runs_only_selected_physical_tier`：远端旧 decode
  runner 将设备选择的 tier 送入 CPU `load_batch`，触发断言。

对 21 个测试依赖进行摘要比对，18 个与本地一致，另外 3 个不同，详见
`cuda-support-comparison.json`。除上述两个 runner 外，`eagle_compact_verify.py`
也存在差异。这些生产文件未由本次 SA 改动修改。额外生产依赖源码包上传被
自动审批拒绝，理由是超出已授权 SA 修改及相关测试范围，未改用其他通道上传。
因此三个检查在当前本地版本上的远端结果仍未验收，不将其算作通过。

扩大回归脚本中的三个 `--deselect` 使用了错误的仓库相对路径，而 pytest
node ID 实际以 `tests/` 开头，所以排除没有生效；上述三项实际执行并失败。
保留原始命令与日志，不将全套结果改写为“排除后全绿”。

紧凑尾段测试沿用远端已有的轻量测试插件：隔离服务模块导入，只替换测试中
本来不用的 serving 边界；真实行号、top-k Triton、Graph 与相关 GPU 算子仍执行。
捕获测试期间关闭循环 GC，避免模块回收干扰 CUDA capture。两个插件源码与
运行脚本均随证据归档。这是无权重组件回归，没有启动模型服务。

结果明细在 `cuda-results.json`；JIT/回归日志、XML、修复前负例、运行脚本与
测试插件在 `cuda-evidence.tar.gz`。初次上传产生的 macOS `._` sidecar 导致过
配置读取错误，清理 sidecar 后重跑通过；初始失败记录也保留在归档中。
本地原始记录在 `local-evidence.tar.gz`，最终补充记录在 `local-final-evidence.tar.gz`。

## 待验证范围

真实模型接受率、TPOT、吞吐、多卡 NCCL 和上述三个当前生产依赖的远端检查
尚未验收。频次维护和内存均有成本，不能从组件测试通过推断端到端性能提升。
