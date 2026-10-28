# 5090 内网反馈：插件与 P5.18 的等价性及两项修复

本页整理用户提供的内网实测数据，重复段落已去重。5090 原始日志、token 文件和完整现场修复源码
未随报告提供，因此这些结果标记为**内网反馈**，不表述为本机独立复跑。
本次仓库修复另外补充了 CPU 回归，以及真实 CUDA/Mooncake/Remote RPC 检查。

## 验证对象

| 项目 | 内网报告值 |
| --- | --- |
| 插件分支 / 功能提交 | `ggg-s/sglang` `remote-attn` / `a67ca2b` |
| 参考实现 | `pd-handoff-p518-stable@17261ca0`，runtime `e545304a…` |
| 硬件 / 模型 | RTX 5090 ×8，使用 6 卡；Qwen3-8B，TP2 |
| 拓扑 | P:31016，D0:31017，D1:31018，gateway:31080；Mooncake TCP |
| 配置 | radix off，CUDA Graph off |
| 现场 runtime 缩写 | 修复前 `fd8d98a…` → 现场修复后 `854670d…` |

本次仓库实现还保留 abort/unsafe fence 的拒绝条件，并修正接收异常的事件语义，
不声称其 hash 与现场 `854670d…` 逐字节相同。合并后的版本需要在同一内网配置再次回归。

## 静态对应与 token 等价性（内网反馈）

| 指标 | 参考 e545 | 插件 fd8d98a |
| --- | --- | --- |
| 函数总数（含方法，按报告口径） | 209 | 218 |
| 类数 | 15 | 15 |
| 缺失函数 | — | 0 |
| 与参考语义相同 / 有差异 | — | 171 / 38 |
| 新增函数 | — | 9 |
| 控制命令分支（按报告口径） | 37 | 37 |

该统计口径与本地早期“顶层函数 / 全协议 cmd”的清单不同，不应直接比较数字。
P5.18 的 watchdog、claim、epoch、fence、resume/abort_import 检查均在位。
差异主要是当前 main 的 Req KV / runtime_context / batch API、stream fence、TP agreement 和 rollback/quarantine。
现场 preflight 为 16 hooks、`send_probe=true`，当时 CPU 回归为 28/28。

| 场景 | 模式 | 输出与参考逐 token 比较 | 迁移覆盖 |
| --- | --- | --- | --- |
| C1，4K，new=96 | Copy / Layer / Remote | 均为 96/96 exact | owner 包含 src、dst |
| C4，32K，new=128，修复后 | Natural / Copy / Layer / Remote | 四条各 128，全部与旧栈相同 | long-3 迁到 dst，与旧栈一致 |

C4 只覆盖四条请求中的一条实际迁移，不能视为四条都经历迁移。
C1 直连场景的 `ROLLED_BACK` 在旧栈也复现，报告归因于 gateway pre-reserve / cutover reserve 的 epoch 交互；
这是当时保留的共同问题；后续仓库修复见 [C1-RECLAIM.md](C1-RECLAIM.md)，需要内网再次验收。

## 问题一：原部署 N=8 在新 main 上造成同步拆批

32K / C4 / new=128 / Natural 的 2×2 隔离数据，单位 ms：

| Gateway | Servers | TTFT p50 | E2E p50 |
| --- | --- | --- | --- |
| 旧 P5.18 | 旧 P5.18 | 3959.9 | 7064.6 |
| 插件控制面 | 旧 P5.18 | 3963.3 | 7053.6 |
| 旧 P5.18 | 新 main | 11027.1 | 14781.2 |
| 插件控制面 | 新 main | 11110.8 | 14870.0 |

这组样本中未观察到明显的控制面额外开销，主要差异在服务器侧；不能据此证明任何负载下开销严格为零。
Prefill 日志显示旧栈约 3.8 s/请求，新栈约 3.4 s/请求，额外约 7.6 s 出现在发送 KV 到首个 Decode 之间。

| Native PD 批次上限 N | TTFT p50 | E2E p50 |
| --- | --- | --- |
| 8 | 11110.8 | 14870.0 |
| 4096 | 3912.8 | 7566.2 |
| 2048 | 3927.0 | 7576.7 |

旧 connector 不读取 `SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES`；新 main 会按它拆分同步 TE 调用。
对 32K、2048-token chunks，N=8 累计约 4096 次调用；报告估算每次约 1.8 ms，与额外延迟量级吻合。
本次只修改 `deploy/common.sh` 的默认值为 **2048**，与现有 `--chunked-prefill-size 2048` 对齐，保留显式 env override。
这改变的是原生 P→D batching，不是把 D2D progressive layer protocol 合并为整块传输。
不直接使用任意大上限；更改 chunk size 后应重新检查队列容量与发送粒度。
旧 shell 已导出的 `8` 仍优先于默认值，部署时必须显式更新并重启实例。

修复后四臂性能，单位 ms（Gap 为 p90）：

| 模式 | 插件 TTFT | 插件 E2E | 插件 Gap | 旧 TTFT | 旧 E2E | 旧 Gap |
| --- | --- | --- | --- | --- | --- | --- |
| Natural | 3927 | 7577 | 84 | 3959 | 7010 | 65 |
| Copy | 3958 | 7851 | 1440 | 3977 | 7402 | 1913 |
| Layer | 4014 | 8095 | 905 | 3978 | 7333 | 1416 |
| Remote | 3912 | 7639 | 289 | 3956 | 6999 | 385 |

报告中 TTFT 回到旧栈附近；Copy/Layer/Remote 的 Gap 分别改善约 25%/36%/25%。
E2E 仍较高，报告将其归因于新 main 的 decode 基线：约 28.7 ms/token 对比旧 24.4 ms/token。
该归因来自报告中的隔离实验，本次本地未复测其性能数值。

## 问题二：LOCAL Drop 被统一错误检查升级为 scheduler 崩溃

原 `_check_target_state` 一律拒绝 `state.error`。当本地缺一层、源端仍有冻结 KV 且 Remote client 可用时，
本应继续通过 RPC 读取缺失 prefix，却在 attention 内抛错并终止 D1，导致 controller 长期 BLOCKED。

| 指标 | 修复前 | 现场修复后 #1 / #2 | 旧 P5.18 |
| --- | --- | --- | --- |
| 终态 | BLOCKED，22.7 s 未终态 | SAFE_TO_STOP / SAFE_TO_STOP | SAFE_TO_STOP |
| outcome | — | migrated / migrated | migrated |
| drain | 22.7 s | 2.85 s / 2.83 s | 3.01 s |
| token | 0 | 96/96 exact，两次 | 96/96 exact |
| owners | — | src + dst | src + dst |
| source 业务残留 | 有 | 无 | 无 |
| D1 | 崩溃 | health 200 | health 200 |

仓库补丁包括：

1. `_can_serve_remote`：Remote + 有 client + 未 abort，且没有 unsafe GPU copy fence，才允许本地导入错误降级。
2. `_check_target_state`：保留错误供状态和控制面观察，首次降级打印日志；copy/layer、无 client、abort/unsafe 仍拒绝。
3. full-mirror 与 bounded-slot receiver 的异常路径：可降级 Remote 不再将全部层事件置位；
   event 只表示该层 H2D 已完成，缺失层在下一个 step mask 中仍为 Remote。
4. `_finish_layer_import`：缺层 Remote 不标记 import_done，也不通过异常唤醒把缺层伪装成 LOCAL。
5. 已有本地导入错误时，不执行慢 RPC 触发的 layer-wait 切换，否则会等待永远不会到达的层。

host buffer 的远端完成状态不明时，仍按原安全策略 quarantine 到进程退出。
报告中的“无残留”不应被解读为这部分 host registration 已全部回收，合并后需要单独检查。

## 其他故障和稳定性（内网反馈）

| 场景 | 报告结果 |
| --- | --- |
| Import Fail | 插件修复前/后与参考均 SAFE_TO_STOP / failed；请求留 src，96/96 exact，无业务残留；drain 3.53 / 3.75 / 2.96 s |
| Target Kill，drain 前 SIGKILL D1 | 插件与参考均 ROLLED_BACK，src 96/96 exact；22.6 s 对 19.0 s，报告归因于 bounded lock timeout |
| Controller Kill @ Freeze | source fast resume，96/96 exact，target 0 token；owner 唯一，双方 registry 归零，KV available 221444 |
| Controller Kill @ Commit | target 已提交后继续到 96/96 exact，source 没有 resume；owner 唯一，双方 registry 归零，KV available 221444 |

原 P5.18 TP2 稳定性报告：rank1 发起 resume 8 轮、rank0 5 轮、任意 rank freeze 4 轮、
import in-flight（delay 3000 ms）4 轮、rank1 补充 2 轮，合计 **23/23**。
每轮 source 96 exact、target 0、TP0/TP1 各 resume 一次、没有重复或遗留 resume、零重算、
各角色 health 200、清理成功，新请求 16 token 的 TTFT 165–190 ms。
修复后插件另抽样 **3/3**；不能把原实现的 23 轮计为插件跑了 23 轮。

## 本次仓库的回归证据

- 新增三项 CPU 用例：两个 receiver 的 Drop/EOF 与 Remote/LOCAL 路由、不可降级边界、
  三个真实 launcher 的默认 batching 和显式 override。
- 修复前，缺层事件/降级用例与 N=8 默认值用例失败；修复后完整 **31/31**（含真实 Gloo TP2 控制广播）通过。
- CUDA probe 扩展：在单卡两个进程间运行真实 RemoteHub/TargetClient socket RPC，
  分别验证 bounded-slot/full-mirror LOCAL Drop，重复读取缺失 prefix 并合并 tail，与 FP32 SDPA 参考比较。
  该检查不等于完整模型 TP2 的复跑；原 4060 首轮结果仍保留于原文件。

本次 4060 实际复跑 **10/10**：原 8 项加上两组缺层 RPC 检查。
两组各连续 3 次真实 RPC，最大绝对误差分别不超过 `3.79e-4`、`3.17e-4`；
缺失层始终 Remote、其余 5 层 LOCAL，source RPC inflight 归零，进程退出后 GPU 显存回到 0 MiB。
版本和源码 SHA256 记录在 [4060 修复回归结果](4060-intranet-fixes-results.json)。
该结果属于本次独立复跑，不与内网报告中的完整模型 token 等价性混为一项证据。

部署时使用当前分支最新代码，显式核对批次值 2048，重启 P/D0/D1 后再按上述内网矩阵回归。
