# Remote-attn Progressive 生产接入手册

本文对应 `codex/remote-attn-pre-layerwise` 分支，运行时代码基线为
`5b6e2f7ef554e71634352284d55c201989e35b29`（2026-10-29）；预检与原生命令文档于
2026-10-30 更新。发布时固定经过验收的完整 SHA、wheel 和配置，不能让生产实例自动追随分支 HEAD。

目标路径是 **P Prefill 层组发送 → D 提前 Remote Decode → scatter 完成后逐层切 Local**。
本手册给出接入现有生产 serving 的发布步骤；它不代表这份 SHA 已完成生产 GPU 验收。
当前本地证据为 64 项 CPU 检查、23 个源码 hook 和 40 项核心接口/状态检查通过。
现场定位及修复实验支持继续验证，但不能替代这份构建在目标 GPU、模型和负载上的完整验收。

旧 P→D0→D1、`cut/freeze/migrate` 和实验 gateway 的操作移至
[历史 P5.18 部署手册](migration/P5.18-DEPLOYMENT-LEGACY-ZH.md)。
当前 Progressive 使用原生 PD router；不向同一请求叠加 P5.18 迁移命令。
实现细节与对照实验见 [P5.19 说明](P5.19-PRE-LAYERWISE-ZH.md)。
单 pair 直接启动命令见 [原生 SMG + P/D 接入](SINGLE-PAIR-ZH.md)，保留线上
optimistic prefill、chunk 和 Batch，不需要 JSON 配置或启动包装器。

## 1. 接入边界与服务拓扑

本次发布单元为一套 P/D pair 加 PD router。示例使用 Qwen3-8B、P TP2 + D TP2，共四张 GPU。

| 组件 | 示例地址 / GPU | 职责 |
| --- | --- | --- |
| P | `127.0.0.1:31016`，GPU 0、1 | Prefill、KV 发送、Remote prefix attention |
| D | `127.0.0.1:31017`，GPU 2、3 | Decode、接收 staging、逐层 Local 切换 |
| Bootstrap | `127.0.0.1:35409` | 原生 PD 注册 |
| PD router | `127.0.0.1:31000` | 现有业务入口后方的 PD 请求路由 |
| 控制目录 | P/D 共用本机绝对路径 | generation/room/rank 隔离的 offer、ready、done 和诊断记录 |

当前控制与 Remote RPC 使用 loopback，P/D 必须处于**同机、可互通 loopback 的网络环境**，并看到同一份本机目录。容器拆分时需要共享相同目录及兼容的网络 namespace；只挂共享存储不足以支持跨机。
多套 pair 应各用独立端口、GPU、控制目录和 journal。业务流量通过现有鉴权/限流入口访问 router，内部控制与 RPC 端口不作为公网 API。

当前路径要求：

- `Qwen3ForCausalLM`、CUDA、dense NHD MHA、BF16/FP16 KV、page size 1。
- P/D 相同 TP 宽度，PP=DP=CP=1；示例为 TP2，不能直接沿用异构 TP 拓扑。
- P 可开启设备内 FULL radix cache，D 关闭 radix cache；P 关闭 scheduler overlap，D Remote 串行、全部 Local 后恢复原生 overlap；保留原生 full decode graphs，capture buckets 包含 1/2/4。D Remote batch 1/2/4 使用分段图，attention/RPC/merge 在图外；未覆盖的 Remote batch 为 eager，统一回收后恢复 full graph。最终 chunk 的分层 prefill 仍 eager。P 关闭 scheduler overlap 不影响分层计算/传输重叠；D two_batch_overlap 仍关闭。
- 原生 `mooncake_tcp`，P/D 均使用 `--disaggregation-decode-allocation-policy early`（代码默认）；不能使用 `prefill_complete`。
- LoRA、MLA/混合/量化 KV、HiCache、speculative、统一 KV、PD multiplexing、live role switching 不在当前支持范围；原生 host KV destination 也不等于这里的 host staging ring。

如果现有生产模型或拓扑不满足这些条件，需要先完成适配，不能仅添加环境变量上线。
**保留模型部署自己的 token chunk 配置**；不新增 Pre-Layerwise chunk size。
中间 token chunk 走原生整层发送，只有实际最终 token chunk 按层组发送。
层分组默认 auto；不足 3072 个合格计算 token 时，`layerwise` 回退原生，`remote` 合为一组并保留提前解码。这种请求不应要求跨层重叠。

## 2. 发布制品与依赖

需要同步部署本分支 SGLang 核心和插件，不能只覆盖 `handoff_control.py` 或复制插件到任意 PyPI SGLang 上。

| 制品 | 要求 |
| --- | --- |
| SGLang runtime | 本次固定 SHA 的 `python/` 及完整构建输入；包含 Mooncake worker、原生 staging handler/ring 的修改 |
| 插件 | 同一 SHA 的 `plugins/remote_attn/`，安装 `sglang_remote_attn:register` entry point |
| PD router | 使用已验收的 `sglang-router` 构建；本仓库构建入口为 `sgl-model-gateway/bindings/python` |
| Mooncake | 现场基线 `mooncake-transfer-engine-cuda13==0.3.13.post1`，必须提供 `TransferEngine.send_probe` |
| Python / CUDA | Python 3.12+；现场基线 Python 3.12 / CUDA 13，驱动须兼容所装 runtime |
| PyTorch / kernels | 当前 `python/pyproject.toml` 固定 `torch==2.14.1`、`sglang-kernel==0.4.10`、`flashinfer_python[cu13]==0.7.2`；`cuda-python>=13.0` |
| 模型与配置 | 固定模型/tokenizer revision、原部署参数、环境变量、镜像/驱动/GPU 信息；随发布留档 |

不需要修改 Mooncake C++。发送私有流 D2H 和接收 host ring 使层组传输用 host 地址绕开 TCP 的 GPU null-stream 拷贝。
不要用 `SKIP_SGL_KERNEL_VERSION_CHECK` 掩盖 ABI 不匹配，也不要同时安装多个提供 `mooncake` 模块的 wheel。
历史 4060 轻量 probe 的依赖组合不是本版本完整模型服务的依赖锁。

### 构建与离线交付

在匹配生产平台的 Linux / Python / CUDA 构建环境、独立 checkout 上操作。macOS wheel 不能交付 Linux。
先把外部 `codex/remote-attn-pre-layerwise` 同步到内网镜像；下面从该镜像选取候选版本并立即冻结 SHA：

```bash
cd /path/to/sglang-build
git fetch origin codex/remote-attn-pre-layerwise
export HANDOFF_REVISION=$(git rev-parse FETCH_HEAD)
git switch --detach "$HANDOFF_REVISION"
git status --short

export HANDOFF_DELIVERY=/tmp/remote-attn-progressive-delivery
mkdir -p "$HANDOFF_DELIVERY/wheelhouse"
git rev-parse HEAD > "$HANDOFF_DELIVERY/REVISION"
git archive --format=tar.gz \
  --output="$HANDOFF_DELIVERY/sglang-source.tar.gz" HEAD
python -m pip wheel --wheel-dir "$HANDOFF_DELIVERY/wheelhouse" \
  ./python ./plugins/remote_attn ./sgl-model-gateway/bindings/python \
  'mooncake-transfer-engine-cuda13==0.3.13.post1'
cd "$HANDOFF_DELIVERY"
sha256sum REVISION sglang-source.tar.gz wheelhouse/*.whl > SHA256SUMS
```

构建必须从干净 checkout 开始。wheelhouse 使用新目录，只含本次平台的一组制品；构建机需要完整 Rust/CUDA 等输入。
`git archive` 不打包独立 submodule 内容，构建需要的输入须事先在构建机准备。
另行交付模型、tokenizer 和 FlashInfer 所需 cache/离线输入。候选构建通过第 7 节后，才可作为正式 release。

### 安装和核对加载来源

在新 release 目录和独立 Python 环境安装；不覆盖正在接流量的进程环境。以下路径按现场替换：

```bash
export HANDOFF_DELIVERY=/path/to/remote-attn-progressive-delivery
export SGLANG_ROOT=/workspace/sglang-progressive-release
export HANDOFF_PYTHON=/opt/conda/envs/sglang-progressive/bin/python
export HANDOFF_MODEL_PATH=/workspace/models/Qwen3-8B
export HANDOFF_WHEELHOUSE="$HANDOFF_DELIVERY/wheelhouse"
cd "$HANDOFF_DELIVERY"
sha256sum -c SHA256SUMS
mkdir -p "$SGLANG_ROOT"
tar -xzf sglang-source.tar.gz -C "$SGLANG_ROOT"
"$HANDOFF_PYTHON" -m pip install --no-index --find-links "$HANDOFF_WHEELHOUSE" \
  "$HANDOFF_WHEELHOUSE"/sglang-*.whl \
  "$HANDOFF_WHEELHOUSE"/sglang_remote_attn-*.whl \
  "$HANDOFF_WHEELHOUSE"/sglang_router-*.whl \
  'mooncake-transfer-engine-cuda13==0.3.13.post1'
"$HANDOFF_PYTHON" -m pip check
"$HANDOFF_PYTHON" -m pip freeze > "$HANDOFF_DELIVERY/installed-freeze.txt"

"$HANDOFF_PYTHON" - <<'PY'
import importlib.metadata as metadata
import torch
import sglang
import sglang_remote_attn
import sglang_router
from mooncake.engine import TransferEngine
print("paths:", sglang.__file__, sglang_remote_attn.__file__, sglang_router.__file__)
print("torch/CUDA:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
for name in ("sglang-kernel", "flashinfer_python", "cuda-python",
             "mooncake-transfer-engine-cuda13", "sglang-router"):
    print(name, metadata.version(name))
print("send_probe:", callable(getattr(TransferEngine, "send_probe", None)))
print("plugins:", list(metadata.entry_points(group="sglang.srt.plugins")))
PY
"$HANDOFF_PYTHON" -m sglang_remote_attn.check
```

当前 preflight 应输出 `hooks: 27`、`core_interfaces: 67`、`send_probe: true`、
`plugin: remote_attn`。核心接口检查同时覆盖 metadata receiver 保留、选层发送、
staging/host copy、任务与状态字段和 typed env；安装态检查实际加载模块，源码模式
检查所选 checkout。直接启动服务的插件注册也执行核心检查。
它只检查安装和接口，不会启动模型，也不会初始化服务进程中的 D2H slot。
wheel 安装后改 checkout 文件不会更新运行代码，必须重新构建安装。
开发联调可以使用 `pip install -e ./python` 和 `pip install --no-deps -e ./plugins/remote_attn`；正式实例使用已验收的不可变制品。

## 3. Progressive 环境配置

以下变量放进实际 P/D 服务的启动环境，每次变更配套重启 pair。示例是 Bash `export` 语法；若使用 systemd `EnvironmentFile`，转成无 `export` 的 `KEY=value`，并事先展开所有路径变量。`EnvironmentFile` 不执行 Bash 的 `$变量` 展开；也可由受管启动脚本加载 Bash 配置。
`SGLANG_PLUGINS` 需保留已有其他插件项并添加 `remote_attn`；安装插件和设置 allowlist 后，每个角色还必须有自己的 `HANDOFF_CTL_DIR`。

### P/D 共同配置

```bash
# 路径在两个进程中必须指向同一份本机目录；每次 pair 重启使用新实例目录。
export HANDOFF_INSTANCE=/workspace/remote-attn/run-20261029-01
export HANDOFF_CTL_ROOT="$HANDOFF_INSTANCE/control"
export HANDOFF_PD_DIR="$HANDOFF_INSTANCE/journal"
export HANDOFF_RUN_DIR="$HANDOFF_INSTANCE/logs"
case ",${SGLANG_PLUGINS:-}," in
  *,remote_attn,*) ;;
  *) export SGLANG_PLUGINS="${SGLANG_PLUGINS:+${SGLANG_PLUGINS},}remote_attn" ;;
esac
export SGLANG_PLUGINS
export HANDOFF_PRE_LAYERWISE=remote
export HANDOFF_PRE_LAYERWISE_GROUP_SIZE=0
export HANDOFF_PD_DRAIN_TIMEOUT_S=300
export MC_FORCE_TCP=1
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096
export SGLANG_MOONCAKE_SEND_LOCK=1
# SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES 沿用现有部署值，不在这里覆盖。
export MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=8192
export MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=8192
export SGLANG_ENABLE_FAILED_SESSION_PROBE=1
export SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=1.0
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export HANDOFF_PROFILE=0
export HANDOFF_RPC_PROFILE=0
mkdir -p "$HANDOFF_CTL_ROOT/p" "$HANDOFF_CTL_ROOT/src" \
  "$HANDOFF_PD_DIR" "$HANDOFF_RUN_DIR" /workspace/handoff_state
```

`/workspace/handoff_state` 是 Remote runtime 仍使用的固定诊断目录；即使 profile 关闭，目标状态创建时也会建立该目录，服务账号必须有访问/创建权限。它不随 `HANDOFF_PD_DIR` 改变，容器需提供可写挂载。多实例诊断须区分文件归属，不能当作实例 journal 一起清空。

**发送 Batch 不限定为 8，也不要求与 token chunk 或请求并发相等。**
`SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES` 沿用现有部署值，插件不改写。未配置时，仓库 `deploy/common.sh` 使用 `2048`；直接启动原生服务、不经过该脚本时，代码默认 `0`（不按此上限拆分）。这两个默认来源必须区分，不能因为更换启动入口而悄悄改变有效值。
正值限制每次同步批量传输的 KV 索引数。更小的值增加 TE 调用次数，更大的值或 `0` 可能增加 P staging 容量与单次传输时长；以目标环境的吞吐、尾延迟和内存预算选择。Batch=8 仅是此前实验的一种设置。正式配置记录有效值，调整时单独复测，不与 staging 开关一起改变。
`HANDOFF_PD_DRAIN_TIMEOUT_S` 是清理等待上限，不是请求自动恢复机制；按业务最长请求和回收耗时调整。

### P 角色

```bash
export HANDOFF_CTL_DIR="$HANDOFF_CTL_ROOT/p"
export HANDOFF_PRE_LAYERWISE_HOST_STAGING=1
export HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=0
export HANDOFF_PL_RETIRE=async
```

### D 角色

```bash
export HANDOFF_CTL_DIR="$HANDOFF_CTL_ROOT/src"
export HANDOFF_PRE_LAYERWISE_HOST_STAGING=0
export HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=1
export HANDOFF_RETRACTION_BACKUP=none
```

P/D 分别在自己的服务环境使用对应角色块，不能在同一 shell 顺序执行两个角色块后共用最终环境。
D 的 `none` 沿用当前配置：内存压力下原生 retraction 会中止请求并返回 503，不表示有 CPU KV backup。
发送端 host staging 默认在 `MC_FORCE_TCP=1` 时开启，但生产配置显式写出；接收 host staging **默认关闭**，必须显式开启。

| 配置 | 代码默认 | 运行含义 |
| --- | --- | --- |
| `HANDOFF_PRE_LAYERWISE` | `off` | `remote` 提前 Decode；`layerwise` 仅重叠、等待完整 KV；`off` 回原生 PD |
| `HANDOFF_PRE_LAYERWISE_GROUP_SIZE` | `0` | auto；正整数固定层组，不改变 token 分块 |
| `HANDOFF_PRE_LAYERWISE_HOST_STAGING` | TCP 强制模式下 `1`，否则 `0` | P 私有流 D2H；`0` 为源 GPU 指针对照 |
| `SGLANG_DISAGG_STAGING_BUFFER` | `0` | P/D 一致开启原生 ring 协议 |
| `HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING` | `0` | D pinned host ring + GPU shadow；须同时启用原生 staging |
| `SGLANG_DISAGG_STAGING_POOL_SIZE_MB` | `4096` | 每个 D rank 的 ring 容量，单位 MiB |
| `HANDOFF_PL_RETIRE` | `async` | P 层组任务异步回收；`off` / `keep64` 仅用于诊断对照 |
| `SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES` | 原生 `0`；参考脚本 `2048` | 沿用部署值，正值限制一次同步 batch 的 KV 索引数；不限定为 8 |
| 冷态 D2H 预热 | 新 slot 自动执行 | 无独立环境开关；首次注册和扩容后执行，复用不重复 |

## 4. 内存、回收和冷态行为

### 发送与接收顺序

```text
层组 KV ready event
  → P 专用非阻塞流 D2H → 等 D2H 完成 → TE 发送 host 地址
  → D 原生 pinned host ring → 原生 CHUNK_READY
  → 已有 scatter stream：异步 H2D 到 GPU shadow → scatter 写 KV pool
  → scatter event 完成 → 允许该层 Local + watermark 释放 ring 区间
```

复用原生 `staging_base_ptr` 握手、`STAGING_REQ/RSP`、`CHUNK_READY` 和 watermark；没有自建 receive journal 握手或逐组 ACK。
Remote 的 offer/ready/done journal 仍然存在。网络发送成功或 H2D 完成均不能单独作为切 Local 的依据。
发送锁、session 和 pre-send probe 保留；普通中间 token chunk 和 Post-Remote 仍走原路径。

### 内存预算

- 每个 D rank：host 模式固定占用 **R MiB pinned RAM + R MiB GPU shadow**，此外还有模型、KV pool、算子 workspace；GPU ring 对照只有 R MiB GPU ring。
- 默认 R=4096，D TP2 合计额外 8 GiB pinned RAM，每张 D GPU 额外 4 GiB shadow。主机内存限额也必须包含 P staging 和其他进程。
- P slot 按实际最大层组和原生发送 batch 上限预留；首次分配及扩容发生在 Prefill 前。普通同布局路径可按 `min(实际尾块 tokens, 正的 Batch 上限) × 本组各 K/V buffer 的每 token 字节数之和` 估算；实际容量以代码计划和分配为准。
- D ring 需要容纳**完整一个层组**，不能只按 P 的单次小 batch 发送量估算。page size 1 时，单组字节数为 `实际最终 chunk tokens × 所选 K/V buffer 的 kv_item_lens 之和`，取每 rank 的本地布局。auto 在小尾块可能合成全层，须纳入最大值。
- R 小于最大单组会失败，不能依赖无限重试或直接写 KV pool 回退；R 只容一组时也可能降低流水效果。按并发和 watermark 积压评估余量。

原启动脚本的静态显存比例 P=0.85、D=0.75 是历史参考值，不能假设给 4 GiB shadow 留足空间。
在目标 GPU 上先做启动、最长上下文和目标并发容量验证，再固定显存比例、R 和并发；本次接入不自动改 token chunk。

### 层组引用延迟回收

`HANDOFF_PL_RETIRE=async` 在 worker 完成层组处理后把任务引用交给 daemon reaper，避免 `partial(..., ready_event=...)` 的最后引用在传输线程触发昂贵的 Event 析构。
待 watermark 重新排队的任务仍归原队列所有，不提前回收；发送成功和失败路径保留原有完成判断。

每个 transfer worker 的队列上限为 **64 个待回收任务，另有 1 个正在回收**。正常路径 `put_nowait`；满载时输出
`Layer-group reference reaper full; waiting for capacity` 并背压等待。
不能把“满则丢引用”当成绝不阻塞：那会在 worker 再次触发析构。持续出现该日志需检查回收积压，不能宣称所有负载下无等待。
`keep64` 只保留最后 64 个任务，淘汰和退出时仍可能同步析构；`off` 恢复 inline release。这两项不是上线默认配置。

### 冷态预热与流量准入

core 共享 slot 的 `reserve()` 新建并注册 slot 后，使用**同一 copy stream、同一已注册 pinned buffer**完成一次真实的 **最多 16 MiB D2H** 并等待完成。
源端使用私有 GPU scratch，不读取正在生成的 KV。slot 复用不重复预热，扩容会再做；异常仅在确认流已排空后允许继续，否则 quarantine。

这只预热首次拷贝初始化，不等于触碰整个 buffer，也不等于完整模型/网络/scatter 预热。
`reserve()` 在请求准备阶段执行，若首次创建发生在第一条请求，其初始化耗时仍计入该请求。
**进程启动、preflight、`/health` 成功不能证明业务请求的 staging 已热。**
生产接流量前按第 6 节经 router 发送代表性请求，覆盖模型实际分块、最大层组和需要的 slot 容量；确保每个目标 pair/rank 实际走过 host staging 且完成回收。
重启或换到新 pair 后重新预热；后续超出预热容量的请求仍可能引发扩容冷态成本。

## 5. 与现有服务启动流程对接

将第 3 节环境加入现有进程管理器的 P/D 启动配置，保留模型路径、token chunk 和已核准的其他模型参数。
必须保留本路径约束：page size 1、同 TP、D 关闭 radix cache、P 关闭 overlap scheduler，D 使用插件的 Remote 串行 / Local 原生 overlap 策略，P 设备内 FULL radix 读租约见 [原生接入](SINGLE-PAIR-ZH.md)、`mooncake_tcp`、提前分配。
graphs 沿用原生配置；Progressive 复用准入/回收的 TP 投票维护统一接收租约，租约清空后
恢复原生 graph 判断，正常 Local decode 不新增 collective。旧手动 import 保留逐步投票。
Remote 租约期间，D 对 batch 1/2/4 选择插件分段图，其他 batch 为 eager；
`HANDOFF_REMOTE_SEGMENTED_GRAPH=0` 可重启为 Remote eager 对照。分段图增加独立 graph pool，
须在实际模型上验收显存与收益，详见[分段验证](validation/REMOTE-SEGMENTED-GRAPHS.md)。
启动及 eager→graph 验证见[单 pair 说明](SINGLE-PAIR-ZH.md#验证-eager--graph-恢复)，
本地测试范围见[适配记录](validation/PROGRESSIVE-CUDA-GRAPHS.md)。
命令行原本有冲突配置时须在 manifest 中消除，不能只叠加变量期待自动兼容。

以下脚本用于与仓库参考 Qwen3-8B TP2 配置一致的首轮联调；脚本自带历史模型参数，并不是读取任意线上 manifest 的通用入口。
**现有生产配置的 chunk 不同时，使用现有服务入口接入上述环境，不原样替换成这些脚本。**
每个终端/服务先加载公共配置、路径和对应角色配置；交由 supervisor 管理前台进程、日志和退出状态：

```bash
# P 服务：已加载 P 环境
cd "$SGLANG_ROOT"
CUDA_VISIBLE_DEVICES=0,1 bash plugins/remote_attn/deploy/start_p511_p_tp2.sh
```

```bash
# D 服务：已加载 D 环境
cd "$SGLANG_ROOT"
CUDA_VISIBLE_DEVICES=2,3 bash plugins/remote_attn/deploy/start_p511_d0_tp2.sh
```

脚本自动合并插件 allowlist、执行 preflight；`common.sh` 的 Send Lock、probe、队列稳定性默认配置继续有效。
不使用脚本时，服务环境也要显式提供这些设置。不要误把 D0 脚本中的 Fast Freeze 默认项当成本路径必须启动 D1 的信号。

等待 P/D 加载完成，分别检查；超时应检查启动错误，不直接开放业务入口：

```bash
curl --fail --max-time 30 http://127.0.0.1:31016/health
curl --fail --max-time 30 http://127.0.0.1:31017/health
ls "$HANDOFF_CTL_ROOT"/{p,src}/ready.*.json
```

TP2 每个角色应有当前进程生成的 `ready.0.json` / `ready.1.json`，核对 PID、启动时间和 `tp_size`，不能只检查旧文件存在。
P/D 使用相同发布制品与匹配开关；启动日志中的有效参数和配置快照要留档。
然后启动内网 router，保持外部入口尚未导流：

```bash
"$HANDOFF_PYTHON" -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:31016 35409 \
  --decode http://127.0.0.1:31017 --host 127.0.0.1 --port 31000
```

当前 P/D pair 不需要 `p4_gateway.py`、D1 或 `/v1/router/migrate`。
业务可沿用已验证的标准接口经过原生 router 访问；第 6 节使用原生 `/generate` 做接入检查。

## 6. 接流量前预热与就绪判定

在业务入口未导流时，用同一 serving Python 通过 router 发送代表性请求。以下输入长度仅是请求样例，**不设置 token chunk**；按模型上下文上限调整，使输入加输出在部署限制内：

```bash
"$HANDOFF_PYTHON" - <<'PY'
import json
import os
import time
import urllib.request
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.environ["HANDOFF_MODEL_PATH"])
seed = tok.encode("Explain how a key value cache works. ", add_special_tokens=False)
for length in (2048, 8192, 32768):
    ids = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
    for repeat in range(2):
        payload = {"input_ids": ids, "sampling_params": {
            "temperature": 0, "max_new_tokens": 32, "ignore_eos": True}}
        request = urllib.request.Request("http://127.0.0.1:31000/generate",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        print(length, repeat, round(time.perf_counter() - start, 3),
              result.get("meta_info", {}), flush=True)
PY

# 请求结束后继续等待后台回收；未回收时不应判定就绪。
"$HANDOFF_PYTHON" "$SGLANG_ROOT/test/manual/remote_attn/pre_layerwise_report.py" \
  "$HANDOFF_PD_DIR" > "$HANDOFF_RUN_DIR/warmup-report.json"
```

就绪条件必须同时满足：

1. P/D 健康且加载的是固定 release；代表性请求经 router 正常完成，无重复响应、异常结束或不合理重试。
2. report 中 `activated_shards > 0`，目标行的 `host_staging`、`native_staging`、`recv_host_staging` 均为 true；核对该 pair 的全部 TP rank。没有 activation 可能是 offer/配置不匹配造成原生回退，不算预热成功。
3. 预热停止后 `released_shards == activated_shards`，无 `error`；结合服务请求状态确认没有普通 PD 请求遗留。report 只统计已激活路径，不能独立证明整个服务无在途请求。
4. 不出现持续 reaper 满载、staging allocation 失败、quarantine、KV 重复释放或 session 错误；内存峰值在核准预算内。

实际需要验证 Remote 或跨层重叠时，在具备相应窗口的受控请求上追加 `--require-remote` 或 `--require-overlap`。
所有层提前到齐时没有 Remote 调用可以是正常结果，小尾块合组也不意味着故障；不要要求每一条预热请求都触发 Remote。
单 pair 顺序预热只是最小门槛，正式接入前还需第 7 节的目标并发和故障验收。

## 7. 灰度验收、观测和扩大流量

先在隔离的候选 pair 上完成验收，再由现有入口按固定比例灰度导流；保留已验收基线 pair 和撤流通道。
所有对照固定模型、输入/输出、采样、token chunk、层组策略、Batch、graphs、TP、profile 和内存参数。

| 对照目的 | 配置变化（需 drain 后重启 pair） |
| --- | --- |
| 原生 PD 基线 | P/D `HANDOFF_PRE_LAYERWISE=off`、`SGLANG_DISAGG_STAGING_BUFFER=0`，D receive flag `0` |
| 只比较提前 Decode | `remote` ↔ `layerwise`，发送/接收 host staging 和其他配置一致 |
| 接收流依赖归因 | P/D staging 保持 `1`；D receive flag `1` ↔ `0`（GPU ring 对照） |
| 发送 staging 归因 | 只切 P host staging `1` ↔ `0` |
| 引用回收归因 | 只切 P `HANDOFF_PL_RETIRE=async` ↔ `off` / `keep64`；仅隔离测试 |
| 冷态 / 热态 | 新 pair 首个请求单独统计，再统计完整预热后的请求；记录扩容事件 |

正确性验收覆盖实际部署的长短输入、最终尾块、混合长度请求、目标并发、连续 ring 绕回、多轮复用、客户端取消与重启。
用 temperature=0 的相同输入比较基线输出；若出现浮点路径差异，定位首个分歧并比对 logprob，不能直接把差异忽略。
故障验收确认网络失败/超时不会发布 Local-ready，也不会提前释放仍可能被写入或被 Remote 读取的 KV。
完整目标 GPU TP2 验收、吞吐/延迟 SLO 和容量结果须与 release SHA 一起保存。

观测至少包含：

- 业务 p50/p95/p99 TTFT、ITL、E2E、吞吐、错误/取消/超时率；按冷态与热态分开统计。
- activation 比例、Remote 调用数、剩余 KV 时间、Remote Window、完成/回收差值；按 pair/rank 保留证据。
- reaper 满载日志、staging/watermark 积压的诊断证据、session/probe 错误、quarantine、CPU 和主机/GPU 内存峰值。

现有 `pre_layerwise_report.py` 汇总 journal，不是常驻监控服务，也没有自动上线/回滚动作；需接入现场已有日志/指标与发布系统。
`observed_remote_window_ms` 受轮询影响，完成早于 Prefill finalize 是辅助证据；验证真实 GPU 重叠仍需 trace。
启动样例关闭 RPC profiling；性能轮维持一致，归因 trace 单独运行。
只有正确性通过、错误率和尾延迟满足现场 SLO、收益可重复且内存稳定时才扩大流量；不把单次 TTFT 改善当作上线结论。

## 8. 撤流、故障处理与回退

这些配置在进程启动时生效，不支持在在途请求上热切换。

1. 从现有业务入口撤下整个候选 pair，停止新请求；先让已接收请求和 Remote RPC 结束。
2. 确认原生 PD 无在途请求，并检查激活请求全部 released、无错误。不能仅依赖 router 健康状态或 TCP 连接断开判断可释放 KV。
3. 正常 drain 后停止 router/P/D，保存 release、配置、日志、journal 和必要 trace。再修改模式或回退到上一套已验收制品，使用新实例目录并重新预热。
4. 紧急情况下先将新请求切到健康基线 pair。若 timeout/quarantine 无法证明在途 DMA/RPC 已结束，保留故障现场并由现场流程终止相关 P/D 进程后重建整套 pair；不得在存活进程中强制 unregister、复用 KV 或清空 journal。

完整功能回退须同时设置 P/D `HANDOFF_PRE_LAYERWISE=off`、`SGLANG_DISAGG_STAGING_BUFFER=0`、D `HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=0`；只把 `remote` 改成 `layerwise` 仍保留新传输路径。
只关闭接收 host flag 会保留 GPU ring 协议，属于性能对照，不是回退到原生直写基线。
保留 Send Lock、pre-send probe 和已核准 Batch 等传输配置。

历史现场发现过单独重启 D 后首请求的 session 失效窗口，见 [复验记录](validation/5090-new-commits.md)。
本次修复没有提供跨版本滚动升级或透明故障恢复保证；新旧协议不混用，首轮按整个 pair 撤流升级。
probe 可帮助识别失效 peer，但不是“任意重启都无损”的保证。

journal 当前**不自动清理**。需要生产侧监控磁盘容量，归档只针对已停止且确认 drain 的实例目录；不对活跃实例目录做按时间粗删。
退场等待期限需覆盖最长请求和回收时间，超时进入故障流程，不能当作安全回收证明。

## 9. 发布检查记录

每个正式 release 留存：完整 SHA / wheel 校验和 / 模型 revision；加载路径与依赖版本；P/D/router 的有效参数和环境；内存预算；首请求与预热后性能；目标并发正确性和取消/失败/重启结果；撤流回退演练；灰度批准的 SLO 与流量范围。

当前代码的开发机验证限于 CPU 合约、源码 hook 和格式检查。可在空闲目标 GPU 上另跑
`test/manual/remote_attn/pre_layerwise_staging_probe.py --receive-staging --busy-device destination`
验证实际 TE/H2D/scatter 字节与流依赖，但它不启动完整模型、不能替代服务验收，也不应在已接流量 GPU 上运行。
完整探针参数、测试矩阵和指标解释见 [P5.19 说明](P5.19-PRE-LAYERWISE-ZH.md)。

发送端已统一到 `CorePinnedStager`：普通 chunk 与层组共用每 rank 一块 128 MiB pinned slot，插件不再独立分配或注册缓冲区。slot 锁覆盖 D2H 和同步 TE 完成；原有 send lock 仍保留。`SGLANG_MOONCAKE_HOST_STAGING_ALL=1` 覆盖两条发送路径；该开关关闭时，`HANDOFF_PRE_LAYERWISE_HOST_STAGING` 只控制层组。GPU 指针对照需要两者都关闭。all-send 模式在启动发布 KV 张量表时 reserve/预热；仅层组模式在首个选中请求的 forward 前完成。
