# Remote-attn 内网部署手册

本文对应 SGLang 的 `remote-attn` 分支，首次功能移植提交为 `a67ca2bb36`，部署应使用分支最新版本。
后续内网反馈与修复见 [5090 等价性验证记录](validation/5090-intranet.md) 和
[新提交复验与 KV 重复释放修复](validation/5090-new-commits.md)。
默认部署沿用 handoff P0–P5.18：**单机 6 张 GPU，P / D0 / D1 各 TP2，Qwen3-8B，Mooncake TCP**。
命令中的 `/workspace/sglang`、Python 和模型目录需要与内网机器实际路径一致。

代码已完成移植，已有 38 项 CPU 回归和此前 4060 上 10 项 CUDA/Mooncake/Remote RPC 检查。
内网反馈确认 C1/C4 token 一致、故障矩阵及新提交下 TP2 稳定性 23/23；
单独重启 D 后首请求的 session 失效窗口仍未解决，详见上述复验记录。
4060 是单卡 8 GB 的验证环境，不能按本文默认配置同时部署三个 TP2 实例。

## 1. 需要带入内网的代码

携带整个 `remote-attn` checkout，或由该 checkout 构建的 SGLang wheel 加完整源码快照。
只拷贝插件或旧版 `handoff_control.py` 不够：当前 main 的接口与原实验分支不同，
本分支还包含核心环境变量声明，源码安装也可能需要 `rust/`、`proto/` 等构建输入。

| 路径 | 用途 | 部署要求 |
| --- | --- | --- |
| `python/` 及仓库构建输入 | 本分支 SGLang runtime | 安装本分支，不能替换成原实验 `3a869635c` 或任意 PyPI SGLang |
| `python/sglang/srt/environ.py` | 声明 `SGLANG_MOONCAKE_SEND_LOCK` | 已在分支中，不要漏掉 |
| `plugins/remote_attn/sglang_remote_attn/` | 渐进式 KV 传输、Remote attention、生命周期与 16 个 hook | 安装插件包 |
| `plugins/remote_attn/pyproject.toml` | `sglang.srt.plugins` entry point | 与插件源码一起保留 |
| `plugins/remote_attn/control_plane/p2d_driver.py` | 文件控制协议、TP 结果聚合 | 保留原包版本 |
| `plugins/remote_attn/control_plane/p4_gateway.py` | SSE、请求提交、D2D drain、journal | 使用本分支的 C1 epoch 修复版 |
| `plugins/remote_attn/deploy/` | 公共 env 和三个功能启动脚本 | 整个目录一起带入 |
| `test/manual/remote_attn/cuda_handoff_probe.py` | 单 GPU CUDA/Mooncake 验证 | 建议带入做环境自检 |
| `migration/`、`REVIEW.md`、`FUNCTIONAL-PARITY.md`、`validation/` | 原包来源、适配边界及证据 | 留作核对；不是运行时补丁 |

最后一行中的路径均相对于 `plugins/remote_attn/`。
`migration/core_handoff_hooks.patch` 和 `migration/mooncake_session.patch` 是原实验参考文件，
**不要再向当前分支 apply**。

## 2. Mooncake 对应代码与版本

不需要同步或修改 Mooncake 源码仓库，也没有新的 Mooncake C++ patch。
两部分改动都在本分支：

- 原生 P→D：`sglang_remote_attn/hooks.py` 包装 SGLang 的 Mooncake connector，提供发送锁和 peer 注册前 probe。
- D0→D1：`sglang_remote_attn/handoff_control.py` 调用 wheel 的 `TransferEngine`，
  管理 CPU 内存注册、同步/异步写入、完成等待、逐层通知、目标 slot ACK 和 CUDA stream。

| 项目 | 要求 |
| --- | --- |
| Python | 插件要求 Python 3.12 或以上；原实验和 4060 验证使用 3.12 |
| Mooncake 包 | `mooncake-transfer-engine-cuda13==0.3.13.post1` |
| CUDA | 本文使用 CUDA 13 环境，GPU 驱动必须能运行所安装的 CUDA runtime |
| Mooncake API | 必须有 `TransferEngine.send_probe`；缺失时插件拒绝启动 |
| 传输方式 | 单机 TCP，`MC_FORCE_TCP=1`，SGLang backend 为 `mooncake_tcp` |
| SGLang 依赖 | 以本分支 `python/pyproject.toml` 为准 |

当前分支声明的核心依赖包括 `torch==2.14.1`、`flashinfer_python[cu13]==0.7.2`。
2026-10-28 内网完整模型复验使用 `torch==2.14.1+cu130`、`torchvision==0.29.0+cu130`、
`sglang-kernel==0.4.10`、FlashInfer `0.7.2`，已移除 `SKIP_SGL_KERNEL_VERSION_CHECK`。
现场 kernel 0.4.9/0.4.10 配合 torch 2.13 曾发生 `CUDAErrorLogCapture` 符号错误；
应使用上述配套组合并执行 `pip check`，不要用跳过版本检查掩盖 ABI 不兼容。
4060 的轻量 probe 使用 PyTorch `2.13.0+cu130` / FlashInfer `0.6.18`，
它验证了插件的实际 CUDA/TE 运算，**不是完整 SGLang 部署的依赖锁定文件**。
不要把其简化环境直接当成已验收的模型服务器环境。

完成第 4 节的公共变量配置和安装后，启动前检查版本和 API。
先导入 PyTorch，以加载 wheel 所需的 CUDA 动态库：

```bash
"$HANDOFF_PYTHON" - <<'PY'
import importlib.metadata as metadata
import torch
from mooncake.engine import TransferEngine

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("torchvision:", metadata.version("torchvision"))
print("sglang-kernel:", metadata.version("sglang-kernel"))
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Mooncake:", metadata.version("mooncake-transfer-engine-cuda13"))
print("send_probe:", callable(getattr(TransferEngine, "send_probe", None)))
PY
```

期望 Mooncake 为 `0.3.13.post1`、`send_probe=True`、CUDA 可用。
不要同时安装多个提供 `mooncake` 模块的不同 wheel；若环境已有其他 Mooncake 包，先在独立环境中确认包来源。

## 3. 代码与依赖如何进入内网

### 3.1 内网已有 Git / Python 包镜像

在已配置内网 `origin` 的仓库执行：

```bash
cd /workspace/sglang
git fetch origin
git switch remote-attn
git pull --ff-only
git log -1 --oneline
```

内网仓库需要先同步外部 `remote-attn` 分支。
不要切到旧实验分支 `feat/dsv4-pdmux-opt`，也不要把参考 patch 硬打到其他 main 上。

### 3.2 完全离线：联网中转机准备交付包

以下操作在与部署机架构、Python ABI 和 Linux 环境匹配的构建机执行，
例如 Linux x86_64 / CPython 3.12 / CUDA 13。不能用 macOS wheel 替代 Linux wheel。
构建机需要完整 CUDA、Rust 等本分支构建工具；建议复用已具备这些依赖的构建镜像。

```bash
cd /path/to/sglang
git switch remote-attn

export HANDOFF_DELIVERY=/tmp/remote-attn-delivery
mkdir -p "$HANDOFF_DELIVERY/wheelhouse"

git rev-parse HEAD > "$HANDOFF_DELIVERY/REVISION"
git archive --format=tar.gz \
  --output="$HANDOFF_DELIVERY/sglang-remote-attn.tar.gz" HEAD

python -m pip wheel --wheel-dir "$HANDOFF_DELIVERY/wheelhouse" \
  ./python ./plugins/remote_attn \
  'mooncake-transfer-engine-cuda13==0.3.13.post1'

cd "$HANDOFF_DELIVERY"
sha256sum sglang-remote-attn.tar.gz wheelhouse/*.whl > SHA256SUMS
```

`wheelhouse` 应是本次交付专用目录，只放对应平台、对应分支的一组构建结果，避免混入旧 SGLang wheel。
该步骤在联网机器上完成依赖解析与 wheel 构建；内网安装时不再现场编译 Rust 扩展。
另行准备完整模型及 tokenizer 文件、CUDA/驱动环境，以及可能用到的 FlashInfer JIT/cache 输入。
仅下载 Mooncake 一个 wheel 不代表其他依赖已经齐全。

将交付目录和模型目录转入内网后：

```bash
cd /path/to/remote-attn-delivery
sha256sum -c SHA256SUMS

mkdir -p /workspace/sglang
tar -xzf sglang-remote-attn.tar.gz -C /workspace/sglang
```

源码快照应解压到新目录，不覆盖正在使用的旧 checkout。
Git archive 不包含独立 submodule checkout；若所选构建需要它们，应在构建机准备完整输入并产出 wheel。

## 4. 配置公共路径并安装

后续命令在同一个 Bash 终端执行；换终端需要重新导出这些变量。
使用独立的 serving Python 环境，例如预先创建的 conda 环境：

```bash
export SGLANG_ROOT=/workspace/sglang
export HANDOFF_PYTHON=/opt/conda/envs/sglang/bin/python
export HANDOFF_MODEL_PATH=/workspace/models/Qwen3-8B
export HANDOFF_CTL_ROOT=/workspace/handoff_ctl_p511
export HANDOFF_RUN_DIR=/workspace/remote_attn_run

mkdir -p "$HANDOFF_RUN_DIR" /workspace/handoff_state
mkdir -p "$HANDOFF_CTL_ROOT"/{p,src,dst}
cd "$SGLANG_ROOT"
```

有内网包镜像时，从本分支安装：

```bash
"$HANDOFF_PYTHON" -m pip install -e ./python
"$HANDOFF_PYTHON" -m pip install --no-deps -e ./plugins/remote_attn
"$HANDOFF_PYTHON" -m pip install 'mooncake-transfer-engine-cuda13==0.3.13.post1'
"$HANDOFF_PYTHON" -m pip check
```

完全离线时，安装上一节构建的 wheel；与源码安装二选一：

```bash
export HANDOFF_WHEELHOUSE=/path/to/remote-attn-delivery/wheelhouse

"$HANDOFF_PYTHON" -m pip install --no-index --find-links "$HANDOFF_WHEELHOUSE" \
  "$HANDOFF_WHEELHOUSE"/sglang-*.whl \
  "$HANDOFF_WHEELHOUSE"/sglang_remote_attn-0.1.0-*.whl \
  'mooncake-transfer-engine-cuda13==0.3.13.post1'
"$HANDOFF_PYTHON" -m pip check
```

wheel 安装后，修改 checkout 中的 Python 文件不会自动更新已安装包，需要重新构建/安装。
不要从其他 wheel 源覆盖本分支 SGLang；用下面的输出核对实际加载路径：

```bash
"$HANDOFF_PYTHON" - <<'PY'
import importlib.metadata as metadata
import sglang
import sglang_remote_attn

print("sglang:", sglang.__file__)
print("plugin:", sglang_remote_attn.__file__)
for entry in metadata.entry_points(group="sglang.srt.plugins"):
    print(entry.name, entry.value)
PY

"$HANDOFF_PYTHON" -m sglang_remote_attn.check
```

预检查应输出 `hooks: 16`、`send_probe: true`、`plugin: remote_attn`。
它检查安装和 hook 接口，不会启动模型或证明 TP2 推理正确。
无完整 GPU 依赖的机器只能执行源接口检查：

```bash
PYTHONPATH=plugins/remote_attn "$HANDOFF_PYTHON" \
  -m sglang_remote_attn.check --source-only --sglang-root "$SGLANG_ROOT"
```

## 5. 控制面保留的固定路径

控制面保留原协议和目录，网关增加了 C1 epoch 修复。下列配置不能只靠 server 的 env 自动覆盖：

- 网关 tokenizer 的 `MODEL` 固定为 `/workspace/models/Qwen3-8B`。
  本手册使用该目录；若换模型，需要同步适配控制面，不能只改 `HANDOFF_MODEL_PATH`。
- journal 和部分 trace 固定写到 `/workspace/handoff_state`，需要可写且保留到实验结束。
  `--out` 只改变网关输出文件，不会改变 journal 目录。
- 网关保留了对 `/workspace/5090_handoff_m1` 的 Python import 搜索路径。
  若该目录有旧 `p2d_driver.py`，可能优先加载旧驱动；确认其与当前 `control_plane/p2d_driver.py`
  内容一致，或在独立部署环境中消除旧副本冲突，再启动网关。

同一固定 state 目录不要同时运行多套独立实验。
不要在进程存活、存在冻结请求或迁移未完成时清空控制文件、journal、reservation 或 lease。
换模式重新做独立实验时，先正常结束并保存上一轮状态，再使用干净的独立实验环境。

## 6. 启动 P、D0、D1

| 实例 | 默认 GPU | HTTP 端口 | 控制目录 | 角色 |
| --- | --- | --- | --- | --- |
| P | 0、1 | 31016 | `$HANDOFF_CTL_ROOT/p` | Prefill |
| D0 / src | 2、3 | 31017 | `$HANDOFF_CTL_ROOT/src` | 初始 Decode、迁移源 |
| D1 / dst | 4、5 | 31018 | `$HANDOFF_CTL_ROOT/dst` | 迁移目标 |
| Gateway | 无专用 GPU | 31080 | `$HANDOFF_CTL_ROOT` | 接收请求、SSE、迁移控制 |

P/D 使用 bootstrap 端口 `35409`。当前 control/TE/RemoteHub 还使用 loopback 动态端口，
因此所有角色必须处在能互通 `127.0.0.1` 和共享控制文件的同一部署环境中。
不要直接把这些配置当作跨主机部署；容器若分开运行，也要相应解决网络 namespace 和目录共享。

```bash
cd "$SGLANG_ROOT"

CUDA_VISIBLE_DEVICES=0,1 HANDOFF_CTL_DIR="$HANDOFF_CTL_ROOT/p" \
  bash plugins/remote_attn/deploy/start_p511_p_tp2.sh \
  > "$HANDOFF_RUN_DIR/p.log" 2>&1 &
echo $! > "$HANDOFF_RUN_DIR/p.pid"

CUDA_VISIBLE_DEVICES=2,3 HANDOFF_CTL_DIR="$HANDOFF_CTL_ROOT/src" \
  bash plugins/remote_attn/deploy/start_p511_d0_tp2.sh \
  > "$HANDOFF_RUN_DIR/d0.log" 2>&1 &
echo $! > "$HANDOFF_RUN_DIR/d0.pid"

CUDA_VISIBLE_DEVICES=4,5 HANDOFF_CTL_DIR="$HANDOFF_CTL_ROOT/dst" \
  bash plugins/remote_attn/deploy/start_p511_d1_tp2.sh \
  > "$HANDOFF_RUN_DIR/d1.log" 2>&1 &
echo $! > "$HANDOFF_RUN_DIR/d1.pid"
```

脚本自动加入 `SGLANG_PLUGINS=remote_attn` allowlist，保留已有其他插件项，并先运行 preflight。
真正启用 runtime hook 还要求该角色设置 `HANDOFF_CTL_DIR`；只安装插件不会改变普通 serving。

三个实例都使用 BF16、FlashInfer、page size 1、context length 40960、max-running-requests 8、
chunked-prefill-size 2048；P 的静态显存比例为 0.85，D 为 0.75。
显存不足时应根据模型和负载调整参数，不能假设任意 6 张 GPU 都能容纳原配置。
可以在脚本后追加 CLI 参数，例如 `--context-length 8192`，但不能覆盖下面的兼容性约束。

必须保留：

```text
--tp 2
--page-size 1
--disable-radix-cache
--disable-cuda-graph
--disable-overlap-schedule
--disaggregation-transfer-backend mooncake_tcp
```

三个角色 TP 宽度应相同。当前仅支持 dense NHD MHA FP16/BF16；
PP/DP/CP/DCP、MLA/混合或量化 KV、HiCache、speculative decoding、mixed chunks、unified KV 和 live PD role switching 不在支持范围。
禁用的是 scheduler overlap；插件自己的 KV 传输/计算重叠仍然启用。

## 7. 必须保留的环境变量

`deploy/common.sh` 已设置以下默认值，正常使用启动脚本时不必重复手工导出：

```bash
MC_FORCE_TCP=1
SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES=2048
MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=8192
MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=8192
SGLANG_ENABLE_FAILED_SESSION_PROBE=1
SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=1.0
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
SGLANG_MOONCAKE_SEND_LOCK=1
HANDOFF_PROFILE=0
HANDOFF_RPC_PROFILE=1
HANDOFF_REMOTE_FALLBACK_MS=10000
```

D0 额外启用：

```bash
HANDOFF_FAST_FREEZE=1
HANDOFF_FAST_BUFFERS=2
HANDOFF_FAST_COPY_STREAMS=2
HANDOFF_FAST_ASYNC=1
HANDOFF_REMOTE_HIGH_PRIORITY=1
```

env 允许外部覆盖；有旧实验变量时，应先核对有效值，避免意外关闭 Fast Freeze、锁或 probe。
D0/D1 默认 `HANDOFF_RETRACTION_BACKUP=none`，显式使用
`--disaggregation-decode-retraction-backup none`。
这与内网验证配置一致：原生 decode 因显存压力发生 retraction 时会返回 503，不保留 KV backup。
该开关不关闭 handoff 的 host 传输缓冲区，也不能替代重复释放修复。
可通过 `HANDOFF_RETRACTION_BACKUP=cpu_tensor` 或 `host_pool` 覆盖；这些覆盖配置需另做容量和故障验证。

尤其要检查旧 shell 中是否仍有 `SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES=8`。
旧实验 connector 不使用该开关，新 main 会按它拆成多次同步调用，32K 请求因此增加数千次往返。
当前默认 `2048` 与 `--chunked-prefill-size 2048` 对齐；部署更新后要重启实例并核对有效值。

```bash
export SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES=2048
```

修改 chunk size 时，需要同步评估批次上限与 TE 队列容量；不能把任意更大的值直接视为已验证配置。
公共脚本还提供 `LD_PRELOAD`、`LD_LIBRARY_PATH`、`LIBRARY_PATH` 默认路径；
如内网 CUDA/libstdc++ 安装位置不同，在启动前导出正确值。
现有 `LD_LIBRARY_PATH` 会整体保留，脚本不会自动给它补 CUDA 路径。

## 8. 就绪检查与网关启动

等待模型完成加载，逐个检查：

```bash
curl -f http://127.0.0.1:31016/health
curl -f http://127.0.0.1:31017/health
curl -f http://127.0.0.1:31018/health

ls "$HANDOFF_CTL_ROOT"/{p,src,dst}/ready.*.json
tail -n 40 "$HANDOFF_RUN_DIR/d0.log"
tail -n 40 "$HANDOFF_RUN_DIR/d1.log"
```

每个角色应有 `ready.0.json`、`ready.1.json`，其中 `tp_size=2`。
ready 文件不能替代 `/health`：旧文件可能来自上一轮，需要结合 pid、started_at 和当前进程确认。

启动网关：

```bash
cd "$SGLANG_ROOT"

HANDOFF_CTL_DIR="$HANDOFF_CTL_ROOT" \
HANDOFF_PD_PREFILL_URL=http://127.0.0.1:31016 \
HANDOFF_PD_BOOTSTRAP_HOST=127.0.0.1 \
HANDOFF_PD_BOOTSTRAP_PORT=35409 \
HANDOFF_ADMISSION_POLICY=decoupled \
"$HANDOFF_PYTHON" -u plugins/remote_attn/control_plane/p4_gateway.py \
  --port 31080 --out "$HANDOFF_RUN_DIR/gateway-events.jsonl" \
  > "$HANDOFF_RUN_DIR/gateway.log" 2>&1 &
echo $! > "$HANDOFF_RUN_DIR/gateway.pid"

curl -f http://127.0.0.1:31080/v1/router/status
```

网关使用控制根目录，三个 SGLang 实例使用各自的 `p/src/dst` 子目录。
`HANDOFF_PD_PREFILL_URL` 使新请求走真实 P→D 流程；漏设后不再是同一套严格 PD 实验。
HTTP 返回 200 还需检查状态内容和日志，确认 worker 已就绪、没有遗留 drain/错误。

## 9. 单请求迁移 smoke

通过网关发送请求，D0 生成约 32 个输出 token 后自动尝试切到 D1：

```bash
curl -N http://127.0.0.1:31080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "rid": "remote-demo-001",
    "prompt": "请详细解释大语言模型推理中的 KV Cache 及其迁移过程。",
    "max_new_tokens": 256,
    "worker": "src",
    "cut": 32,
    "transfer": "remote"
  }'
```

`rid` 每次用新的值，避免与 journal 中旧请求冲突；`cut` 是触发检查的输出 token 阈值，
实际冻结点以调度器返回的状态为准。
成功时 SSE 中先出现 `owner: src`，随后出现 `owner: dst`，并正常结束；核对 `seq` 不重复、不跳号及日志无错误。
该实验网关返回 token ID/owner 等事件，不是标准 OpenAI 文本流或 chat API。

| 请求中的 `transfer` | 行为 |
| --- | --- |
| `mooncake` | Copy：完整 KV 到达后继续 |
| `layer` | Layer：按层等待 KV 到达 |
| `remote` | Remote：未到达的 prefix 在源端计算，目标合并本地 tail；到达的层逐步转为 LOCAL |

`remote` 配合 Fast Freeze，源端逐层 D2H/TE 与目标计算并行，目标完成 H2D 后再切 LOCAL。
先分别验证三种模式，再做并发或长上下文实验；成功返回请求不等于已证明输出数值一致性。

## 10. 将 D0 的运行中请求整体迁走

这是独立于上节单请求自动切换的 drain 场景。
先在一个终端创建长请求，将 `cut` 设得大于输出上限，避免先触发单请求自动迁移：

```bash
curl -N http://127.0.0.1:31080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "rid": "drain-demo-001",
    "prompt": "请详细介绍分布式推理系统的调度、KV Cache 和容错。",
    "max_new_tokens": 2048,
    "worker": "src",
    "cut": 1000000,
    "transfer": "remote"
  }'
```

看到 D0 输出后，在另一个已设置相同公共变量的终端执行：

```bash
curl -f http://127.0.0.1:31080/v1/drain \
  -H 'Content-Type: application/json' \
  -d '{"targets":["dst"],"mode":"d2d","transfer":"remote"}'

curl -f http://127.0.0.1:31080/v1/drain/status
```

关注 `state`、`pending`、`failed`、`blocked_reason`、`last_error`、
`source_safe_to_stop` 和 `source_clean_all_ranks`。
没有运行中请求时，drain 不能证明实际发生过 KV 迁移。

只有确认状态为 `SAFE_TO_STOP`，且源端所有 rank 已清理、没有待处理失败时，才停止 D0。
`BLOCKED` 或控制请求超时不表示目标没有提交；不要据此强行释放源 KV。

```bash
# 仅在检查上述状态且本轮 D0 PID 仍匹配当前进程后执行。
curl -f -X POST http://127.0.0.1:31080/v1/drain/stop
kill -TERM "$(cat "$HANDOFF_RUN_DIR/d0.pid")"
```

`/v1/drain/stop` 只把控制面状态标记为 STOPPED，不会自动杀掉 D0。
保留 P、D1、网关及其状态供后续请求使用，并检查客户端流仍能正确结束。
重复 drain 是幂等的，旧 journal 中的终态可能被继续返回；它不是自动重置实验的接口。

## 11. 部署验收与排错

| 现象 | 检查 |
| --- | --- |
| 缺少 `SGLANG_MOONCAKE_SEND_LOCK` | 是否加载了本分支 SGLang，是否误装普通发行 wheel |
| 找不到插件 entry point / hook 不齐 | 插件与 server 是否用同一个 Python，启动时是否启用插件和角色控制目录 |
| 缺少 `send_probe` | 核对 Mooncake distribution、版本和实际导入路径，不能静默跳过 |
| `libcudart.so.13` / libstdc++ 加载失败 | 核对 CUDA runtime、`LD_LIBRARY_PATH`、`LD_PRELOAD`；API 自检先导入 PyTorch |
| FlashInfer 编译失败 | 检查 ninja、nvcc、CUDA_HOME 和缓存；4060 系统旧 nvcc 12.0 不能编译其 CUDA 13 配置 |
| 网关模型/tokenizer 加载失败 | 原控制面固定模型目录是否完整，是否错把 server 模型 env 当作网关配置 |
| 命令不执行 | 根目录和 `p/src/dst` 是否一致，每个 rank 是否活着，是否加载了旧 p2d_driver |
| TP command diverged | 保存各 rank 的 result 和日志；引擎主动停止不一致调度，不能靠重发命令认定已自动恢复 |
| `quarantined_transfers` 非空 | 完成凭据或 native unregister 尚未成功时保留；scheduler 会重试回收。不能确认远端完成或 GPU fence 失败时仍需故障恢复 |
| Remote 本地缺层 / `remote downgrade` | 有可用 client 且未取消、GPU fence 安全时继续远端读缺失层；不能将错误唤醒当作 LOCAL-ready |
| drain 长期 BLOCKED | 结合 target commit、epoch、fence、reservation、lease 和所有 rank 的事实排查，不能直接清 journal |

内网至少完成以下验收，再判断适配可用于目标部署：

1. P/D0/D1 的 health 和所有 rank ready；普通 PD 请求成功。
2. Copy、Layer、Remote 的单请求及并发迁移，逐 token/logprob 对比不迁移参考；没有重复输出。
3. 正常结束、取消、pre-commit abort、resume 后，row/token/reservation/lease/registration 的计数符合预期。
4. 真正双 GPU TP2/NCCL，controller/D 重启和 session churn，检查各 rank 一致性及故障边界。
5. GPU trace 检查实际业务下 D2H、TE、H2D 与 attention 的重叠和逐层 LOCAL；不能用单卡 probe 推断吞吐提升。

可先在单卡环境执行已有 CUDA probe：

```bash
cd "$SGLANG_ROOT"
PYTHONPATH=plugins/remote_attn "$HANDOFF_PYTHON" \
  test/manual/remote_attn/cuda_handoff_probe.py \
  --output "$HANDOFF_RUN_DIR/cuda-probe" --profile
```

它包含 2 项 attention 比较、5 组实际 CUDA/Mooncake 传输、2 组缺层 Remote RPC 数值比较和 1 项取消 fence 检查，
使用小型 KV storage fixture，不启动完整模型服务。与服务压测分开运行，避免争用 GPU。

## 12. 当前功能边界和参考记录

当前适配范围为收到的 P0–P5.18 功能包；P5.19 pre-layerwise、TP Graph-on、跨主机、
生产 router、通用 SSE 重连续传和自动扩缩容不作为本次交付能力。
TP 命令一致性检查在分歧时停止引擎，不是跨 rank 原子回滚协议；故障路径可能保留内存直到进程退出。

**D 重启注意事项：**目前单独重启 D 后立即请求可能因旧 Mooncake session 返回 500，
后台 probe 恢复只影响后续请求。维护时按内网规避方法配套重启 P 与 D，
完成实际 KV 传输请求验证后再恢复流量；health 200 不等于传输链路已恢复。
该窗口尚未修复，不要将 KV 重复释放修复解释为无损滚动重启支持。

- [插件总览与安装边界](README.md)
- [原包功能对应与来源差异](FUNCTIONAL-PARITY.md)
- [复查问题与修复](REVIEW.md)
- [4060 验证记录](validation/4060.md)
- [4060 原始结构化结果](validation/4060-results.json)
- [5090 内网等价性与故障反馈、修复说明](validation/5090-intranet.md)

原实验 SGLang 基线为 `feat/dsv4-pdmux-opt@3a869635c` 的 dirty working tree，
Mooncake 实验代码参考为 `pd-handoff-p518-stable@17261ca0`。
实际收到的 runtime MD5 与原文档声明不同，详见功能对应表；本部署使用已适配的 `remote-attn` 分支，
不依赖重新找到原 dirty tree 或重新应用迁移 patch。

C1 及 host 回收更新详见 [修复说明](validation/C1-RECLAIM.md)。升级时同时更新并重启网关与 P/D 实例，
保留 journal；`HANDOFF_CTL_ROOT` 必须共享，新增 `p/src/dst/transfer-completion/` 用于已完成传输的回收凭据。
不要在进程存活时删除未确认完成的凭据。
