# 功能移植 README：PD + D2D（P0–P5.18）

本 README 只讲**功能代码**怎么移植到另一份 SGLang 仓库。  
不包含 `p511_pd_bench.py` / `p514_faults.py` / `p515_*` / `p518_*` 等实验场景驱动。

- 已验证基线：`pd-handoff-p518-stable` / `17261ca0`
- 参考 SGLang base：`3a869635c`
- 最终 `handoff_control.py` md5：`e545304a50a64f64e1f1cf9e679e634f`
- 未完成：P5.19 Pre-Layerwise + Remote；不要把 PR #34905 当成已验证功能

本目录对应文件：

- [handoff_control.py](sglang/handoff_control.py)
- [core_handoff_hooks.patch](sglang/patches/core_handoff_hooks.patch)
- [mooncake_session.patch](sglang/patches/mooncake_session.patch)
- [optional_instrumentation.patch](sglang/patches/optional_instrumentation.patch)

---

## 0. 功能边界

本移植包含：

```text
PD:  P-only + D0/D1 decode-only，Mooncake TCP
D2D: D0 -> D1 freeze / import / commit / release
     Copy / Layer / Remote 三种 transfer
     Fast Freeze / Fast Resume / TP2 all-rank
     Remote Hub / progressive LOCAL
     Controller watchdog / claim / epoch / fence
     Mooncake session send lock + pre-send probe
```

不包含：

```text
测试脚本
报告/数据
P5.19 Pre-Layerwise
Router 生产实现
Gateway SSE 重连续传
自动扩缩容
```

---

## 1. 要复制的文件

### 1.1 SGLang 新增模块

```text
源:  pd-deploy/handoff/handoff_control.py
目标: python/sglang/srt/managers/handoff_control.py
```

这是完整文件，直接复制。目标已有同名文件时先确认不是另一套实现。

### 1.2 SGLang hook

不要整体复制这些文件；按下面的点手工合并。

| 文件 | 必要 | 内容 |
|---|---|---|
| `managers/scheduler.py` | 是 | handoff import、两个 event loop 注入、Hub 初始化/关闭 |
| `disaggregation/decode.py` | 是 | decode event loop 注入 |
| `managers/scheduler_components/request_receiver.py` | 是 | rank0 `cmd.0.json` -> TP broadcast sentinel |
| `managers/schedule_batch.py` | 是 | Fast Resume `handoff_prefix_indices` bind/clear |
| `managers/scheduler_components/output_streamer.py` | 是 | `_handoff_req` 跳过正常输出 |
| `disaggregation/mooncake/conn.py` | 是 | send lock + pre-send probe |
| `model_executor/runner/decode_cuda_graph_runner.py` | 开 graph 时是 | remote 时禁用 graph + step begin |
| `model_executor/runner/prefill_cuda_graph_runner.py` | 开 graph 时是 | target state 时禁用 graph |
| `server_args.py` | P5.10 可选 | `prefill_decode_interval` |

### 1.3 控制面

| 文件 | 必要 | 作用 |
|---|---|---|
| `p2d_driver.py` | 是 | 文件控制协议、TP fan-out |
| `p4_gateway.py` | 是* | SSE、P→D、D2D drain、journal、cancel fence |

\* 如果目标系统已有自己的网关/controller，可以只移植 `p2d_driver.py` 的协议和
`p4_gateway.py` 的 drain/freeze/import 逻辑，不必原样使用这个测试网关。

### 1.4 启动配置

```text
start_p511_p_tp2.sh
start_p511_d0_tp2.sh
start_p511_d1_tp2.sh
```

这些不是测试脚本，而是端口、GPU、TP2、Mooncake TCP、Fast Freeze、
`HANDOFF_CTL_DIR` 的启动配置。目标拓扑不同就改它们，不要把 env 丢了。

### 1.5 不需要

```text
p511_pd_bench.py
p513_pd_probe.py
p514_faults.py
p515_*.py
p517_freeze_validate.py
p518_stress.py
p518_queue_test.py
p4_sse_client.py
p4_bulk_import_test.py
b1_proxy.py / b3_router.py / b3a_* / b3c_*
p55_* / p57_* / p58_progressive.py / p59_run_matrix.sh
swebench_stream.py
```

---

## 2. SGLang hook 合并点

### 2.1 `managers/scheduler.py`

import：

```python
from sglang.srt.managers.handoff_control import (
    close_handoff_hub,
    execute_handoff_control,
    get_remote_hub,
    maybe_handle_handoff_control,
    split_handoff_controls,
)
```

两个 event loop（`event_loop_normal` / `event_loop_overlap`）在
`recv_requests()` 之后：

```python
recv_reqs, handoff_controls = split_handoff_controls(recv_reqs)
self.process_input_requests(recv_reqs)
for handoff_control in handoff_controls:
    execute_handoff_control(self, handoff_control)
maybe_handle_handoff_control(self)
```

`run_scheduler_process()` 创建 Scheduler 后：

```python
if os.environ.get("HANDOFF_CTL_DIR"):
    try:
        get_remote_hub(scheduler)
    except Exception as exc:
        logger.warning("handoff hub init failed: %r", exc)
```

`finally`：

```python
try:
    close_handoff_hub(scheduler)
except Exception:
    pass
```

如果目标仓库没有 `event_loop_normal/overlap` 同名函数，就找
`recv_requests()` 之后的公共 scheduler 入口，把同一组代码放在
`process_input_requests()` 前后；必须保证每个 TP rank 都执行。

### 2.2 `disaggregation/decode.py`

decode event loop 的 `recv_requests()` 之后：

```python
recv_reqs, handoff_controls = split_handoff_controls(recv_reqs)
self.process_input_requests(recv_reqs)
for handoff_control in handoff_controls:
    execute_handoff_control(self, handoff_control)
maybe_handle_handoff_control(self)
```

两个 decode loop（normal/overlap）都要加。

### 2.3 `request_receiver.py`

在 `_broadcast_reqs_across_ranks` 之前：

```python
recv_reqs = self._append_handoff_control(recv_reqs)
recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs)
```

方法：

```python
def _append_handoff_control(self, recv_reqs):
    if not (
        self.ps.pp_rank == 0
        and self.ps.attn_tp_rank == 0
        and self.ps.attn_cp_rank == 0
    ):
        return recv_reqs
    try:
        from sglang.srt.managers.handoff_control import poll_broadcast_control
        ctl = poll_broadcast_control()
    except Exception:
        return recv_reqs
    if ctl is None:
        return recv_reqs
    return list(recv_reqs or []) + [ctl]
```

关键点：**必须在 TP broadcast 之前**。否则 rank0 和 rank1 看到的命令不一致，
下一轮 collective 会死锁。

### 2.4 `schedule_batch.py`

只加三处：

1. 文件顶部：

```python
HANDOFF_PREFIX_CLEARED = 0
```

2. `Req.init_next_round_input()` 的 fill-id 刷新之后：

```python
handoff_prefix = getattr(self, "handoff_prefix_indices", None)
if handoff_prefix is not None:
    self.prefix_indices = handoff_prefix
    self.set_extend_range(
        len(self.prefix_indices), len(self.full_untruncated_fill_ids)
    )
    return
```

3. `ScheduleBatch` extend allocation 之后：

```python
global HANDOFF_PREFIX_CLEARED
for _i, _r in enumerate(reqs):
    _hp = getattr(_r, "handoff_prefix_indices", None)
    if _hp is None:
        continue
    try:
        _pool = self.req_to_token_pool
        _rpi = int(req_pool_indices_cpu[_i])
        _pool.req_to_token[_rpi, : int(_hp.numel())] = _hp.to(
            dtype=_pool.req_to_token.dtype,
            device=_pool.req_to_token.device,
        )
    except Exception as _exc:
        print(f"[handoff-prefix] bind failed {_r.rid}: {_exc!r}", flush=True)
    _r.handoff_prefix_indices = None
    HANDOFF_PREFIX_CLEARED += 1
```

不要整体复制 `schedule_batch.py`：当前 diff 有几千行 CRLF/LF 噪声，真实语义只有这些。

### 2.5 `output_streamer.py`

```python
for req in reqs:
    if req is skip_req:
        continue
    if getattr(req, "_handoff_req", False):
        continue
    ...
```

避免 handoff 请求走正常 SSE output，导致重复/越界输出。

### 2.6 `mooncake/conn.py`

三件事：

1. `MooncakeKVManager.__init__`：

```python
self._send_locks = {}
self._send_locks_guard = threading.Lock()
self._send_global_lock = threading.Lock()
self._send_lock_enabled = (
    os.environ.get("SGLANG_MOONCAKE_SEND_LOCK", "1") != "0"
)
```

2. `_transfer_data`：

```python
with self._session_send_lock(mooncake_session_id):
    return self.engine.batch_transfer_sync(
        mooncake_session_id,
        list(src_addrs),
        list(dst_addrs),
        list(lengths),
    )
```

3. 注册新 decode peer 时 probe：

```python
old = self.decode_kv_args_table.get(mooncake_session_id)
peer_changed = (
    old is None
    or old.dst_kv_ptrs[:1] != decode_kv_args.dst_kv_ptrs[:1]
    or old.dst_kv_item_len != decode_kv_args.dst_kv_item_len
)
if peer_changed:
    probe = getattr(self.engine, "send_probe", None)
    if probe is not None:
        rc = -1
        for _ in range(3):
            try:
                rc = probe(mooncake_session_id)
            except Exception:
                rc = -1
            if rc == 0:
                break
            time.sleep(0.05)
```

`SGLANG_MOONCAKE_SEND_LOCK=0` 只用于 A/B。生产保持默认 1。

如果目标 Mooncake 没有 `send_probe`，probe 会跳过，P5.14 的 D 重启
broken pipe 修复不会完整生效。修复不会完整生效。

### 2.7 CUDA Graph hook

只有启用 graph/overlap 时才需要。

`decode_cuda_graph_runner.py`：

```python
def can_run_graph(self, forward_batch):
    try:
        from sglang.srt.managers import handoff_control as _hc
        if _hc.has_remote_work():
            return False
    except Exception:
        pass
    ...
```

replay 前：

```python
try:
    from sglang.srt.managers import handoff_control as _hc
    _hc.graph_step_begin(forward_batch)
except Exception:
    pass
```

`prefill_cuda_graph_runner.py`：

```python
def can_run_graph(self, forward_batch):
    try:
        from sglang.srt.managers import handoff_control as _hc
        if _hc.has_target_states():
            return False
    except Exception:
        pass
    ...
```

`handoff_control.py` 里对应函数：

```python
def has_target_states() -> bool:
    return bool(_TARGET_STATES)

def has_remote_work() -> bool:
    if os.environ.get("HANDOFF_GRAPH_DISABLE_ANY_STATE", "0") == "1":
        return bool(_TARGET_STATES)
    for st in _TARGET_STATES:
        if getattr(st, "mode", None) == "remote":
            if any(not e.is_set() for e in st.events):
                return True
    return False

def graph_step_begin(forward_batch) -> None:
    try:
        _row_meta(forward_batch)
    except Exception:
        pass
```

P5.18 配置 `--disable-cuda-graph`，这些 hook inert；P3/以后开 graph 时必须保留。

### 2.8 `server_args.py`（只 P5.10）

```python
prefill_decode_interval: A[
    int,
    "Number of consecutive decode steps to run after each prefill/chunk "
    "before scheduling the next prefill. 0 disables interleaving.",
    NS("schedule"),
] = 0
```

不加也能跑，只是没有 P5.10 的 decode priority。

---

## 3. 控制面协议

### 3.1 目录

```text
HANDOFF_CTL_DIR       gateway/controller 根目录
  p/                  P-only
  src/                D0 source
  dst/                D1 target
```

P/D0/D1 的 `HANDOFF_CTL_DIR` 必须分别指向 `p/src/dst`；gateway 指向根目录。

### 3.2 命令流

```text
p4_gateway
  -> p2d_driver.send(role, {"cmd": ...})
  -> role/cmd.0.json
  -> rank0 request_receiver.poll_broadcast_control()
  -> TP broadcast sentinel
  -> execute_handoff_control()
  -> role/result.<rank>.json
  -> p2d_driver 聚合返回
```

没有 `p2d_driver` 或等价 controller，`handoff_control` 不会收到任何命令。

### 3.3 必要 env

Gateway：

```bash
HANDOFF_CTL_DIR=$CTL
HANDOFF_PD_PREFILL_URL=http://127.0.0.1:31016
HANDOFF_PD_BOOTSTRAP_HOST=127.0.0.1
HANDOFF_PD_BOOTSTRAP_PORT=35409
HANDOFF_ADMISSION_POLICY=decoupled
```

P/D：

```bash
MC_FORCE_TCP=1
SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES=8
MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=8192
MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=8192
SGLANG_ENABLE_FAILED_SESSION_PROBE=1
SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=1.0
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
SGLANG_MOONCAKE_SEND_LOCK=1
```

D0 额外：

```bash
HANDOFF_FAST_FREEZE=1
HANDOFF_FAST_BUFFERS=2
HANDOFF_FAST_COPY_STREAMS=2
HANDOFF_FAST_ASYNC=1
HANDOFF_REMOTE_HIGH_PRIORITY=1
HANDOFF_REMOTE_FALLBACK_MS=10000
```

---

## 4. 移植步骤

### Step 1：复制模块

```bash
cp handoff_control.py \
   "$SGL/python/sglang/srt/managers/handoff_control.py"
md5sum "$SGL/python/sglang/srt/managers/handoff_control.py"
# e545304a50a64f64e1f1cf9e679e634f
```

### Step 2：合并 hook

同 base `3a869635c` 可参考 patch：

```bash
cd "$SGL"
git apply --check --ignore-whitespace \
  sglang/patches/core_handoff_hooks.patch \
  sglang/patches/mooncake_session.patch \
  sglang/patches/optional_instrumentation.patch
```

不同 base 按第 2 节手工合并。`--ignore-whitespace` 是必要的，
`schedule_batch.py` 有 CRLF/LF 混杂。

### Step 3：编译/import

```bash
python -m py_compile \
  "$SGL/python/sglang/srt/managers/handoff_control.py" \
  "$SGL/python/sglang/srt/managers/scheduler.py" \
  "$SGL/python/sglang/srt/disaggregation/decode.py"

PYTHONPATH="$SGL/python" python -c \
  "from sglang.srt.managers.handoff_control import get_remote_hub; print('ok')"
```

### Step 4：启动 P/D0/D1

```bash
cd "$HANDOFF_HOME"
bash start_p511_p_tp2.sh  > "$STATE/p.log"  2>&1 &
bash start_p511_d0_tp2.sh > "$STATE/d0.log" 2>&1 &
bash start_p511_d1_tp2.sh > "$STATE/d1.log" 2>&1 &
```

检查：

```bash
for p in 31016 31017 31018; do
  curl -s -o /dev/null -w "$p=%{http_code}\n" http://127.0.0.1:$p/health
done
ls "$CTL/src/ready."*.json "$CTL/dst/ready."*.json
```

### Step 5：启动 gateway

```bash
HANDOFF_CTL_DIR="$CTL" \
HANDOFF_PD_PREFILL_URL=http://127.0.0.1:31016 \
HANDOFF_PD_BOOTSTRAP_HOST=127.0.0.1 \
HANDOFF_PD_BOOTSTRAP_PORT=35409 \
HANDOFF_ADMISSION_POLICY=decoupled \
python p4_gateway.py --port 31080
```

检查：

```bash
curl -s http://127.0.0.1:31080/v1/router/status
```

### Step 6：最小功能 smoke

不用实验场景驱动，只手动跑一次：

1. POST `/v1/completions`，body 包含
   `rid/prompt_ids/max_new_tokens/cut/transfer/worker=src`；
2. 读到若干 token 后 POST `/v1/drain`：
   `{"targets":["dst"],"mode":"d2d","transfer":"remote"}`；
3. 轮询 `/v1/drain/status`，确认进入 `SAFE_TO_STOP` 或预期终态；
4. 检查：
   - source/target health 200
   - source 输出 token 与自然参考一致
   - target 不重复输出
   - source/target `handoff_registry` 归零
   - target reservation / lease 归零

这一步通过，说明功能链路已经通了；然后再考虑用 `p518_stress.py` 做完整验证。

---

## 5. 常见失败定位

| 现象 | 检查 |
|---|---|
| 进程启动但没有 handoff 日志 | `HANDOFF_CTL_DIR` 是否设置；`scheduler.py` 的 `get_remote_hub` hook 是否在 |
| 控制命令一直不执行 | `request_receiver._append_handoff_control` 是否在 TP broadcast 之前；`cmd.0.json` 是否写到正确 role 目录 |
| rank0 恢复、rank1 卡死 | 只改了 rank0；sentinel 没走 TP broadcast |
| Fast Resume 后 token 错/重复 | `schedule_batch.py` 的 `handoff_prefix_indices` bind/clear 没合并 |
| D 重启后首个 chunk broken pipe | `mooncake/conn.py` 的 probe 没移植，或 Mooncake 没有 `send_probe` |
| Remote 层结果错/卡住 | GPU graph 开着但没加 `has_remote_work`/`has_target_states` |
| 控制面调用超时 | `p2d_driver` 的 `CTL` 与 SGLang 的 `HANDOFF_CTL_DIR` 不一致 |
| P5.10 decode priority 无效 | `server_args.py` 的 `prefill_decode_interval` 或 scheduler defer hook 没合并 |

---

## 6. 最后提醒

- **只移植功能**时，不需要 `p511_pd_bench.py`、`p514_faults.py`、`p515_*`、`p518_*`。
- **必须保留** `p2d_driver.py` + `p4_gateway.py` 或等价的 controller，否则没有 freeze/import/commit/drain 的发起者。
- **不要整体覆盖** `scheduler.py` / `schedule_batch.py` / `decode.py`；目标仓库版本不同会破坏上游逻辑。
- **P5.19 不包含**：当前没有 Pre-Layerwise 实现；PR #34905 只是参考。
- 完整实验复现另见 [`README.md`](README.md) 和 [`MANIFEST.tsv`](MANIFEST.tsv)。
