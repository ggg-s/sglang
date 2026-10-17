# DSV4 SA lease 回收修复与内网验收

## 源码结论

`58ae30ac69` 的 `begin_prefill` 用 `len(states)` 限制活跃 lease 数。
正常 decode 完成会调用 `BaseSpecWorker.note_request_finished`，插件随即
`close_request(..., retain=True)`：活跃 lease 从 `states` 删除，已安装的历史
才进入独立 `shared` 队列。因此 `states` 并非正常混入了所有 global-pool 来源。

实际漏口在 `SchedulerBatchResultProcessor.process_batch_result_prefill`：
请求若在 prefill 的第一个 token 就完成，该路径释放 KV，却没有调用 decode 的
finish hook。插件留下 lease，即使请求对象后来被销毁，也没有回收弱引用对应的槽。
这与 health/warmup 后 `max_requests=1` 永久失效的现象一致；具体内网遗留请求
身份没有远端证据，本次不声称已复核。c=4 同样会逐步丢失可用容量。

修复在每次非 idle 的 `begin_prefill` 分配前、锁内清理 finished/dead lease，
复用已有 `close_request`：pending build 被取消/丢弃，已安装且仍有请求对象的
完成历史按原规则保留到 shared。保留活跃容量限制，避免借用 shared 预留槽；
不在逐 token decode 路径增加 CPU 扫描，也不增加 GPU 同步。

新增 10 项回归：max_requests=1/4、global pool 开/关、历史已安装/未安装、
连续六次换入、存活请求保留、容量溢出、槽唯一性、generation 递增及弱引用失效。
修复前 10 项全失败，修复后全通过。

混合 SA/neural decode 的静态检查未发现新的确定错误：最终 gate/候选 TP 共识
在选择 compact tail 前完成；dispatch 排除 padding；每个 tail 大小独立维护
metadata/KV 计划，逐步写地址、hidden state、位置和原行 scatter 均有现有测试。
现有 CPU recurrence 通过不能替代真实 DSV4、CUDA capture 和 TP8 NCCL 验收。
本次不改 mixed-tail 算法，也不以吞吐或输出长度差异证明正确性。

历史参考：`model-pr-history-knowledge/sglang/deepseek-v4/README.en.md`，
包括 #26239 的 MTP Graph/metadata 约束；以当前 `deepseek_v4_backend.py`、
`compact_graph.py`、`device_extension.py` 和 scheduler 调用链为修复依据。

## 本次工作机结果

修复提交 `a3a0d112b6ff02d48a57855afb7d68e3b964bf30` 已由 ai-workstation
从 GitHub clone/pull 后验证。RTX 4060 Laptop 8GB、WSL2/Linux、Python 3.10.18、
PyTorch 2.8.0+cu128、Triton 3.4.0、nvcc 12.0、GCC/G++ 12：
**218 passed、2 skipped，24.24 秒**，其中 CPU/Gloo 181 项、CUDA 37 项。
单卡跳过 NCCL 2/8 卡两项。结束后无测试进程，GPU 显存为 0 MiB。

8 项 compact CUDA 测试使用额外的轻量导入辅助：替换未执行的完整 Graph delegate
标记类，以及测试本来就要替换的合成 attention 容器边界；从该提交直接加载原生
topk Triton kernel 和 ForwardContext。SA CUDA kernel、stream、Graph capture/replay、
gather/scatter 均实际运行。这是合成模型边界的 GPU 契约测试，不是完整服务、真实
DSV4 attention 或 TP8 模型验收；其余 29 项 CUDA 测试不使用此边界替换。
新增 4 项 CUDA lease 用例在 max_requests=1/4、global pool 开/关下，各连续换入
6 批请求并重放查询，逐项检查命中 mask 与完整候选链。

原始日志、环境/源码哈希和辅助文件见 [4060 证据](dsv4-lease-4060/README.md)。
工作机运行目录：`/workspace/dsv4-lease-validation-20261017/`。

用户提供的 macOS Python 崩溃报告对应 21:53 的本地 Gloo 测试子进程。
同次测试日志为 `uv_bind: operation not permitted`，报告栈在
`gloo::transport::uv::Device` 调用 abort/SIGABRT；这是沙箱禁止本地 socket
绑定触发的进程退出，不是这轮 DSV4 CUDA 崩溃。沙箱外本地 Gloo TP8 已通过；
之后本次完整套件移到上述 Linux 工作机，避免重复触发 Mac 弹窗。

## GPU / TP 回归

在完整更新的仓库根目录执行，插件和 SGLang 必须来自同一源码版本：

```bash
export PYTHONPATH="$PWD/experimental/hybrid-spec/src:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest experimental/hybrid-spec/tests -q -rs
# 8 张卡可见时明确执行；单卡跳过不能算通过。
python -m pytest 'experimental/hybrid-spec/tests/test_tp.py::test_tp_consensus_and_graph_staging[nccl-8]' -q -rs
```

RTX 4060 可以验证 SA CUDA kernel、辅助 stream、Graph replay、所有 mixed mask、
padding、gather/scatter 和 generation 隔离；单张 4060 不能验证 TP8 NCCL，
也不能运行 DSV4-Flash TP8 模型。test_cuda 的模型/attention 边界是合成计算，
即使通过也不代表真实 DSV4 attention 已通过。

## 复用上一轮 500 条数据生成同参数对照

以下脚本读取已确认能运行的 plan，沿用模型路径、数据和服务参数。
生成 max_requests=1/c=1 与 max_requests=4/c=4 两份计划；各三轮，轮换
native/full/split 的启动顺序。full/split 的插件参数仅 graph mode 不同。

```bash
export BASE_PLAN=/workspace/validation-runs/dsv4-split-20261017-191157/plan.json
export RUN_DIR=/workspace/validation-runs/dsv4-lease-$(date +%Y%m%d-%H%M%S)
python - <<'PY'
import copy, json, os
from pathlib import Path

source = json.loads(Path(os.environ['BASE_PLAN']).read_text())
out = Path(os.environ['RUN_DIR'])
out.mkdir(parents=True, exist_ok=False)
native = next(s for s in source['servers'] if 'hybrid_config' not in s)
hybrid = next(s for s in source['servers'] if 'hybrid_config' in s)
config = hybrid['hybrid_config']
assert config['num_drafts'] == 8
assert config['num_neural_gate_tokens'] == 2
assert config['partial_drafts'] == 'fallback' and config['enable_global_pool']

def replace_option(command, flag, values):
    while flag in command:
        i = command.index(flag)
        j = i + 1
        while j < len(command) and not command[j].startswith('--'):
            j += 1
        del command[i:j]
    command.extend([flag, *map(str, values)])

for capacity in (1, 4):
    plan = copy.deepcopy(source)
    plan['concurrencies'] = [capacity]
    plan['servers'] = []
    for repeat, order in enumerate((('native','full','split'),
                                    ('full','split','native'),
                                    ('split','native','full')), 1):
        for mode in order:
            server = copy.deepcopy(native if mode == 'native' else hybrid)
            server['name'] = f'{mode}-r{repeat}'
            command = server['command']
            replace_option(command, '--max-running-requests', [capacity])
            replace_option(command, '--cuda-graph-max-bs-decode', [capacity])
            replace_option(command, '--cuda-graph-bs-decode',
                           [1] if capacity == 1 else [1, 2, 4])
            replace_option(command, '--speculative-draft-attention-backend', ['dsv4'])
            if mode != 'native':
                server['hybrid_config'].update(max_requests=capacity, draft_graph_mode=mode)
            plan['servers'].append(server)
    (out / f'plan-c{capacity}.json').write_text(json.dumps(plan, indent=2))

prompts = json.loads(Path(source['workloads'][0]['dataset']).with_name(
    'sa-spec-prompts.json').read_text())
(out / 'smoke-prompts.json').write_text(json.dumps(prompts[:32]))
print(out)
PY
```

先验证不 flush 的 c=1 泄漏场景。终端 A 使用上面生成的 split 配置启动：

```bash
python - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ['RUN_DIR'])
plan = json.loads((out / 'plan-c1.json').read_text())
server = next(s for s in plan['servers'] if s['name'] == 'split-r1')
config = out / 'smoke-config.json'
config.write_text(json.dumps(server['hybrid_config'], indent=2))
os.environ['HYBRID_SPEC_CONFIG'] = str(config)
print('URL:', server['url'], flush=True)
os.execvpe(server['command'][0], server['command'], os.environ)
PY
```

终端 B 设置相同的 RUN_DIR，BASE_URL 使用终端 A 打印的地址，等服务 ready 后执行：

```bash
export BASE_URL=http://127.0.0.1:39921 # 替换为 plan 的实际 URL
for i in $(seq 1 8); do
  curl -fsS "$BASE_URL/generate" -H 'Content-Type: application/json' \
    -d '{"text":"Hello","sampling_params":{"temperature":0,"max_new_tokens":1}}' >/dev/null || break
done
python experimental/hybrid-spec/benchmarks/compare_servers.py \
  --record "$BASE_URL" --prompts "$RUN_DIR/smoke-prompts.json" \
  --concurrency 1 --max-tokens 2048 --output "$RUN_DIR/smoke-c1.json"
```

此 smoke **不要加 `--flush-cache-before-run` 或手动 flush**。比较保存的
`server_info_before/after`：预期 retrieval_requests 有正增量，split skip_ct
有增长。全零说明未通过本轮已知 SA 可命中数据的激活检查，不能继续报告收益。
结束终端 A 服务后再跑正式计划（runner 自行启动、停止每一臂）：

```bash
python experimental/hybrid-spec/benchmarks/run_workload_plan.py \
  --plan "$RUN_DIR/plan-c1.json" --output-dir "$RUN_DIR/c1"
python experimental/hybrid-spec/benchmarks/run_workload_plan.py \
  --plan "$RUN_DIR/plan-c4.json" --output-dir "$RUN_DIR/c4"
```

c=4 至少检查：无请求失败/挂起/CUDA 错误，`mixed_ct > 0`，tail histogram
确实出现 1/2/3 行，`skip_request_ct > 0`。计数应满足
`request_ct = skip_request_ct + tail_request_ct`、
`prefix_ct = skip_ct + tail_ct = sum(tail_batch_histogram.values())`，以及
`tail_request_ct = sum(size * count)`。用每次服务启动后的同一统计窗口比较。
full/split 的接受质量应相近；真实模型逐 token 差异须另设正确性对照，不能由
greedy 长度差、吞吐或 CPU 测试推导等价。
