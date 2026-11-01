# Qwen3 BCG 输出兼容性：本地验证

日期：2026-11-01；基线 `21b96bf914`。按用户要求仅做本地测试，未上传验证包、未执行4060测试。

## 修改范围

原生 `BreakableCudaGraphBackend` 为 decode dataclass 的行张量增加共享缓冲分配、
有效行切片和图内拷贝，支持 Qwen3 的 `LogitsProcessorOutput`。非 tensor 字段按捕获时的
静态元数据透传；普通 decode 的 sampler 字段在捕获时为 `None`。

每次 replay 返回新的 dataclass 外壳，复用原有 tensor 地址；sampler 写入的 logprobs、
top-logprobs、sampling mask 不会污染捕获模板或后续 batch 的结果。
默认行数推断拒绝混合行数，分配拒绝零维 tensor 字段，避免将不支持的布局静默广播或截断。
这不是对 prefill、beam、自定义动态嵌套输出或不同字段行布局的完整支持。

SMG、路由、PD 传输、overlap、缓存和 TP 协议没有修改。Remote buckets 仍为 **1/2/4**，
没有合入 8/16 扩容。

## 本地验证

- 新增7项 CPU 定向回归：实际 `LogitsProcessorOutput` 类型、logits/hidden states/索引、
  buffer 别名与容量、4→1→2→4→1 切换、sampler 字段隔离、行数推断与 runner override、
  静态元数据、可选 tensor 结构变化、既有 tensor/container 结果及不支持布局拒绝。
- 修复前复现 `Unsupported BCG output type: LogitsProcessorOutput` 和行数推断错误；
  修复后7项全部通过。CPU 环境缺少完整 GPU 依赖，使用源码加载的真实 backend、
  输出 dataclass 和 `CustomTestCase`；CUDA replay/graph-pool 边界使用替身。
- 170项完整插件回归通过，包括59项 DSV4.1 回归和真实 Gloo TP2 检查。
- 源码预检：44 hooks / 108 core interfaces；isort、ruff 和 diff 检查通过。

已为既有 CUDA 小型探针增加 `--logits-output`，从真实输出类定义构造结果，并检查
Remote replay、Local full graph、batch 切换和 sampler 字段。此次未运行该 CUDA 探针；
未加载完整 Qwen3 checkpoint，不宣称新的 GPU 正确性或性能结果。

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
PYTHONPATH=plugins/remote_attn PYTHONDONTWRITEBYTECODE=1 \
python test/manual/remote_attn/segmented_graph_probe.py \
  --source-root . --logits-output
```
