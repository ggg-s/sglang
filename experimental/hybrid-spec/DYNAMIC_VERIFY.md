# 在线校准与动态 compact verify

显式设置 `dynamic_verify=true`，在既有 gate、来源选择和 split 起草之后，
动态决定每条请求的验证前缀长度。默认关闭；不修改 target greedy 验收规则。
示例：[DSV4 MTP 配置](configs/device-dsv4-mtp-dynamic-verify.json)。

仍按 SA 上限启动服务：`--speculative-num-steps 8 --speculative-num-draft-tokens 9`。
配置中的 `num_drafts=8`、`num_neural_drafts=3` 是两种来源的草稿上限。
动态模式只缩短已经生成的草稿的验证长度；neural 起草上限和 gate 不在线改变。
`gate=3/neural=2`、`gate=8/neural=1` 继续合法，gate 检查本身不当作 target 标签。

## 接受率

不需要 SA logits，也不为评分增加 neural forward。两类候选统一使用 target
实际接受结果估计条件正确率，再连乘为前缀存活率。

11 个桶：neural 一个；local SA/suffix 和 shared SA/suffix 各五个匹配长度桶，
边界为 `<=4`、`5..8`、`9..16`、`17..32`、`>32`。桶内再按草稿深度统计。
gate 和后端由 worker 配置固定，统计仅描述当前配置、通过当前 gate 的样本。
neural 第一版使用经验接受率，不读取 logits，也没有逐请求训练的 confidence head。

对桶 b、深度 k：

```
p[b,k] = (correct_ct[b,k] + prior_strength * prior_rate)
         / (check_ct[b,k] + prior_strength)
S[b,k] = product(p[b,0:k+1])
```

`accept_lens` 含 bonus，正确草稿数为 `accept_lens-1`。只更新真正检查且前缀
正确的位置：例如接受三个、第四个拒绝，只给前三个正标签、第四个负标签；
后面不标注。裁剪后全部通过时，截断处不记为拒绝。非法反馈计数不参与学习。
这些是 target 验证标签，不是 EOS 截断后的交付长度，也不是整个补零 dense 窗口。

少于 `dynamic_verify_min_samples`（默认 32 个首位置标签）的桶保留原来源上限。
Beta 平滑默认强度 4、条件正确率 0.8。所有计数在 GPU 上累积，不按请求清零；
模型/配置重启或显式 flush 重置统计。当前为累计统计，不保证快速跟随突发分布变化。

## 预算与成本

候选按存活率在 batch 内排序，保留连续前缀，至少留
`dynamic_verify_min_drafts`（默认 1）个草稿，加每请求一个 anchor。
相同分数按稳定顺序处理，父位置始终先于后续位置。

预算直接取可执行的 target token Graph 档位。每档以
`(请求数, 上下文长度二次幂桶, 实际 Graph token 数)` 记录验证耗时 EMA，
上下文桶最小 512；上下文使用 CPU 请求长度近似，仅用于成本分类。
verify 阶段从进入 worker 验证前开始，到验证结果返回时结束，包含 compact
打包、target forward 和验收/commit。起草/选择以及验证返回后的反馈、native
draft-extend 等作为共享固定成本，在线维护一个 EMA，不按候选预算重复测量。
使用“轮首、verify 开始、verify 结束、下一轮轮首”四个边界，将整个 decode
周期分为 verify 和其余固定成本，不重复或漏算；中间插入 prefill/idle 则丢弃样本。
选择 `(请求数 + 预计正确草稿数) / (共享固定成本 + verify 成本)` 最大的已测档位。
多个逻辑长度落入同一档时按同一物理成本计算，并尽量利用已有容量。

CUDA events 在模型 Graph 外记录，后续轮次只消费已完成事件，不为计时等待 GPU。
默认每 `dynamic_verify_timing_interval=16` 轮采样一次；其他轮不创建计时事件，
只有下一轮轮首需要一个事件闭合上轮的采样。预算探索轮强制采样，避免探索间隔
与采样间隔错开，导致新档位永远没有成本。接受率仍每轮更新。离线校准模式每轮采样。
未测成本先保留来源上限；默认每 32 轮探索一次，交替采用完整来源长度和轮转的
Graph 容量：完整长度补充尾部标签，轮转容量补充成本样本。冷桶最低长度优先。
计时事件最多保留八组，GPU 落后时可以丢弃成本样本，接受率标签仍照常更新。
成本表不预设 H100/4060 延迟，也不由单卡测量推算 TP8 成本。

可通过 `dynamic_verify_cost_table_path` 加载离线校准得到的初始耗时表。
表只包含成本，不包含接受率；加载后冷桶仍按上述规则收集真实 target 标签。
已知档位的第一条在线样本就按 `0.9 * 初始值 + 0.1 * 在线样本` 更新，
此后继续相同 EMA。未测 batch/context/Graph 档位不插值、不外推，沿用完整
来源长度和探索回退。运行时只给当前已捕获的 Graph 档位定价；逻辑 token 数
先加每请求一个 anchor，再向上取整到实际执行的物理档位。

表记录 hybrid 配置以及 target/draft 模型路径、GPU 型号、TP 大小、模型/KV dtype
和 target attention backend、实际 capture 上限及 token grid；不匹配会报错，
避免跨部署或捕获形状误用成本。
采样遇到 eager fallback 会保留接受率标签，但丢弃该轮成本，防止污染同 token
数的 Graph 档位。没有 target Graph runner 时保留完整来源长度。

### 离线校准

在与正式部署相同的模型、GPU、TP、backend 和 capture 配置上启动校准服务。
复制正式 hybrid 配置，设置 `dynamic_verify_profile=true`，移除
`dynamic_verify_cost_table_path`，启动进程前设置 `SGLANG_SIMULATE_ACC_LEN=1`。
这是专用校准服务：模拟每轮只接受一个 bonus token，让 KV 长度按固定步长增长；
校准结束后停掉该服务，正式服务关闭 profile 模式并移除模拟接受环境变量。

离线采样流程对齐 [DSpark SPS profiler](../../python/sglang/benchmark/dspark_sps_profiler.py)：
固定 batch 和预算、排除预热、等待稳定窗口、多次重复取中位数。本实现保留
CUDA Event 分段计时及上下文分桶；DSpark SPS 默认读 CPU 解码节拍，两者计时
口径不同，因此不直接复用其 SPS JSON 或二维加性拟合。

使用代表真实上下文长度的 prompts（JSON 字符串数组）运行：

```bash
python experimental/hybrid-spec/benchmarks/profile_dynamic_verify.py \
  --url http://127.0.0.1:30000 \
  --config /tmp/hybrid-calibration.json \
  --prompts experimental/hybrid-spec/benchmarks/prompts.json \
  --batch-size 1 4 --fracs 0.25 0.5 0.75 1 \
  --repeats 3 --max-tokens 4096 \
  --output /tmp/hybrid-verify-costs.json
```

工具将各 fraction 对应的逻辑 token 数向上取整为实际 Graph 档位，并去重，
依次固定 `(batch size, physical graph tokens)`，不再由动态策略自由探索。
只有校准模式才会通过 `/set_internal_state` 暴露
`hybrid_spec_profile_batch_size` / `hybrid_spec_profile_graph_tokens` 两个数字命令。
短暂 ramp-in 的其他 batch 不应用该 pin，也不会混入采样。

每个采样点默认丢弃前 8 个对齐步骤，要求至少 32 个稳态样本且稳定运行 10 秒；
各上下文桶也需至少 32 个样本。多 rank 按同一 step ID 对齐，对同一步的各分量
取最大值，再对稳态步骤取中位数；重复扫描后对同一采样点再取中位数。
原始记录写入 `.records.jsonl`，每轮统计写入 `.rounds.jsonl`，实验参数与 capture
配置写入 `.manifest.json`，最终 JSON 只保存成本，不保存模拟接受标签或在线 EMA。

每个采样点先 flush 再 pin，结束后中止并排空剩余校准请求，最后恢复 pin=0；
因此必须使用独立校准服务。请求用 `ignore_eos=true`；`--max-tokens` 仍受模型
上下文/KV 容量上限约束。如果请求在满足稳态门限前结束，会报错而不是导出尾段
数据，可调整 token 数、上下文或 `--min-steady-seconds`。

正式配置使用 `dynamic_verify_profile=false`、探索间隔 32、耗时采样间隔 16，添加：

```json
"dynamic_verify_cost_table_path": "/tmp/hybrid-verify-costs.json"
```

配置读取路径在服务端；重启后载入成本，接受率重新在线累计。
表格式 v2 用 `verify_ms` 记录物理 token 档位的 verify 阶段耗时，用 `fixed_ms`
记录每个 batch/context 的共享固定成本。之前 v1 的整轮尾段计时口径不同，需重新
校准，不能直接作为 verify 阶段耗时加载。
这只是成本初始化，不能代替正式 workload 的吞吐/TPOT 对照验证。

默认 token grid 保留原生档位和每个允许的 batch 的完整长度安全档位；batch<=8
保留来源混合的精确档位。动态附加档位只在 batch 锚点 `1,2,4,8,...,并发上限`
增加最短前缀、neural、25/50/75% 容量，附加数量按并发上限的对数增长；不枚举
每条请求的长度组合。统一沿用原生 TP/SP 对齐、去重，图只在启动时捕获。
完整长度安全档位仍随并发上限线性增长，部署应按实际需要设置 `max_running_requests`。
`verify_token_buckets` 可显式指定 token 容量，仍会补齐完整长度安全档位。
每轮只选择已捕获的图，先按实际源长度上限限制逻辑预算，再向上取整定价；同一个
图内尽量用满有价值的候选，不把减少逻辑 token 当成等量的物理耗时收益。
选择器在 Graph 外执行；target 继续使用已捕获的变长 Graph 和固定地址 metadata。
候选位置数 `batch * num_drafts <= 256`、候选图档位数不超过 256 时，
评分/排序/预算选择使用单个 CUDA block，反馈计数和来源分桶也使用合并算子。
更大形状保留 Torch 路径，没有新增 batch 上限。CPU Torch 路径用于独立语义对照。
动态 Arctic 路径也编译这些 CUDA 算子，但不分配 GPU SA 索引。

TP 额外对桶、最终长度、接受计数分别做 MIN 共识，每轮三次 collective；
所有 rank 调用顺序相同。各 rank 计时可能不同，长度取共同的较短前缀，
随后按相同的逻辑总数选 target 图。TP1 不增加通信。
动态长度仍需读回一个总 token 数；不能称为零同步或零调度开销。

## 配置与诊断

```
dynamic_verify: true
dynamic_verify_min_samples: 32
dynamic_verify_explore_interval: 32
dynamic_verify_timing_interval: 16
dynamic_verify_profile: false
dynamic_verify_prior_strength: 4.0
dynamic_verify_prior_rate: 0.8
dynamic_verify_min_drafts: 1
dynamic_verify_cost_table_path: null
```

要求 `verify_mode=compact`、`partial_drafts=fallback` 和现有纯文本 greedy/topk=1
兼容条件。支持已有 MTP/EAGLE3/STANDALONE 插件入口；Arctic 使用官方 suffix 查询，
同样传递匹配长度与 local/shared 来源。GPU SA 在既有查询 kernel 内输出这些元数据，
不新增 SA 查询；此次 CUDA 源码变化会触发扩展的自动重编译。

`hybrid_spec_dynamic_verify` 导出桶内 check/correct 计数、探索次数、成本样本数和
实际档位成本；`seed_cost_entry_ct` 是加载的成本项数，`cost_sample_ct` 只统计在线
样本，`cost_profile` 是可保存的成本表。`hybrid_spec_compact_verify` 继续导出
逻辑/物理 token 与图档位直方图。
只在显式读取诊断时回传接受率表。

先比较固定 SA8/neural3/gate1 的 compact 配置与本配置，再补
gate3/neural2、全 SA/全 neural/混合及请求重排；覆盖 c=1/4 和 TP8 的档位对齐。
交替顺序重复至少三次，分别报告冷启动和统计稳定阶段的吞吐、TPOT、接受长度、
实际 target 行数、调度耗时和 collective 开销。动态裁剪不保证在 c=1 获益。
