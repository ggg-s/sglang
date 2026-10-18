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
后半段时间包括 compact 打包、target forward、验收/commit、反馈和 native draft-extend；
计时在下一轮 draft 开始时闭合，计入整个 decode 周期。中间插入 prefill/idle 则丢弃该样本。
起草及选择为共同固定成本。
选择 `(请求数 + 预计正确草稿数) / (起草成本 + 验证成本)` 最大的已测档位。
多个逻辑长度落入同一档时按同一物理成本计算，并尽量利用已有容量。

CUDA events 在模型 Graph 外记录，后续轮次只消费已完成事件，不为计时等待 GPU。
未测成本先保留来源上限；默认每 32 轮探索一次，交替采用完整来源长度和轮转的
Graph 容量：完整长度补充尾部标签，轮转容量补充成本样本。冷桶最低长度优先。
计时事件最多保留八组，GPU 落后时可以丢弃成本样本，接受率标签仍照常更新。
成本表不预设 H100/4060 延迟，也不由单卡测量推算 TP8 成本。

默认 token grid 在原固定来源档位上增加各批次 25/50/75% 容量和最短前缀档位；
统一沿用原生 TP/SP 对齐。`verify_token_buckets` 可自定义有限档位，运行时不捕获新图。
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
dynamic_verify_prior_strength: 4.0
dynamic_verify_prior_rate: 0.8
dynamic_verify_min_drafts: 1
```

要求 `verify_mode=compact`、`partial_drafts=fallback` 和现有纯文本 greedy/topk=1
兼容条件。支持已有 MTP/EAGLE3/STANDALONE 插件入口；Arctic 使用官方 suffix 查询，
同样传递匹配长度与 local/shared 来源。GPU SA 在既有查询 kernel 内输出这些元数据，
不新增 SA 查询；此次 CUDA 源码变化会触发扩展的自动重编译。

`hybrid_spec_dynamic_verify` 导出桶内 check/correct 计数、探索次数、成本样本数和
实际档位成本。`hybrid_spec_compact_verify` 继续导出逻辑/物理 token 与图档位直方图。
只在显式读取诊断时回传接受率表。

先比较固定 SA8/neural3/gate1 的 compact 配置与本配置，再补
gate3/neural2、全 SA/全 neural/混合及请求重排；覆盖 c=1/4 和 TP8 的档位对齐。
交替顺序重复至少三次，分别报告冷启动和统计稳定阶段的吞吐、TPOT、接受长度、
实际 target 行数、调度耗时和 collective 开销。动态裁剪不保证在 c=1 获益。
