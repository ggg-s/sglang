# GPU SA 续写频次与默认 split / gate=1

自 2026-10-21 起，新配置默认使用 `history_mode=device`、
`draft_graph_mode=split`、`num_neural_gate_tokens=1`，GPU SA 的
`sa_continuation_policy` 默认是 `frequency`。普通模型 JSON 模板同步更新。
DSPARK split 仍仅支持 gate=0，因此其两个显式模板保留原来的 full/gate=1、
split/gate=0；`device-sa-spec.json` 显式保留 first/full/gate=0，作为历史参考策略。

## 候选规则

先按原有规则找最长重复后缀。在这个状态的每一条 token 转移中，比较
追加该 token 后的子串在当前历史中的出现次数；每步选择次数最多的转移，
同次数选择 token ID 较小者，直到生成 K 个候选或没有下一条转移。
重叠出现也计数；末尾尚无续写的出现不贡献下一 token 的频次。

例如上下文 AB 的最早一次续写为 XXX，后续两次为 YYY，frequency 会选 YYY，
而 `sa_continuation_policy=first` 会复制 XXX。频次不是 target 的接受概率，
也没有替代 neural gate、TP 共识或 target verify。它是逐步贪心选择，不搜索
全局最优 K-token 链。若最高频路径不足 K，默认整行回退 neural；不搜索次优
的较长路径，不缩短匹配上下文来凑长度。`partial_drafts=zero` 仍可显式补零。

全局池保留逐 donor 的索引与原有来源选择：最长有效匹配优先、同长本地优先。
frequency 模式下 donor 之间按可生成的候选数（最多 K）再按发布顺序打破平局。
频次只在每个 donor 的历史内统计，不将不同 donor 的次数累加。命中的 donor
使用自己的频次产生候选。`sa_require_unique_continuation` 仍然检查观察到的分支
歧义，不因为某条分支频率高就允许存在冲突。

## 增量计数

每个 SA 状态维护其 end-position 出现次数。为避免每追加一个 token 都遍历
所有后缀，用 suffix-link 树上的 link-cut forest 做根到 last 的路径加一。
clone 创建时先继承原状态的出现次数，再调整计数树父子关系，最后登记当前新
位置。因此 prompt、已接受草稿和 bonus 都纳入统计；拒绝的尾部不进入历史。

初始 prompt 在 CPU 上使用同一份 C++ 实现构建，再异步导入 GPU；后续反馈
在 GPU 上更新。没有定期全历史重扫、频次 D2H 或为计数增加 neural forward。
动态树更新是摊还对数操作，但不是零开销。查询只读节点值与祖先的延迟增量，
不做旋转或传播：多个请求/线程读取同一个 donor 时不会改写共享状态。
只读访问的成本依赖辅助树深度，不保证每次查询都是对数时间；高分叉状态还需
比较各条出边。真实模型收益必须单独测量。

原生反馈的代际/版本检查、重复消费保护、padding、完成后停止裁剪和槽位重用
规则继续生效。更新在查询前完成；共享池先由独立 kernel 消费所有已发布 donor
的反馈，再允许并发读取。

## 内存与升级

SA 仍有 10 个逻辑字段：原来的 5 个状态字段，以及精确频次辅助树的
left/right/parent、lazy、count。布局按配置的 `max_history_tokens=H` 在启动时选择：

| 历史容量 | 状态物理布局 | 索引预算/token/槽位 |
|---|---|---|
| H ≤ 32767 | 9 个 uint16 + 1 个 int32，共 22 字节/状态 | **约 72 字节** |
| H ≥ 32768 | 10 个 int32，共 40 字节/状态 | **约 108 字节** |

短布局的九个字段以 `值+1` 编码，0 表示 -1 哨兵。最大的预留状态编号
`2H ≤ 65534`，加一仍放得进 uint16；长度、位置、次数和延迟增量也有 H 的上界。
secondary 转移字段保留完整有符号 32 位，按两个 uint16 半字存取，避免 22 字节
状态步长造成的未对齐 int32 访问。token 历史与边标签仍是 int32，词表不受
65535 的限制。没有截断、饱和或近似计数，也没有改变频次候选策略。

每槽的精确预算为 `4*(H + 6*(H+1) + 7) + S*(2H+1)`，S 为 22 或 40；
再加设备反馈 `40 + 4*(K+1)` 字节。32 槽 × 16K 历史的索引约从 54 降到
36 MiB，Graph 私有池、查询缓冲和 CUDA 分配器对齐另计。

布局由槽位的最大容量决定，不在 decode 中临时扩容或换布局。长容量槽位即使
只有一个短 prompt，CPU builder 仍直接构建 32 位状态；导入只复制已使用的
同布局前缀，没有额外的 GPU 解压、转换 kernel 或 D2H。CPU builder 保留显式
wide ABI，并增加 compact ABI；导入会拒绝不匹配的布局。

`first` 对照也维护精确计数，只隔离候选策略，不能作为旧版无计数性能基线。
扩展缓存名升级为 `hybrid_spec_gpu_sa_v6`。CPU builder、Python 分配器和 CUDA
源码必须一起更新并重启，新名称触发首次 JIT 编译；旧 v5 状态不能直接导入。
CUDA 根据状态 dtype 分派两份编译好的 kernel；Graph 捕获后地址和布局固定。
动态成本表现在带 `gpu_sa_storage_revision=6` 指纹，拒绝旧版成本表，需要重新校准。

## 验证

`tests/native/frequency_test.cpp` 在每次追加后，以独立的历史子串扫描核对所有
状态的次数和贪心候选，覆盖二值穷举、随机 clone、重复文本、只读查询和
host 小容量构建后导入大容量继续更新。由 `test_native.py` 编译执行。

`tests/test_frequency.py` 核对默认配置、所有常规模板和 first/frequency 的
可观察差异；CUDA 参数用例覆盖 eager/Graph、本地/共享 donor、接受反馈、
拒绝尾部、padding 与重复 replay。CUDA 跳过不能计为 GPU 验证通过。

频次 v5 的历史记录见 [频次验收](validation/sa-frequency-20261021/README.md)；
自适应布局 v6 的结果见 [72 字节布局验收](validation/sa-compact-20261021/README.md)。
更新/查询分别计时与外围计时异常调查见
[4060 分阶段报告](validation/sa-phase-20261021/README.md)。
没有测量真实模型性能。

```bash
python -m pytest experimental/hybrid-spec/tests/test_native.py \
  experimental/hybrid-spec/tests/test_frequency.py -q
```

这次实现不扩展树状验证、DSPARK gate≥1 split 或统计跨 donor 合并。
