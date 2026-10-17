# 跨请求 SA 全局池

日期：2026-10-17。开发基线：`2d53a3d2d2`，分支 `codex/hybrid-suffix-speculation`。

## 参考源码

本地 Baseten `sa_spec` 固定提交 `1cc7e8c86b1914a9c05eb60fb11ebe52daf47969` 的
`src/api.cc` 和 `src/api.cu` 仍按 request ID 分配独立 SA；全局分配器不是跨请求搜索。

本轮重点读取 TensorRT-LLM 官方 main（核查时为
`7ebcf4debcac29b3f6225cb4affff3ecf05514aa`）的：

- [`suffix_automaton.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/7ebcf4debcac29b3f6225cb4affff3ecf05514aa/tensorrt_llm/_torch/speculative/suffix_automaton.py)：`SAConfig.enable_global_pool`、池大小与活跃/保留槽位、`_allocate_slot` FIFO 淘汰、`remove_request` 跳过未完成导入的槽位、`extend_global`。
- [`suffixAutomatonKernels.cu`](https://github.com/NVIDIA/TensorRT-LLM/blob/7ebcf4debcac29b3f6225cb4affff3ecf05514aa/cpp/tensorrt_llm/kernels/speculativeDecoding/suffixAutomaton/suffixAutomatonKernels.cu)：先更新全部 SA，再只读跨槽位查询；匹配长度、本请求优先、续写长度的排序。
- [`sa_enhancer.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/7ebcf4debcac29b3f6225cb4affff3ecf05514aa/tensorrt_llm/_torch/speculative/sa_enhancer.py)：当前类名为 `SADraftEnhancer`，独立 stream 更新/查询，最后统一覆盖 neural draft。
- [`suffixAutomaton.h`](https://github.com/NVIDIA/TensorRT-LLM/blob/7ebcf4debcac29b3f6225cb4affff3ecf05514aa/cpp/tensorrt_llm/kernels/speculativeDecoding/suffixAutomaton/suffixAutomaton.h)：`lookupWithSuffix` 的外部后缀遍历，以及匹配落在来源末尾时沿 suffix link 回退到可续写的更短匹配。

读取文件的 SHA-256 分别为
`109d0b38cbc43a30555b8e6aa0a7d788ae6ac44111bfbfc3b1d563fc53a125c0`、
`6e5c79686d357f3aac2faad3c4f5a5801e7833bf7eb21a059520c8cd0a504abd`、
`45941e22f67b0304280970a0657f07de7f55752d1b47d767f28ba767face1848`。
末尾回退参考头文件的 SHA-256 为
`3b5e12900a38b2e973ffea127e679168ad5a196ff763bca69a1b9e2dcfc1ecb0`。
实现独立编写，没有复制或运行时依赖 TensorRT-LLM 源码。

已有 Qwen3.5 PR 历史记录和本分支 `OPTIMIZATIONS.md` 一并复核。历史 PR 仅用于提示
native verify、MTP 递推和 split 的验证边界；本轮不修改模型 forward、位置或 KV 提交。

## 本轮实现和差异

- 开关默认关闭，仅支持 device 历史。`max_requests` 限制活跃 SA；额外
  `max_shared_requests` 限制完成历史。后者固定预留并在完成时 FIFO 淘汰，
  与 TRT 在总池分配压力下才淘汰的策略不同。
- 搜索已导入的活跃槽位和保留槽位。先消费所有发布的反馈，随后只读搜索；
  无跨 block 的边更新边读，不在 scheduler 回调里启动 CUDA。
- 每个来源保持独立 SA，无跨请求拼接。跨来源上下文上限独立配置；本请求原查询不截断。
  最长匹配若没有后续 token，沿 suffix link 查找可续写的更短后缀。
  先按现有 `partial_drafts` 资格过滤，再比较匹配长度；相同长度优先本请求，
  外部来源优先更长续写，最后按较新 generation 决定。这不宣称逐项复制 TRT 的选择策略。
- 完成历史携带 CPU 确认的最终长度。最后增量可截断后追加；已经索引的丢弃尾部不能原地
  回滚，因此放弃该来源。超容量、无版本匹配、旧 generation 均回退。
- Graph 使用固定地址的槽位、generation、最终长度输入，来源变化也会使 staging key
  失效。flush 清除所有来源，ID 重用不继承旧所有权。
- 继续使用本分支的 forward stream、逐位置候选选择、neural gate、TP 共识和原生
  target verify。没有引入 TRT 的 side stream 或循环结束统一覆盖行为。
- 跨槽位查询参考 TRT 改为每请求一个 block、线程并行扫描来源、共享内存归约。
  block 为 32/64/128 线程，超过 128 个来源由线程分步扫描；比较键为匹配长度、
  未截断续写长度、发布索引，保证跨线程同分结果与原串行扫描一致。
  相同匹配仍优先本请求，完整度资格先于排序判断。
  更新/查询仍在原 forward stream 按 kernel 边界排序，可用于 full 和 split。

池的作用域是同一 worker 的所有 opt-in 请求，包含 prompt 和 target-confirmed 输出。
无租户隔离或持久化。启用方必须允许这些请求之间共享内容。

## 验证

后续并行扫描版本已在 RTX 6000D 完整回归：182 passed、1 skipped（76.60 s），
其中 26 个 CUDA 用例全部通过。新增来源并行归约的独立文本 oracle、20/64/257 来源、
跨线程与跨步同分排序、两种 partial 策略、Graph 重放测试。
串行/并行微基准的 12 组输入、候选 token 和 mask 完全一致。
full 与 split 的同日吞吐对照及完整证据见
[并行扫描专项记录](sa-parallel-6000d-throughput/report.md)。

以下保留并行化之前的全局池实现验收和开/关池对照。

测试覆盖外部后缀与独立穷举 oracle 的差分、完成/未导入/重复关闭、FIFO、
稳定活跃 batch 下的 Graph 来源更新、flush、padding、旧 generation、最后 verify
反馈的幂等消费、跨文档边界、活跃请求互查和停止尾部截断。

本地 CPU、原生 C++、真实 draft 递推函数体：153 passed。
A100 PCIe 40GB，Torch 2.14.1+cu130，最终完整包回归：175 passed、1 skipped（103.85 s，包含重编译）；
唯一跳过项为需要双卡的 NCCL，用 Gloo 的 1/2/4 rank 共识测试通过。
19 个 CUDA 用例全部通过，包含已有 full/split/compact 回归和 4 个新全局池用例。
补充外部来源的续写长度排序断言后，两个相关 Graph 用例再次通过。
最终全量回归包含来源末尾的 suffix-link 回退断言。命令和源码哈希保存在
[`global-pool-a100.json`](global-pool-a100.json)。

### 真实服务短对照

沿用 A100 上已验证的 `Qwen3.5-35B-A3B-GPTQ-Int4`、FP16/gptq_marlin、
unquantized MTP、TP=1、K=8、Triton draft/GDN 后端。native 源码为 `972aa16098`，
与本轮 `2d53a3d2d2` 基线仅差文档。本轮插件通过独立 PYTHONPATH 加载，
使用临时端口 39981 和隔离的 CUDA 编译缓存。28 个生产源码文件 SHA-256
与本地工作区逐项一致；重新加载包含最终 suffix-link 回退的新 CUDA 库后再测。

配置为 full、greedy、neural gate=0、2 个活跃槽位 + 2 个保留槽位、H=4096，
跨请求上下文上限 128。先完成一个最长 256 token 的代码生成请求，
再对相同 prompt 请求 64 token；随后 flush，再用相同 prompt 请求 64 token。

| 观测 | 保留首请求后的第二次请求 | flush 后的新请求 |
| --- | ---: | ---: |
| 检索来源的 decode 请求行数增量（并发 1） | 7 | 1 |
| 本样本返回文本 | 与右侧一致 | 与左侧一致 |

三次请求均成功。该差异支持保留历史在实际 worker/完成回调/Graph replay 中生效，
而不只是构造人工 CUDA 输入得到候选。flush 后仍有 1 次命中来自新请求自身的重复，
不应将其误报为池未清空。原始响应、配置和累计计数见
[`global-pool-a100-smoke.json`](global-pool-a100-smoke.json)。临时服务完成后停止，
A100 显存回到 0 MiB。这里只有一组短输入对照，不证明一般 token 等价、任务准确率
或吞吐收益；没有运行新的 GSM8K 或正式性能基准。

### RTX 6000D 复验

2026-10-17，在 `hz01-ssh.gpuhome.cc:30167` 的单卡 RTX 6000D（SM120，85,651 MiB）
继续验证。Torch 2.14.1+cu130、CUDA Toolkit 13.0；当前工作区源码上传到独立目录，
5,616 个 Python/runtime 与插件源码文件逐项 SHA-256 一致。原服务器 checkout
`53e1300` 保持干净，服务通过 PYTHONPATH 使用本轮 `2d53a3d2d2` 基线及未提交改动。
这次短功能复验选择了 Python TreeCore fallback，没有据此测量性能。

完整包回归为 **175 passed、1 skipped，76.26 s**，包含首次 SM120 CUDA 编译。
19 个 CUDA 用例全部通过；唯一跳过项仍为双卡 NCCL，Gloo 1/2/4 rank 通过。

实际模型使用本机已有的 `Qwen3.6-35B-A3B-FP8`，TP=1、K=8、两个活跃槽位、
两个完成历史槽位、H=4096、上下文上限 128。沿用上述短代码 prompt，每个配置独立
启动服务，先完成最长 256 token 的请求，再请求 64 token，flush 后再请求 64 token。
除 full/global-pool 的最初三请求复验外，其余配置再并发发送两个最长 128 token 的请求。
共 18 次生成全部成功。

| 模式 | neural gate | 保留历史后命中 | flush 后命中 | 并发 2 |
| --- | ---: | ---: | ---: | --- |
| full，全局池开启 | 0 | 7 | 0 | 初次短复验未运行 |
| full，全局池关闭 | 0 | 0 | 0 | 通过 |
| split_no_skip，全局池开启 | 2 | 7 | 0 | 通过 |
| split，全局池开启 | 2 | 7 | 0 | 通过 |

各配置三个串行请求的完整 output token IDs 与关闭全局池的对照一致；两个 split
配置的并发请求也逐项与该对照一致。这里只覆盖这一个短输入，不能推导一般 token
等价或任务准确率。

split 在第二个请求结束时已有 7 次 tail skip；全部请求结束后累计 14 个 batch、
21 个请求行跳过尾部，而 split_no_skip 的 skip 为 0。并发 split 本样本的 mixed_ct=0，
真实服务没有覆盖“一行命中、一行回退”的混合路径，该路径仍以组件测试为证据。
full 模式继续执行 MTP；这里的 skip 证据仅属于显式启用的 split 模式。

最初补测脚本误设 split gate=4，被已有配置校验拒绝；修正为要求的 gate=2 后完成上述
两轮验证，没有修改生产实现。原始配置、启动脚本、token 响应、哈希和计数索引见
[`global-pool-6000d.json`](global-pool-6000d.json)。远端完整日志、脚本和源码归档保存在
`/root/rivermind-data/validation/global-pool-20261017/`。临时服务已停止，显存回到 0 MiB。
上述短功能复验没有吞吐基准、GSM8K 或真实多卡 NCCL 结果。

### RTX 6000D 吞吐对照

后续在相同 GPU 和模型上使用官方 `sglang.benchmark.serving` 流式客户端，对比
full/关池、full/开池、split/开池，并补充 split/关池以分离两项优化的收益。
统一 K=8、gate=2、4 个活跃槽位、16 个完成历史槽位（开池时）、H=4096；
所有配置关闭 KV 前缀缓存。每个测量点为 80 请求、固定生成 128 token，
并发 1/4，各重复三轮。使用真实代码编辑数据，另将其中 16 个输入各重复五次，
分为空池起步和通过真实请求预先建池两种情况，未导入参考答案。

full 模式开启全局池后，不同代码请求的吞吐中位数降低约 7%–9%；重复请求冷池
基本持平，暖池有小幅且波动的变化。split 的改善依赖负载，不能把其全部收益
归于跨请求共享。补充 split/关池控制后，跨请求共享本身在重复请求冷池的吞吐
中位增益为 3.7%/2.6%，暖池为 3.7%/6.4%（并发 1/4），而不同代码请求为
-8.7%/-6.1%。这些百分比为相同轮次比值的中位数；各组绝对吞吐中位数的比值
可能略有不同。

共 72 个有效测量点、5,760 个请求、737,280 个输出 token 全部完成；split/开池
覆盖了 2,493 个真实 mixed batch。四组控制、轮间范围、TTFT/TPOT、候选接受长度和
分支计数见[吞吐报告](global-pool-6000d-throughput/report.md)。临时服务均已停止，
最终显存为 0 MiB，原服务器 checkout 保持干净。补充控制曾因容器进程重置中断，
中断前 11 个点另行保留并排除，随后完整三轮重跑；原三组 54 个点已在中断前完成。

较大样本中，不同配置相对 full/关池的 4,320 对输出有 636 对文本不同，与上面的
单个短输入对照不同。这包括 split/关池的差异，不能全部归因于共享池。这些测量
只说明固定输入和输出预算下的吞吐表现，不能作为一般输出等价或任务准确率证明；
差异索引和样本保存在报告目录的 `output-parity.json`。本轮没有修改生产实现。

模型输出一致性、真实多卡 NCCL 和端到端性能与组件测试分开验收，
不沿用旧 C1/C4 性能数字作为本轮证据。

本轮增加 CUDA 算子参数和 kernel，布局仍为 v3；更新源码后必须重启服务并完成
JIT 重新编译，不能沿用上一轮纯 Python 改动“不需重编译”的说明。
