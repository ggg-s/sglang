# Remote-attn 适配复查

本次复查针对本地 `main@1335d372f0a58857422e40b89a097daa7d3bf86c` 的插件适配，
重点追踪 TP2 命令、源/目标 KV 所有权、逐层 LOCAL 切换和传输/计算之间的依赖。
初轮接口检查和 CPU 测试没有覆盖以下运行时问题；本轮已修复并补充回归。
控制面两个 Python 文件保持原样。

## 发现及修复

| 优先级 | 问题及影响 | 当前处理 |
| --- | --- | --- |
| P1 | TP 广播只保证收到同一命令；某 rank 导入失败而另一 rank 成功，会在后续模型 collective 卡住 | 每条命令结束后在现有 TP CPU group 汇总成功/失败及逻辑调度状态；分歧时所有存活 rank 停止调度并报错 |
| P1 | TP 各 rank 的层到达进度不同，可能分别选择 Graph replay 与 eager，导致 collective 次序不一致 | TP > 1 强制关闭 prefill/decode CUDA Graph；启动检查拒绝开启的配置 |
| P1 | Fast Freeze 元数据缺少分支隔离，读取不存在的 CPU snapshot；普通 Copy 的 chunks 也可能未赋值 | 将 fast/copy 的 wire layout 拆为明确分支 |
| P1 | 源传输线程默认 CUDA device 0，TP rank 1 可能访问错误设备；生产 stream 与传输 stream 没有完整依赖 | 显式设置 rank device，并用 event/stream wait 等待冻结前的 KV 和 indices；目标端也等待 indices 的生产 event |
| P1 | source release/resume 只关监听端口，已连接的传输线程仍可能读取随后被复用的 KV | 设置 cancel、关闭已接受连接、join 线程、同步 GPU reader；不能证明退出时禁止复用 |
| P1 | 目标取消、正常释放、reset 没有一致地等待 receiver；可能在异步 H2D 仍写入时回收槽位 | 先停止并 join receiver，排空 copy stream，再调用上游 free；正常传输完成不再被标成 abort |
| P1 | 接收失败设置所有 event 唤醒等待者，但事件可能被误认为 LOCAL-ready；提前 done 则可能不唤醒等待者 | attention 在等待前后检查 error/abort；提前结束显式报错并唤醒；等待超时抛错；异常状态禁止 Graph replay |
| P1 | source staging 被 TE 持有时可能注销/回收；目标断连也不能证明远端已停止写注册内存 | 正常路径等待 TE completion/H2D，再注销或复用；完成状态不明时保留注册及其 tensor，记录 quarantine |
| P1 | Fast 异步 pipeline 在 target slots < source buffers 时，源端可能先等 ACK，而产生 ACK 所需的 layer control 尚未发送 | 等槽位前先完成并发送相关 layer control；测试覆盖 source buffers=3、target slots=1 |
| P1 | 完整 Copy 在 GPU load 成功后构建请求/采样失败会泄漏 allocation；abort_import 也可能只释放 prefix 而遗留 row/tail | 将整个后续构建阶段纳入回滚，统一按 pending/已绑定 row 的所有权释放 |
| P1 | RPC 收尾的整卡同步会等待无关计算/传输，并把并行流重新串联；同步失败仍释放源 KV 引用 | 每个 RPC worker 使用独立 stream（普通优先级也如此），只 fence 该 stream；失败保留 source lease 并禁止 KV 复用 |
| P1 | scheduler 退出只关闭 Hub，仍可能有 target receiver 或源传输线程访问模型内存 | 退出时关闭 Hub/连接、等待 RPC workers，再停止 target 和 source workers；失败明确上报，并继续尝试其他清理 |
| P2 | 同时多个源传输共用一个 active bool，先完成者会提前清空标志 | 按 transfer token 计数，最后一个完成后才清除 |

TP 一致性比较使用 rid、输出长度、已提交 KV 长度、row 是否存在、冻结状态和队列顺序，
不比较各 rank 本来就可能不同的物理 KV 地址。普通请求和本地 timeout-abort 仍交由原广播逻辑处理。

## 生命周期与重叠的约束

源端：冻结生产 stream → 索引/就绪依赖 → D2H staging → TE completion → layer control。
源 staging 在对应 TE 完成后才可复用；目标 staging 在 H2D 完成并返回 free ACK 后才可复用。
ACK 等待前必须发送旧 layer control，避免形成控制消息与槽位复用的循环依赖。

目标端：分配 KV indices → indices-ready event → 独立 H2D stream → stream 完成 → layer event。
仅在该层 event 已完成且状态无错误时使用本地 prefix，否则通过 Remote attention 读取源端冻结 prefix，
再与目标的本地 tail 合并。layer_wait 模式等待事件；失败唤醒只用于退出，不能作为 LOCAL-ready。

源释放还受 RemoteHub registry 的 inflight RPC 引用约束。GPU reader 无法完成 fence 时保留引用，
阻止 release/resume 复用该 KV。目标释放必须在 receiver 和其 GPU 写入退出以后发生。
这些 stream 依赖使重叠具备实现条件，不能替代 GPU trace 对实际并行度的证明。

## 验证证据

- 28 项 CPU 回归通过；请求 row 与 KV allocator 使用本分支真实实现的 source-loaded 方法和真实 CPU tensor。
- 真实双进程 Gloo TP2：leader-only poll、普通请求/timeout-abort 与 control 混合广播、双方成功、
  单 rank 失败、调度状态分歧、双方一致拒绝。
- CPU tensor + 可控 CUDA/TE/socket 边界：验证 rank 设备、TE 完成前不覆盖 staging、
  3-buffer/1-slot ACK 环、提交失败隔离、RPC stream fence 以及退出清理。
- 导入失败、取消、未 admission、已 admission、正常完成、重复释放和存活 receiver/source reader 的资源边界。
- 当前 16 个 hook 的源接口检查，以及 Python 编译、Ruff F、isort 与格式检查、shell 语法和插件 wheel 构建。

全量仓库 Ruff 规则仍报告实验代码的广泛异常捕获、旧式类型标注等问题；本轮没有做整文件风格重写，
不能将上述限定检查表述为全量 lint 通过。

本机没有完整 SGLang 模型依赖或 CUDA/Mooncake 服务环境；本地 harness 只加载仓库的
`CustomTestCase`，没有执行完整 serving 包的导入或推理。
Gloo 测试验证 CPU 控制协议，不代表 NCCL/TP2 推理或 Mooncake 网络故障已验证。

后续已在 RTX 4060 Laptop 8 GB 上完成单 GPU 验证：FP16/BF16 attention 与
FP32 参考比较、五组真实双进程 Mooncake TCP + CUDA 逐层传输、取消后 GPU 槽位复用。
CUDA trace 记录到独立 stream 上 H2D 与 GEMM 的重叠。详见
[4060 验证记录](validation/4060.md)。这些使用小型 KV storage fixture，仍不覆盖模型服务器、TP2/NCCL 或实际迁移性能。

## 仍需 GPU 环境验收

1. 在相同 TP 宽度的 P/D0/D1 TP2 上运行 Copy、Layer、Remote，逐 token 对比不迁移参考的确定性输出/logprob。
2. 层延迟、提前 EOF、TE 错误、目标重启和单 rank 失败：确认停止/回滚边界，没有错误 KV 的 LOCAL 使用，
   没有 scheduler 进入不匹配 collective。rank 失联仍依赖 process-group timeout 和 supervisor 终止进程。
3. 在 freeze、import、pre-commit abort、resume、正常完成及多次迁移后检查 row/token/registration 计数。
4. 使用 CUDA profiler 检查源 D2H、TE、目标 H2D 与 attention 的重叠，观察各层逐步转为 LOCAL。
5. 确认实验 wheel 的 `send_probe` 和 async completion API 行为；源 wheel 无可追溯构建 commit。

当前异常处理以防止 KV 复用错误为目标。命令一致性检查不是跨 rank 原子事务，不能无损自动恢复
部分执行的命令；registered host buffer 的远端完成状态不明时会保留到进程退出，status 中的
`quarantined_transfers` 可见这部分内存。无法 fence 的 GPU KV 也会保留，需要进程重启回收。
TP Graph-on、异构 TP、跨主机、其他 KV layout 不在本次支持范围内。
