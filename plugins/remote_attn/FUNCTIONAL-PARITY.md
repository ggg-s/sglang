# 与 handoff 原包的功能对应

对照对象是本次实际收到的 `handoff/handoff_control.py`，MD5 为
`ad5290feffe0109dc8bbd18c94a4bafb`。它与原包文档声明的
`e545304a50a64f64e1f1cf9e679e634f` 不同；没有将未收到的那个文件视为已验证输入。
原包参考 SGLang base 为 `3a869635c`；本分支以 `main@1335d372f0` 为基础。

AST 清单核对：原文件的 **124 个顶层函数、15 个类及其全部方法、40 个 `cmd` 分支**均保留，
runtime 环境变量开关及默认值保留；部署批次默认值因内网回归从 8 修正为 2048，见 [5090 内网反馈](validation/5090-intranet.md)。这是代码覆盖核对，不代表每个命令都做过端到端测试。

| 原包功能 | 当前适配 |
| --- | --- |
| P-only + D0/D1，Mooncake TCP | 保留 native PD 接口与三个功能启动配置；端口、GPU、TP2 和 env 等价保留 |
| D2D freeze/import/commit/release，Copy/Layer/Remote | 完整 runtime 保留；Req KV 字段、token mapping、allocation 和 batch 构建适配当前 main |
| Fast Freeze / Fast Resume | 保留；修复 fast/copy 元数据分支、row 所有权、跨 stream 依赖及失败回滚 |
| TP2 all-rank command | rank 0 poll 后注入原有请求广播；每个 rank 执行，再检查命令结果与逻辑调度状态一致性 |
| RemoteHub / progressive LOCAL | 保留 source registry、远程 prefix、本地 tail 和逐层 LOCAL；错误事件不能当作数据就绪 |
| 传输与计算重叠 | 独立 D2H/H2D/RPC streams；TE 完成后复用源 buffer，H2D 完成并 ACK 后复用目标 slot |
| watchdog / claim / epoch / fence / reservation / lease | 保留；取消/释放前补齐后台 GPU 读写 fence，不确定的 host transfer 保留到进程退出 |
| Mooncake send lock / pre-send probe | 由插件 hook 接入当前 native PD connector；`send_probe` 缺失则拒绝启动 |
| P5.10 prefill/decode interval | 使用 main 已有实现，无重复 scheduler loop |
| 输出 owner | handoff 请求由控制面读取，过滤常规 output streamer，避免重复输出 |
| 控制面 | `p2d_driver.py`、`p4_gateway.py` 与实际原包逐字节相同，SHA256 已核对 |

为保留原始输入的字节内容，提交格式工具只对 `control_plane/` 和 `migration/` 的格式/拼写处理
设置了路径例外；Python AST、密钥检测等检查仍运行。runtime 和新增测试按仓库格式检查。

## 范围与验收状态

当前实现覆盖原包 **P0–P5.18 功能启动配置**：同机、相同 TP 宽度、dense NHD MHA FP16/BF16、
page size 1、禁用 radix cache、scheduler overlap 和 CUDA Graph。TP2 Graph-on 会被显式拒绝。
原包没有完成的 P5.19、生产 router、SSE 重连续传、自动扩缩容，以及可选观测日志，不计为已适配功能。

已通过 31 项 CPU 契约回归（含真实 Gloo TP2 控制广播）和 4060 上的 10 项 CUDA/Mooncake/Remote RPC 检查；
详见 [REVIEW.md](REVIEW.md)、[4060 验证](validation/4060.md)。
仍需双 GPU TP2/NCCL、完整模型服务器迁移、session 重启与 controller 故障的端到端验收。
因此结论是：**所收到原包的功能代码和必要 hook 已移植；完整部署验收尚未完成。**
