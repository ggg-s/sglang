# TCP O/LSE 合并回传

128K/C1 的逐层记录显示远程 attention 的层间耗时接近，P 端
`source_return` 是最大的单段。本次把 TCP 回传的 O、LSE 合并成一个
Mooncake transfer region，保留原有完成证明和 slab 生命周期。

## 传输格式

双方在 `te_negotiate` 中同意 `packed_return=true` 且选中
`tcp-host-v1` 时，回传格式为：

```
registered Host slab: [实际 O 字节][实际 LSE 字节][32-byte footer]
```

LSE 紧跟实际 O 长度，不发送 O 区域的预留容量或中间空洞。footer 使用
tag 3，绑定接收方 nonce、帧序号、O+LSE 总长度；一个有序 TCP descriptor
覆盖完整 payload 和 footer。D 必须验证 footer 并完成 H2D，才能确认消费。
O/LSE 继续保持二进制 tensor 格式，包括非有限浮点数。

旧端未声明或未确认能力时，继续使用两个 region；RDMA 也继续使用原来的
两个 GPU region。协商只在连接建立时发生，启用新路径需要更新并重启 P/D。
没有新增环境变量。

这只合并 Mooncake 的传输 region。P 的两个 D2H 拷贝、D 的两个 H2D 拷贝
以及完成 fence 仍然保留。未实现提前返回 RPC 或异步发送；已有 `inline_ack`
用于确认上一帧消费完毕，不代表发送端 DMA 已完成。

## 观测

`source_timing_ms` 和累计 `attention_ms` 保留 `source_return`，新增：

| 指标（ms） | 内容 |
| --- | --- |
| `source_return_prepare` | producer 完成等待、D2H、footer 和传输描述符准备 |
| `source_return_slot_wait` | 等待发送锁 |
| `source_return_transfer` | Mooncake 单次提交及完成等待 |

Plane stats 新增 `transfer_regions`、`packed_returns`。P 上每个成功的
合并回传应分别增加 1、1；不要把 `transfers`（提交次数）当作 region 数量。

`source_compute_submit` 是提交路径的墙钟计时，不是 CUDA-event GPU 计时。
尚未完成的 GPU 工作可能在后续 prepare fence 上被计入 `source_return`，
因此不能将原来的 1.41 ms 全部视为网络传输，也不能直接将 0.81 ms 视为
纯 GPU attention 时间。

## 验证与复测

CPU 合约测试使用真实 PyTorch tensor、RemoteHub TCP 控制连接和显式 fake
NIC；覆盖协商兼容性、单 region 字节数、变长/多行输出、浮点按位保持、
上一帧 footer、延迟到达、失败后保留 DMA backing/请求 lease。

本地轻量入口绕过 SGLang 顶层初始化和 CI test base 导入，执行测试文件中
的真实数据面及断言；它不替代完整 SGLang CI、CUDA 或真实 Mooncake 验证。

复测保持原来的 128K/C1 请求和远程窗口长度，比较 `source_return`、三个新增
分段、ITL 和 E2E，并确认 `packed_returns` 计数增长。收益尚未实测；
若主要耗时仍在 prepare，应继续拆 GPU 等待/D2H；若主要在 transfer，才进一步
评估异步提交与控制消息重叠，不能通过提前 ACK 绕过完成证明。
