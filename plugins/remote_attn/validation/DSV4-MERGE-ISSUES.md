# DSV4 合并问题核对与验证口径

分支：`codex/remote-attn-pre-layerwise`。核对日期：2026-11-07。

## 修复范围

| 报告项 | 本分支核对 / 处理 |
|---|---|
| ① `blake3` 缺包 | 所有 Python 安装配置已经声明 `blake3>=1.0.0`。部署必须同步安装依赖；仅拉代码或 `pip install --no-deps` 不会修复旧环境。 |
| ② `enable_pdmux` 已移除 | 直接删除三处旧检查（含 hybrid adapter）及兼容函数，不兼容旧版本。`enable_fpm` 是 Forward Pass Metrics，与此无关。 |
| ③ staging 重复登记 | 同一 live room、同一 DecodeRequest 和 receiver 幂等，保留 token、事件、成功状态。不同请求、receiver 替换及 drain 失败后的重试仍拒绝覆盖。 |
| ④ D ready 文件缺失 | D controller 初始化完成后 eager 建 hub；不依赖首次充当远端源。 |
| ⑤ 首请求 aux strides 缺失 | DSV 注册消息始终携带 D 的实际 aux strides；customized-info 默认关闭时，校验显式 stride 的一致性，不要求开启该功能。错误列出具体失败条件。 |
| ⑥ remote 阻止 D2D | 当前只迁移全部 Local 的请求。gateway 显式报告等待原因/次数/时间；plan 重新进行 TP 准入检查，竞态引起的推迟不消耗重试。等待至源结束时明确标记未迁移。部分 Remote 请求的迁移仍未实现。 |
| ⑦ layerwise 控制语义 | layerwise 只支持 native PD/SMG。D2D 要求两个 Decode worker 都使用 remote 且禁用 D radix；不支持的命令通过 `result.<rank>.json` 返回错误。ready 文件只证明控制端已初始化。 |

DSV remote/receive adapter 会安装自己的 staging ring；
`SGLANG_DISAGG_STAGING_BUFFER=0` 不会关闭该插件路径。

部署前使用实际运行 server 的 Python：

```bash
python -m pip install -e python -e plugins/remote_attn
python -m pip check
python -c 'import blake3; import sglang.srt.entrypoints.http_server'
```

## 性能验证模板

每组 native / layerwise / remote 必须记录下列条件，不能用 component probe 的结果
代替 H20 完整模型测量，也不能把修复启动问题解释为性能已经翻正。

1. 固定模型、权重、TP、prompt token IDs、输出长度、采样、P/D radix、图配置及机器。
   记录 commit、完整启动命令和依赖检查结果。
2. 热前缀实验先对**全部待测 prompt**预热并等待请求结束，再测 c=4/8/16；
   每个 serving 变体均独立完成同样预热。冷热实验分开记录；轮换档位顺序并重复，
   不把前一档制造的缓存收益归入后一档并发收益。
3. SSE 请求启用 `stream_options={"include_usage": true}`。生成 token 总数使用最终
   `usage.completion_tokens`，不得使用 `delta.content` 数量或字符数。
   TPOT 按 `(最后输出时间 - 首输出时间) / (completion_tokens - 1)` 计算；
   少于两个 token 或缺少 usage 时记为不可用。usage-only 帧不能作为末 token 时间。
4. 分开报告 TTFT、TPOT、总耗时、throughput 及重复实验分布。token 间隔只能在
   chunk 与 token 一一对应时解释为 token 间隔，否则称为 SSE chunk 间隔。
5. D2D 同时记录准入候选数、实际迁移次数、拒绝原因、freeze/commit/Local 时间、
   `first_target_ms` 和最大输出间隔；与同负载的自然排空比较。0 次迁移不是主动迁移测量。

逐层 RPC 合并是独立的协议与性能优化，未包含在本次正确性修复中。

## 本次验证

- Ubuntu 独立源码快照，RTX 4060 Laptop，Torch 2.13.0+cu130。
- `test_remote_attn.py`：364 项通过（164.9 s）；随后替换的启动/控制用例单独
  复验 2 项通过，覆盖开启 FPM 的初始化、首请求前 ready
  发布和 layerwise 控制命令的显式错误回包。
- `test_remote_main_abi.py`：6 项通过；保留 drain 失败后的拒绝复用，并验证 live
  重复登记不丢 scatter 事件、首个注册消息传递 aux strides。
- `test_remote_request_lists.py`：6 项通过。主套件包含 TP2 D2D/控制一致性用例。
- 删除旧 PDMux 检查后，另行复验 Qwen/DSV 启动 4 项和 hybrid 启动 2 项，全部通过。
- 源码预检：50 个 hooks、131 个 core interfaces 通过；定向 Ruff 和空白检查通过。
- 真实 CUDA/Mooncake TCP：DSV receive 默认独立 stream 和显式 forward stream
  两组通过；每组覆盖有/无 P host staging、字节一致、取消等待 GPU、ring 排空和
  无释放后写入。P4 D2D 探针覆盖预留后冻结、bulk copy 前 Remote 解码、commit
  回包丢失后的恢复、源仅释放一次以及最终 Local 恢复。
- 测试日志和 CUDA JSON 归档：`/private/tmp/remote-attn-merge-issues-results/results.tar.gz`。

边界：未运行 H20 完整权重/跨机 RDMA 或性能复测。隔离 GPU 测试环境同样缺少
`blake3`；网络补包未成功，完整 `http_server` 导入 smoke test 未完成。
依赖声明本身已存在，本次没有以懒加载绕过依赖同步。
