# D2D 二维搬运与原生预留：4060 验证

日期：2026-11-06。分支 `codex/remote-attn-pre-layerwise`，基线
`5884814306731be4485f3fa7ef3666e8fbe1979e` 加本次未提交修改。
配置、控制协议和资源映射见 [统一二维搬运](../D2D-LAYER-TOKEN-RANGE-ZH.md)。
完整原始结果和 97 个源码 SHA256 见 [JSON](d2d-layer-token-range-4060.json)，
4060 快照逐文件核对后与本地交付源码一致。

## 环境与边界

用户 RTX4060 Laptop 8GB；PyTorch 2.13.0+cu130、真实 CUDA stream/event、pinned
Host memory、Mooncake TCP、Qwen FlashInfer 和 DSV sparse kernel。隔离目录为
`/tmp/d2d-ranges-4060.KG0ReD`。CPU 回归使用独立 PyTorch 2.15.0 环境。

原生预留 probe source-load 真实 `ReqKvInfo`、`ReqToTokenPool`、PD `_pre_alloc`、
分页预算、paged allocator 和 release 代码；page 4/16 执行真实 CUDA Triton
`alloc_extend` kernel。模型、scheduler/admission/session 边界是小池夹具，没有
checkpoint 权重，不等同于完整 Qwen/DSV、生产 SMG 或 H20 TP4/RDMA 验收。

## 结果

361 项 CPU 合同全部通过（127.849 秒），包含真实 Qwen native TP2 Gloo：一侧
冻结长度越界时双方都在冻结前拒绝；一侧预留记录丢失时双方拒绝补分配，保留原行。
原有 DSV TP2 共识、P4/SSE/恢复与回收合同同时通过。Ruff 和 diff whitespace 检查通过。

28 个 CUDA 用例全部通过：

| 范围 | 数量 | 检查 |
| --- | ---: | --- |
| Qwen 二维传输 | 8 | 默认组宽 1、组宽 2/3/4；range 257/511/513/1024；FP16/BF16、sync/async、CPU snapshot、relay slot 限制、RemoteDrop、取消和状态证明丢失 |
| Qwen 原生预留 | 3 | page 1/4/16，真实预分配一次、原请求行复用、完整尾页修剪、半页保留和一次性 free |
| DSV 多 range / P4 | 8 | C2 V4/V4.1、C4、C128，Q/O socket/GPU 两种模式；丢失 commit 回复后重启恢复、native state、尾页/padding |
| DSV 同组独立 Local | 2 | V4/V4.1 在最后一个 entry 的最终 range 阻塞时，前一个 entry 可 Local，源 KV 仍保留 |
| DSV bulk 故障 | 6 | 断连取消、D2H 输入证明丢失各重复 3 次，检查真实注册/写入与隔离 |
| DSV 默认配置 | 1 | group=1、range=1024 的 V4.1 C2 兼容性 |

Qwen 的五种正常矩阵记录实际 native batch 数为 `12 / 32 / 24 / 15 / 10`，
与固定逻辑层组内拆分的 token ranges、relay slot 数一致；正常路径 GPU KV
逐字节一致，所有正常注册回收。RemoteDrop 的一层保持 Remote，其余五层 Local，
真实 RPC attention 结果通过既有容差。取消后验证 GPU buffer 可安全复用；没有
完整到达证明的 Host slot 可以继续隔离，不把一次取消当成传输完成证明。

完成状态丢失用例在真实 native write 成功后注入 status 异常：源端注册保持，
`stop_source_transfer` 拒绝复用；目标没有写入 GPU KV，运行时 guard 拒绝错误请求，
Host backing 在 reaper 后仍保留。layer 模式的错误唤醒 event 不作为 Local 证明。

原生预留上界 23、实际冻结长度 9 时：

| allocator page | 实际预留 token slots | 导入后仍持有 |
| ---: | ---: | ---: |
| 1 | 23 | 9 |
| 4 | 24 | 12 |
| 16 | 32 | 16 |

每例都验证 `_pre_alloc` 只调用一次、admit 绑定保持原 row generation、最终 free
恢复全部容量。CPU 还覆盖 ordinary lease 冲突、部分分配失败、取消 tombstone、
请求身份变化、冻结历史/几何不匹配和缺层 manifest 的消费前拒绝。

DSV 使用 group=3、range=17，实际按 native FULL page 对齐，八例分别完成
`3 / 3 / 3 / 3 / 10 / 10 / 522 / 522` 帧。所有例子均验证 bulk 阻塞期间可以
Remote Decode，state 字节及 tail/padding 精确一致，copy 完成后仍受 GPU forward
proof 限制，最终 Local、源端只 free 一次。C2/C4/C128 最大 attention 绝对误差
分别为 `0 / 0.00390625 / 0.125`，均通过既有 BF16 容差；不是跨模型精度指标。

## 复现

```bash
export CUDA_HOME=/workspace/conditional-graph-20261019/cuda13
export PATH="$CUDA_HOME/bin:/workspace/mixed-adapt-20261025/venv/bin:$PATH"
export PYTHONPATH=python:plugins/remote_attn
python test/manual/remote_attn/range_layer_cuda_probe.py --output /tmp/qwen-ranges.json
python test/manual/remote_attn/qwen_reservation_cuda_probe.py \
  --source-root "$PWD" --output /tmp/qwen-reserve.json
HANDOFF_D2D_GROUP_SIZE=3 HANDOFF_D2D_TOKEN_RANGE=17 \
  python test/manual/remote_attn/dsv_d2d_cuda_probe.py --p4 \
  --source-root "$PWD" --output /tmp/dsv-ranges.json
HANDOFF_D2D_TOKEN_RANGE=17 python test/manual/remote_attn/group_layer_cuda_probe.py \
  --suite dsv-local --source-root "$PWD" --output /tmp/dsv-local.json
```

本次验证正确性、完成依赖、页级所有权与保留规则；未测整模吞吐。统一 Qwen
路径当前按矩形等待 ACK 后复用 slot，没有组间双缓冲流水，不能由 batch 数或
staging 内存减少直接推断吞吐提升。
