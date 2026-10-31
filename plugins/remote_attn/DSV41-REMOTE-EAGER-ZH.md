# DSV4.1 Remote sparse attention：post-PD eager 最小闭环

此版本仅完成原生 PD **完成后**首轮 D decode 的真实稀疏 RPC：P 计算不可变 global
前缀的已选 attention，D 保留本地部分，再切回 Local。没有 early admission、传输
未完成时 decode、按资源 progressive 迁入或 Remote 分段图；不据此声称 TTFT/吞吐收益。
全量 KV 仍通过既有 layerwise PD 传到 D。生产代码仅修改插件，SMG/SGL 核心不改。

## 数据归属

| 资源 | 归属和读取规则 |
|---|---|
| 已完成 global compressed KV 前缀 | P 保留只读 lease；RPC 只读 D 选中的 logical position，不传 D 物理地址给 P |
| indexer payload/scale、candidate/top-k | 通过原生 PD 到 D；D 原生 indexer、候选缓存和 top-k 是唯一权威，不在 P 重排/截断或重做 top-k |
| 新生成 compressed KV | D 原生 compressor 写入；最终 top-k 中属于新增部分的 slot 在 D 计算 |
| ratio-2 pending half-pair/carry | 原生 final-chunk carry 到 D 并在 D 接续；P 仅可读 `< floor(prompt_len / ratio)` 的完整压缩行 |
| SWA 和 attention sink | D 原生存储/更新；SWA-only 层走原生 attention，带 Remote global 的层在 D 合并本地 SWA/新增行，sink 仅计一次 |

共享消费者按原生 `source_layer_of` / `layer_mapping` 读取生产者资源；不假定每层一份 KV。
D sparse metadata 的 physical slots 先按该请求的 FULL page map 分为 P logical 前缀位置
和 D 新增 slot。保留 top-k 顺序、重复项、长度与 padding 语义。P 再用自己冻结的 page
map 定位，拒绝越界位置、错误 ratio、陈旧 epoch/attempt 或其他请求的页面。

## 请求链路与保护

1. P 最终 chunk 沿用既有分层发送、host staging 和原生 SWA/carry/metadata final 接口。
   P 为只读前缀注册原生插件 `RemoteHub` 的 token/epoch lease，发布本代 `remote-service.json`。
   服务端在原生 sender Success 前拒绝 sparse read，即使请求已经 attach。
2. D 保留原生 receive host staging、H2D/scatter、metadata gate 和 TP MIN。
   只有 native `_commit_transfer_to_req` 完成才激活此次首轮 eager 窗口。
3. 模型继续运行原生 compressor/indexer。插件只在 D 的 native attention backend 实例上绑定
   eager 入口；不全局导入/替换 Qwen attention。选中的 P global 前缀经真实 RemoteHub TCP
   计算 partial output/LSE，D 合并本地结果。当前是选中行的 FP32 Torch eager 参考计算；
   读取支持 V41/V41_NEW、V41_FP4/V41_FP4_NEW，SWA 主格式须为 V41 编码。
   混合 batch 的普通请求保留 native 输出，SWA store 不重复。
4. 完成该请求的第一轮 D model forward 后停止 RPC、写本代 `remote-done.json`，后续恢复
   原生 Local full graph/overlap。此转换基于“KV 已经全部本地 + 完成同一轮 decode”，
   不依据某个 rank 的独立 copy 查询，也没有新加入的逐 token CPU collective。
5. P registry 收到完成事实先 tombstone，拒绝新 reader；已有 reader 须经过原生
   `RemoteJob.finish` 的 GPU fence 和最后一个 `release_ref` 才能触发 release callback。
   FULL/SWA 缓存 pin、request row 与源分配继续保留，随后沿用 P overlap 已有事件驱动 TP
   最终退休协议。GPU fence 失败保留资源；没有跨故障自动恢复。

取消/retraction 关闭旧 target 的 attach 并写旧 generation 完成事实；旧回调不会清除
复用行上的新 target。TCP 断开不代替 GPU 完成证明。P 请求 attempt/row、D target
attempt/row 和 epoch 校验阻止陈旧读取。P 串行对照仍沿用原有保守释放路径。

## 原生启动

先使用 [原生 P/D 启动文档](DSV41-LAYERWISE-ZH.md) 中已验证的参数数组及公共环境。
保持纯文本、同机单 pair、同 TP、PP=DP=CP=1、DSpark/HiCache 关闭，D radix cache 仍关闭。
P FULL/SWA cache、P/D scheduler overlap、optimistic prefill、chunk 和 transfer batch 保持现值。
Remote 当前要求原生 `flashmla` backend；D 的 mega/TRTLLM 路径启动时拒绝。

两端设置：

```bash
export HANDOFF_PRE_LAYERWISE=remote
export SGLANG_DISAGG_STAGING_BUFFER=0
# HANDOFF_PD_DIR 用同一份本机共享目录，每轮重启换新目录。
```

P：

```bash
export HANDOFF_PRE_LAYERWISE_HOST_STAGING=1
export HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=0
python -m sglang.launch_server "${P_NATIVE_ARGS[@]}" \
  --language-model-only --dsv4-attn-backend flashmla
```

D：

```bash
export HANDOFF_PRE_LAYERWISE_HOST_STAGING=0
export HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING=1
python -m sglang.launch_server "${D_NATIVE_ARGS[@]}" \
  --language-model-only --disable-radix-cache --dsv4-attn-backend flashmla
```

已有 native args 不重复填写 backend；将其原值改为上述兼容选项。P send / D receive
staging 可各自关闭做对照。无需 JSON 包装器，SMG 继续原有 PD router 命令。
DSV 模式不捕获 Qwen 的 Remote graph bank；普通 Local graph 配置保留。

## 确认路径与验收

- 原生 `/health`、`/server_info` 先确认模型、backend、cache/overlap 和 chunk 配置。
- 本代 `remote-service.json` 应有 `boundary=first-post-PD-decode` 及上述 owners。
- 首轮 eager decode 后 `remote-done.json` 有 `remote_calls > 0`、读取层号及完成 reason；
  prefix 太短/top-k 没有 P 部分时可为 0，不算成功覆盖了 Remote 计算。
- `released.json` 只能出现在 P reader fence、native PD 和 TP 退休证明均满足之后。
  普通 `done.json` 仅表示 native PD 完成，不能替代 Remote reader 的完成证明。

本地 136 项测试、真实 TP2 和 RTX4060 小型 GPU/TCP 记录见
[验证记录](validation/dsv41-remote-eager-4060.md)。尚未加载完整模型、运行完整 SMG/P/D
服务或多 GPU NCCL；未证明 token/logprob 一致性和生产收益。完整模型验收应分别比较
native/layerwise/post-PD eager 的输出、资源回收和开销。本阶段主要验证语义和接口，
下一阶段才接全局 KV 传输尚未完成时的 D 接纳与逐资源 Local 切换。
