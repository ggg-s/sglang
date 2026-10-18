# 动态 compact verify — RTX 4090 真实服务

Qwen3-0.6B 同时作为 target 和 STANDALONE draft，FP16、TP1、Triton attention。
GPU 标称 RTX 4090，实际显存 49,140 MiB；环境 Torch 2.14.1+cu130、Python 3.12。
基线 `9b88ffa222` + 本轮补丁；25 项最终源码散列见 `summary.json`。

五组各完成 53 条请求、7,552 个输出 token，总计 **265/265**：

| 配置 | 验证轮数 | 成本样本 | 验证标签 | 混合批次 | 实际 token 图档数 |
|---|---:|---:|---:|---:|---:|
| 原生 STANDALONE K8 | — | — | — | — | — |
| 固定 compact SA8/neural3/gate1 | 165 | — | — | 84 | 28 |
| 动态 compact SA8/neural3/gate1 | 167 | 161 | 6,573 | 95 | 25 |
| 动态 SA8/neural2/gate3 | 203 | 197 | 6,408 | 0 | 22 |
| 动态 SA8/neural1/gate8 | 298 | 292 | 5,938 | 0 | 26 |

覆盖 c=1/4/16；c16 连续三波，请求输出长度交替为 96/128/160/192，包含缩批、
重排与来源混合。普通动态组使用默认冷桶/探索参数 32；两个 gate 边界组调为 8。
gate3/neural2、gate8/neural1 分别实际接受 777/767 次 SA 草稿；本负载这两组
未形成同一步混合，强制混合边界由 4060 原生算子回归补齐。
服务使用 `--disable-overlap-schedule --chunked-prefill-size -1`；本次真实服务不覆盖 overlap/chunked。

动态与固定 compact 的输出 token **53/53 完全一致**。所有 compact 组相对原生
均为 52/53 一致，唯一首分歧在 c4 的第 2 行、第 3 个输出 token：We / The。
额外重复固定 compact 并导出 top-5：二者在该路径 FP16 分数完全相同，
logprob 均为 -1.8312536478042603；640 个补查 token 均达到对应 target 最大分数。
Top-k 在同分元素间的排列不能当作 argmax 的 tie-break 规则。
独立 Transformers FP32 eager 参考更偏向 We，top-2 margin 为 0.01766777。
这些证据支持数值/同分分支的解释，不能据此宣称原生与 compact 逐 token 恒等。
原始证据见 `target-argmax-audit.json`、`divergence-reference.json` 和归档内 top-5。

真实启动发现并修复两处旧 compact 接入问题：draft runner 绕过父构造器时缺少
`ragged_verify_mode`；插件在 attention backend 尚未初始化时读取其能力。
4060 最终完整回归包含对应六个 CPU 流程用例，共 853 passed、2 skipped。

`evidence.tar.gz` 保存每组命令、配置、原始响应、服务日志、诊断及实际执行脚本。
历史 `run.py` 含最初的恢复旧服务逻辑；按用户后续要求，旧 Qwen3.6 主模型和
MTP 权重已删除、恢复取消，释放 24,465,334,272 字节。不要执行该历史入口。
`validate_service.py` 是整理后的五组复现脚本，只管理其自行启动的测试服务；
先准备对应源码、依赖和模型，确认 39923 端口空闲。模型散列见 `model-hashes.json`。
结束后测试服务全部停止，4090 显存 1 MiB；小模型保留。

这是正确性与调度 smoke。记录了单次请求耗时，但没有公平吞吐基准或性能收益结论。
DSV4 MTP 压缩 KV、TP8 NCCL、真实多卡档位对齐及吞吐仍需内网验证；EAGLE3/MTP
本轮由组件回归覆盖，不能由 STANDALONE 服务替代其端到端验收。
