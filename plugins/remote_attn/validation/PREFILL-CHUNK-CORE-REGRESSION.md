# 上游合并回归：Prefill middle-chunk 字段移除

日期：2026-11-07。修复基线：`b8272bd0f6`，分支 `codex/remote-attn-pre-layerwise`。

`98fd13c5` 合并后的通用 PD remote/layerwise 和 DSV4.1 layerwise 均仍引用
已删除的 `Req.inflight_middle_chunks`；该访问在本次最新分支上也存在。它与 radix
是否开启、前缀命中率无关。此前 hybrid 路径的验证不能覆盖这两处准入判断。

上游 `dbeecc932d` 将 middle/final 判断改为当前批次的 `chunked_req` 身份；
`inflight_steps` 统计请求所有已 launch、但尚未处理结果的 step，不能直接替换
旧计数器并保留 `<= 0` 判断，否则 overlap 的 final batch 可能被错误拒绝。

## 修改

- `Controller.prepare()` 使用 `req is not batch.chunked_req`，仍要求 final
  `extend_end`、bootstrap、sender/layout 和原有所有权条件满足后才预留尾部。
- `DSV41Controller.candidates()` 同样使用批次身份。`allow_middle=True` 的原有
  progressive/overlap 准入保留；未修改其发送、冻结、退休和 KV 释放协议。
- `debug_queue` 改为输出真实 `inflight_steps`，去掉旧字段默认为 0 的误导性输出。
  此诊断 JSON 不再提供 `inflight_middle_chunks`。
- 启动预检要求 `ScheduleBatch.chunked_req` 和 `Req.inflight_steps` 的实际声明，
  mixed core 缺失时在启动阶段明确拒绝，不安装旧版本兼容属性或默认垫片。
- 去掉夹具中手工补出的旧字段，以当前 core 的请求/批次字段构造测试。

## 验证

4060 主机的独立源码快照，CPU 回归：

- 新用例覆盖通用与 DSV 两条路径。当前 batch 将请求标为 middle 时，即使 live
  请求的 `extend_end` 已到末尾、`inflight_steps=0`，也不创建 final 发送或推进
  sender cursor；DSV 的显式 `allow_middle` 准入仍可用。
- final batch 在 `inflight_steps=1` 时仍正常预留、绑定真实 prefix-stream source；
  另一个 parked 请求也不能将该 final batch 误判为 middle。四个子场景全部通过。
- 既有无 prefix-stream 的通用尾块测试继续通过，覆盖 middle 不消耗 cursor、
  final 尾部只预留一次、cache lease 与 Decode 已缓存前缀的相对索引。
- 新增预检负例分别删除批次字段和请求 step 字段，均在运行前被拒绝。
- 独立负对照仅恢复 `b8272bd0f6` 的两个 controller 文件，保留新夹具和用例：
  四个子场景全部复现 `AttributeError: ... inflight_middle_chunks`，两条路径
  分别命中原 `pre_layerwise.py:853`、`dsv41_layerwise.py:528`。

完整主套件 **377 项通过**（188.909 s），native contracts **23 项通过**
（18.581 s），共 400 项；3 项定向回归也通过（为主套件子集，不重复计数）。
Ruff 定向检查、空白检查、51 hooks / 132 core interfaces 预检通过。
源码 SHA-256 与机器结果见
[prefill-chunk-core-regression.json](prefill-chunk-core-regression.json)。
原始日志：`/private/tmp/remote-inflight-results/results.tar.gz`。

被测请求/批次是 ABI 夹具；实际运行插件准入、sender 尾部预留、prefix-stream
绑定、原生 cache 方法和接口预检。没有启动完整 Qwen3-32B / DSV 模型服务，
不将这些结果视为 H20 的 0%/90% 命中性能复测。
