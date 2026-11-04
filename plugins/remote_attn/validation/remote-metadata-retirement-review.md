# 完整最后 chunk 后的 metadata 与 KV 退休审阅

2026-11-04，基于 adb76215cc。修复后 TestRemoteAttn 全部 267 项通过，
49 hooks / 125 core interfaces、Ruff E9/F63/F7/F82、git diff --check 通过。

## 提前 metadata 与接纳

Qwen 的 Source.finalize 保持独立 metadata 线程，完整 GPU proof 后发送 aux，
随后发布 ready.json，再提交 canonical final。bulk 在另一个 native FIFO 中处理。
审阅发现 metadata 虽脱离 FIFO，但此前仍以普通任务争用 TE 锁。现在 Qwen aux
与 DSV sideband aux 都使用线程局部 control scope，使内部原生 _transfer 以 urgent
排队；不提前取得非重入锁，避免 send_aux -> _transfer 嵌套死锁。
原有四次 urgent 后允许 bulk 前进的公平策略保持。metadata 无法抢占正在执行的
TE 批次，但无需等待整条 KV 队列清空。TCP aux 路径本身独立于 TE bulk FIFO。

新确定性回归先持有 TE 锁、排队 bulk，再调用真实 Source.finalize。metadata 必须
先获得锁；随后 bulk 故意保持未完成。调用真实 Controller.early_targets，确认
receiver 尚未 Success 时 D 已被接纳，keep_receiver=True，KV release callback
尚未执行。attention/RPC 与完成事件由已有控制、GPU 探针覆盖。
这验证提前 Decode 的接纳条件，并不声称一次完整模型推理的实测 TTFT。

DSV 必须额外等待 indexer/scales scatter、SWA/carry/metadata 同代屏障；global
仍在途不阻止接纳。现有 progressive early-commit、GPU/TP retirement、D cache
publish/cancel 与真实 TP2 Gloo quarantine 回归均通过。小请求可能先传完 KV，
这时自然没有 Remote window，不能承诺每次都会发生提前 Decode。

## 修复的重复释放边界

- P Source.retire：KV callback 成功后，metadata 清理若抛错，之前可能在重试中
  再执行 KV callback。现在首次破坏性清理前记录 retirement_failed，部分失败后
  拒绝重试；正常完成仍幂等。
- D Target.finish：receiver/ring/metadata 清理此前在 release_failed 保护之外。
  allocator 先修改再抛错时可能重试同一 slot。现在保护覆盖整个破坏性清理段。
- DSV ProgressiveTarget/Resource 完成后直接幂等返回；请求级失败在 RPC close
  之前拒绝重试。未证明完成的传输仍沿用 quarantine 保留策略。

新故障回归分别在 P 已 free KV 后注入 metadata 失败，以及 D metadata allocator
已修改后抛错，断言第二次调用不会再次执行释放。既有正常双 retire、在途 RPC、
SWA/真实 checkpoint/插入、overlap 和 TP quarantine 回归全部通过。
部分清理失败会保留未确认安全的资源并报错；没有新增在线恢复或自动重试协议。

## 4060 结果与边界

- [实际 Mooncake GPU reader/slot 生命周期](remote-metadata-lifetime-4060.json)：
  在途取消保留 pin，过早关闭被拒绝，完成后复用，RPC teardown 不破坏原生注册。
- [P overlap 与 cache drain](remote-metadata-overlap-4060.json)：normal/cancel/revoke
  均无释放后写入；缓存锁/IO callback 为 fixture，真实 hybrid cache 与 TP2 在 CPU。
- [最后 chunk 完整计算 CUDA proof](remote-metadata-forward-4060.json)：Qwen BF16/FP16
  传输流等待完整 event，DSV event 覆盖真实 tail graph 的写入。

GPU 设备为 RTX 4060 Laptop，Mooncake 物理 TCP，非 H20 RDMA；未加载完整模型。
三份 JSON 的源码 SHA256 已与最终修复逐项核对一致。
