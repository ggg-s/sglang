# H20 bs=1 full capture：错误作用域补齐，根因待整模确认

2026-11-03，基于 `78d1ee705d`，分支 `codex/remote-attn-pre-layerwise`。

用户报告 Remote buckets 为 1 时通过，为 2,1 或 1,2,4,8,16 时，最后的原生 full bs=1
在 capture_end 返回 cudaErrorStreamCaptureInvalidated。现无该 H20 首错日志或可用连接，
不能将此顺序相关性直接归因为某个状态未恢复，也不能宣称 H20 整模已修复。

## 确认的覆盖缺口与修改

RemoteGraphBackend 每个 bucket 先捕获 full，再捕获 Remote。此前只有 Remote 的
BreakableCudaGraphBackend 使用 thread_local；下一 bucket 的原生 FullCudaGraphBackend
仍默认 global。workspace 预分配也仅针对 Remote，不能作为 full bs=1 已修复的证据。

FullCudaGraphBackend 现接受 capture_error_mode，默认仍是 global，默认调用不增加
新 graph factory keyword。RemoteGraphBackend 装配时为其 full bank 设置 thread_local，
覆盖连续 full→Remote→full。其他原生使用者保留原有行为。预检检查构造参数与对应字段，
避免插件使用新字段而旧 core 静默忽略。没有调整 bucket 顺序、删除 bs=1 或放宽数据布局。

## 已取得证据

RTX4060 上独立 CUDA 控制：另一个线程在捕获期间调用 cudaMalloc。

- global：后台 cuMemAlloc_v2 返回 900，捕获状态 Invalidated，主线程 cuStreamEndCapture
  返回 901。
- thread_local：后台分配/释放成功，捕获状态 Active，graph replay 数值正确。

CUDA_LOG_FILE=stderr 记录了后台 API 与主线程 capture_end 的不同线程 ID。
这证明 capture_end 或第一个看到 Invalidated 的子模块可能只是受影响者，不能单凭
子模块 hook 将其判为触发源。此控制主动注入分配，不代表已找到 H20 实际触发线程。
本地探针和原始记录位于 /private/tmp/remote-attn-capture-audit-20261103。

本地新增 full scope 契约 3 项通过，完整插件回归 224 项通过（39.610 秒，含本机
TCP/TP2 Gloo）。测试使用已有 pyzmq，未安装新依赖。预检 48 hooks / 119 interfaces；
选定 ruff 规则、format、isort 与 git diff --check 通过。

## 连续捕获 GPU 对照

用户已批准 5 文件测试包 /private/tmp/remote-full-scope-cuda.tar.gz，16,559 字节，SHA256：
cf2a1e469183b782a873511b58547116d2decaa36bf6093f8a5f9dffbf74ea3e。
原 Windows SSH 跳板超时后，用户恢复了同机 Tailscale 的 Ubuntu 2222 直连。新地址
使用已有 [127.0.0.1]:2222 主机密钥严格校验通过，未接受新密钥或关闭校验；主机名 gq，
RTX4060 Laptop。上传与运行目录 /tmp/remote-full-scope-20261103.D9zYJp，复用已有 venv。

新增 full_capture_scope_probe.py 使用 source-loaded 原生 full/breakable 后端。
它**主动构造故障**：仅在先前已捕获 Remote bucket、且进入 full bs=1 捕获时，
通知另一线程执行 cudaMalloc。因此下表是错误作用域的受控验证，并非自然复现 H20 故障。
模型数学、TP、dedup、precarve 是小型夹具，不能据此宣称 H20 实际触发源已找到。

| Remote buckets | full=global 负对照 | full=thread_local 修改后 |
| --- | --- | --- |
| 1 | 通过；没有此前 Remote，未在捕获中注入 | 通过 |
| 2,1 | full 2 / Remote 2 成功，full 1 注入后失效 | 全部捕获/replay、数值检查通过 |
| 1,2,4,8,16 | full/Remote 16/8/4/2 成功，full 1 注入后失效 | 全部捕获/replay、数值检查通过 |

global 两组失效对照均记录后台分配错误 900、捕获状态 Invalidated、capture_end
错误 901；thread_local 两组对应后台分配成功、捕获状态 Active。bs=1 未被删除，
捕获顺序没有调整。测试也实际 replay 每个 full 和 Remote bucket 并校验数值。

修改后进一步复跑原有图探针：

- dense BF16 / Marlin workspace：42 次 Remote、6 次 full replay，15 份独立 workspace，
  72 次 RPC；capture RPC 为 0，数值及 sampler 输出检查通过。
- DSV V4 / 串行 prepare：78 次 Remote、8 次 full replay、582 次 RPC；capture RPC 为 0，
  最大绝对误差 0，live SWA/carry、candidate variants、padding 与 Local 恢复检查通过。

上述六组作用域对照与两组图回归的原始结果、API 日志见
[h20-full-scope-4060-results.json](h20-full-scope-4060-results.json)。返回结果合计 45 份
源码指纹与本地逐项一致，新增包上传及 GPU 测试现均已完成。

## 仍需 H20 整模确认

本次已验证跨线程不安全 API 可以引发此错误链，以及补齐 full capture 的 thread_local
可以阻止本次注入的跨线程分配干扰。H20 后续仍需第一条不安全 CUDA API/线程证据，
以及相同 MXFP4 Marlin/FlashMLA 服务在修改后的整模 bucket 对照。4060 组件图探针
不替代 H20、完整 checkpoint 或多卡 NCCL 验收。
