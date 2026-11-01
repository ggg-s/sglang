# P 缓存前缀 Remote progressive：本地验证

日期：2026-11-01；基线 `dd370385af`。按用户要求仅做本地测试，未上传源码、
未执行4060或其他远端 GPU 测试。

## 实现与边界

Qwen3/MHA 与 DSV4.1 共用请求范围计划。复用原生 cached-prefix、middle、final
的逻辑 cursor、目标页规划和 Mooncake worker；物理 payload 分批提交，DSV indexer
优先，Qwen metadata 期间暂停提交更多 global。SMG、模型计算与 graph runner 不变。

P 的既有缓存读租约覆盖排队、在途和 Remote 读者。D 按 generation、资源和实际
区间记录完成，不能用尾块补齐缺失前缀。native staging 的 GPU scatter event，
或 Qwen direct 接收可见性处理及 GPU event，构成切 Local 的完成证明；复用有序
退休协议确认 TP 一致。取消/retraction 先排空或隔离未知 IO，再交还缓存所有权。

SGL 核心只为 `request_layer_staging` 增加两个可选关键字：`chunk_id_base` 与
`stream_generation`；默认8帧协议不变。Qwen 非尾块分配由插件调用原生
`handle_staging_req`，通过范围证明替代外层入口的 final-only 限制。

资格不满足时，在首次接管前保留旧路径。DSV 沿用纯文本、既有后端和 P overlap
接管租约条件。接管后不能因注册变化或失败而静默回退并释放页。

## 本地结果

- **205项插件回归通过**，包含13项新增范围测试和真实 Gloo TP2 协议测试。
- **7项共享 BCG 输出回归通过**，合计212项；没有以 CPU 测试代替真实 CUDA replay。
- 源码接口预检通过：**47 hooks / 114 core interfaces**。
- isort、ruff 语法/未定义变量检查与 `git diff --check` 通过。

新增覆盖：

| 路径 | 可观察断言 |
|---|---|
| Qwen/DSV 原生 cached-prefix send | cursor只保留一次原生预留；有界 payload、唯一 staging id、重复调用不重发 |
| Qwen cached→middle→final | final Source身份接续；metadata已发而global仍排队；canonical marker等待所有区间；真实 radix页最终可回收 |
| DSV cached→middle→final | 完整indexer优先完成，原生SWA/carry/aux只发送一次，global继续后台推进 |
| Qwen native ring | 原生allocator、K/V scatter与watermark；非尾块可分配；event前不计完成；释放后的重复请求不重新分配 |
| DSV native ring | 同一物理indexer entry的两个不相交区间分别scatter；全部完成前不发布entry Local |
| 跨机控制先到 | native9帧请求先于journal镜像仍可注册；重试幂等，generation换绑/重叠被拒绝 |
| Direct receive | P发送回调不足以Local；D可见性处理后的event未完成仍保持Remote |
| 取消与重用 | native abort/free等待已排队IO；活跃attempt不能reset；peer注册变化保留pin及在途存储 |
| 对照开关 | P `HANDOFF_PREFIX_REMOTE_PROGRESSIVE=0` 保留原生单块缓存发送，不建立新stream/pin/pump |
| 兼容性 | 默认staging编号及8帧消息保持原样；原有graph/overlap/cache回归继续通过 |

TP2 对两种模型分别检查：resource0只有尾块时，即使resource1完成，也只能切Local1；
仅rank0收到缺失前缀不产生新命令；两rank均收到后才切Local0，最后退休父请求。
两rank的collective票向量长度均为 `[5,5,5]`，Local集合均为 `[0,1]`。
等待期间及Local后各32轮观察新增投票数均为0。

## 反向验证

- 用旧cached-prefix路径替代接管：有界范围用例失败，未建立范围计划。
- 将完成的尾区间错误扩成整层完成：缺前缀用例失败，阻止FIFO式隐含覆盖回归。
- 恢复Qwen原生final-only外层分配入口：非尾块ring用例复现
  `layer staging must cover the final prompt chunk at PP1`；插件适配后通过。

这些反向替换只存在于本地临时测试进程，没有写入仓库运行代码。

## 测试环境与后续验收

本机macOS/CPU torch环境缺少完整GPU运行依赖，使用源码加载的真实原生sender、
cache、allocator、scatter、BCG输出逻辑及 `CustomTestCase`；GPU事件、DMA/网络
worker与模型forward边界由测试控制。TP一致性使用真实双进程Gloo和本机socket。
本地执行器为 `/private/tmp/remote-attn-preflight-20261030/bin/python`，插件套件通过
同目录 `run_cpu_contracts.py` 执行；BCG通过 `/private/tmp/run_qwen_bcg.py` 执行。
完整依赖环境的仓库测试入口：

```bash
python test/registered/unit/plugins/test_remote_attn.py
python test/registered/unit/model_executor/runner_backend/test_breakable_cuda_graph_backend.py
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only --sglang-root .
```

此次未加载完整Qwen3/DSV checkpoint，未执行真实CUDA/RDMA PD，尚不能宣称新的
GPU正确性或性能收益。部署验收需固定原生配置，比较native、旧Remote、新Remote，
证明P缓存命中时D首次decode早于缓存global传输完成，并检查输出及资源最终回收。
详见[接入说明](../PREFIX-REMOTE-PROGRESSIVE-ZH.md)。
