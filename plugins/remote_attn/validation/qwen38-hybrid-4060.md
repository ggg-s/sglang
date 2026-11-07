# Qwen3.8 混合 D2D：4060 验证记录

日期：2026-11-07。分支：`codex/remote-attn-pre-layerwise`，修改基线：`385af70b53`。
原始结果：[qwen38-hybrid-4060.json](qwen38-hybrid-4060.json)。
实现与部署约束：[Qwen3.8 混合 D2D](../QWEN38-HYBRID-D2D-ZH.md)。

后续 review 发现初次夹具手工提供 `layer_types`，未覆盖原生配置启动。
该 P1 已修复，真实配置的红绿回归与最新 4060 结果见
[原生层映射 P1 验证](qwen38-native-layer-mapping-4060.md)；以下保留初次验证记录。

## 环境与范围

RTX 4060 Laptop，8 GB，Torch 2.13.0 / CUDA 13.0。验证使用小尺寸原生 KV、
GDN 状态布局与生产分配方法；隔离可选模型/CLI 导入和调度器调用边界。
实际执行 CUDA Conv、packed GDN、FlashInfer 和原生 Mooncake TCP。
完整 attention 使用 GQA 6:1、head_dim=256；GDN 使用 V/K heads=3:1、head_dim=128、
conv kernel=4（原生持久状态窗口=3）。模型层数缩为 8：6 个 GDN、2 个 full attention。

没有加载完整 27B 权重，没有进行真实多 GPU TP 推理。4060 没有 RDMA HCA；
共用 GPU staging 故障用例的 NIC/CQ 和外部写可见性使用明确的模拟边界，不能作为真实 RDMA 验收。

## 结果

| 项目 | 结果 |
|---|---|
| CPU 既有 remote-attn 回归 | 362 通过 |
| CPU 原生边界与混合迁移回归 | 19 通过，包含 8 个混合模型用例 |
| CUDA 混合迁移生命周期 | 8 通过 |
| 共用 Q/O/LSE 数据面 | 21 通过：真实 TCP、能力回退、模拟 GPU RDMA、取消/异常/迟到写 |
| 共用 bulk GPU staging | 11 通过：模拟 RDMA 正常、失败、迟到写 |
| 原生 hook / 接口核对 | 49 / 130 通过 |
| 新增实现/probe Ruff、改动 diff 空白检查 | 通过 |

GDN Conv 输出与 SSM 续算输出最大误差均为 **0**。Source 的冻结状态不参与续算；
不中断参考在私有副本上计算，目标在实际导入的不同原生 slot 上计算。
Source/target 状态 slot 为 **1 / 2**，请求 row 不等于状态 slot。

Remote GQA 相对完整 KV 的 BF16 基线最大绝对误差为 **0.0009765625**；相对 FP32
参考为 **0.0007832050323486328**，通过 `rtol=0.02, atol=0.001` 校验。
切 Local 后相对相同完整 KV 基线误差为 **0**。

group size=8、token range=3、冻结前缀=10，两个完整 attention 层的 K/V 共四个 entry，
分成 **4 帧**搬运，每帧最多四个 entry。保留 entry 完成事件，逐 entry 收集 Local 证明；
完整层需要 K/V 两项均完成。新生成的 token 位于冻结前缀之外，搬运后字节保持不变。
共观察到 **18** 次原生 Mooncake 单次提交（状态、bulk 和 Q/O）。

## 生命周期断言

- 原生预留同时计入工作和 tracking slots；KV/请求行仍充足而 GDN slots 耗尽时拒绝继续预留。
- 未冻结时先完成目标预留；拒绝无预留 freeze，拒绝缺失/重排 GDN 状态 manifest。
- 消费预留复用已有原生行和工作 slot，释放未用 KV 上界，不二次分配状态。
- 全部 Conv/SSM 安装并通过 CUDA event 后才清除 `mamba_needs_clear`，再进入 commit。
- 冻结读显式依赖独立 producer/forward stream；copy stream 依赖状态安装完成 event。
- 取消等待复制线程与 GPU/传输证明；取消 tombstone 拒绝迟到 import，原生资源只回收一次。
- 首个状态写后抛异常时，目标不会接纳或复用 GDN slot；明确 abort/退休后才释放。
- 完成未知时，KV、GDN slot 和两端 staging 仍被持有；取消/源端 resume 均不能绕过。
  随后执行迟到写，确认注册与 backing 仍有效，且不会因此发布 Local。
- 仅 copied 不足以释放源端；commit、copied 与远端 read lease 关闭后才退休源端 KV/GDN。

运行快照共核验 **6776** 个文件的 SHA256，执行前后均通过；结果 JSON 保存本次
14 个改动实现/测试文件的哈希，本地逐项核验与测试快照一致。

## 复现

安装本分支及 remote-attn 插件、Mooncake、FlashInfer、Triton 后运行：

```bash
PYTHONPATH=python:plugins/remote_attn:test/manual/remote_attn \
  python test/manual/remote_attn/qwen_hybrid_cuda_probe.py \
  --source-root "$PWD" --output /tmp/qwen38-hybrid.json
```

CPU 回归：`test/registered/unit/plugins/test_remote_attn.py` 与
`test/registered/unit/plugins/test_remote_main_contracts.py`。
共用数据面：`qo_mooncake_cuda_probe.py` 与 `rdma_staging_cuda_probe.py`。

Remote 窗口当前使用 eager；Local 回到原生 graph。实际 27B 模型的加载、原生 PD
整链路、多卡 TP、图恢复、正确性及真实性能，以及 HCA/CQ 的 RDMA 完成/可见性，
仍需在目标多卡机器上验收。
