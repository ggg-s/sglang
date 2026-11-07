# Qwen 原生层映射 P1：4060 验证

日期：2026-11-07。分支 `codex/remote-attn-pre-layerwise`，修改基线 `385af70b53`。
[机器结果与实现哈希](qwen38-native-layer-mapping-4060.json)。

## 修复

原生 `Qwen3_5TextConfig` 继承 `Qwen3NextConfig`，消费 HF 的 `layer_types`
参数却不保存该属性。模型与分配器实际依赖 `full_attention_layer_ids` 和
`linear_layer_ids`。控制器启动校验和 D2D manifest 现共用这两个原生属性；
检查它们完整、不重叠地覆盖模型层，并核对实际 full-attention KV / GDN pool 的层归属。
模型与 pool 不匹配时拒绝启动和生成 manifest。

wire schema 使用 `layer_mapping`，显式包含上述两组逻辑 layer ids；KV entries
和 Conv/SSM states 仍来自原生注册及物理映射。两端沿用完整 schema 相等校验，
旧的 `layer_types` schema 无法通过校验，需要两端一起升级。
本次未修改传输、CUDA event/stream、Local 证明、取消或退休流程。

## 回归结果

4060 Laptop 8 GB，Torch 2.13.0+cu130，Transformers 5.12.1。

- **红**：在未修复代码上，用真实原生 `Qwen3_5Config` 构造夹具；manifest
  阶段复现 `AttributeError: 'Qwen3_5TextConfig' object has no attribute 'layer_types'`。
- **绿**：整个 `test_remote_main_contracts.py` **21 项 CPU 回归通过**，在 4060 主机运行。
  新增测试分别构造 P/D 控制器并执行真实 `validate()`，校验 manifest 中的层归属和
  entry 映射；另一个测试改变原生 attention interval，验证启动与 manifest 均拒绝陈旧 pool。
- **CUDA**：`qwen_hybrid_cuda_probe.py` 的真实计算通过，包含 **10 项回归**。
  继续覆盖状态安装后接纳、K/V entry 分别完成、原生 slot 预算、取消、部分状态写异常、
  完成未知及迟到写隔离、无预留拒绝和不完整状态 manifest 拒绝。
- **源码**：测试前后 **6,776 文件 SHA256 核验通过**；本次三项实现/测试修改的哈希
  已与本地文件逐一核对。

GDN Conv / SSM 续算最大误差均为 **0**；Remote GQA BF16 最大误差
**0.0009765625**，Local 输出误差 **0**。源/目标状态 slot 为 **1/2**。
group size=8，token range=3，冻结前缀=10，共四帧、每帧最多四个 KV entry；
18 次原生 Mooncake TCP 单次提交。独立 producer stream 的冻结依赖及新 suffix
保持断言通过。取消和异常测试仍要求读写完成证明，完成未知时保留 KV、GDN slot 和 staging。

## 验证范围与复现

启动回归使用本仓库原生配置类（8 层缩小几何），不再手工创建 `layer_types` 属性。
它真实执行配置属性、控制器构造、校验、manifest 和 attention hook 绑定；
重型模型与 pool 构造边界使用夹具，原生 pool 分配/注册方法保留。
这是控制器启动回归，**没有启动完整 27B 模型服务器**。

CUDA 用真实 Conv / GDN / FlashInfer 和原生 Mooncake TCP。4060 无 RDMA HCA，
未验收完整权重、真实多卡 TP、端到端原生 PD 服务或真实 RDMA CQ。

```bash
PYTHONPATH=python:plugins/remote_attn:test/manual/remote_attn \
  python test/manual/remote_attn/qwen_hybrid_cuda_probe.py \
  --source-root "$PWD" --output /tmp/qwen-native-layer-mapping.json
```

CPU 注册用例：`test/registered/unit/plugins/test_remote_main_contracts.py`。
