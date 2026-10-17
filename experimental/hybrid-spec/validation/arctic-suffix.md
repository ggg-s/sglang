# ArcticInference SuffixDecoding 后端

通过 `backend="arctic"` 使用 ArcticInference 官方 `SuffixDecodingCache` 和 C++
后缀树，替换默认 `gpu_sa` 检索。无需将 Arctic 的 vLLM worker 移植到 SGLang；
MTP/EAGLE3/STANDALONE 保留原生 neural draft、target verify 和提交路径。

## 范围与语义

- 支持 `full`、`split`、`split_no_skip`。gate=0…K、目标词表映射、TP 共识、
  GPU 稳定行号及紧凑 tail Graph 与 GPU SA 共用；split API 仍为 v5。
- 使用 Arctic 原生单链 (`use_tree_spec=False`) 查询，至多 K 个候选，保留其
  频次、概率阈值和匹配长度驱动的候选规则。原生 score 用于其 local/global
  树内部选择，不替代本项目的 gate，也不作为 neural proposal 概率。
- 当前 verify 布局固定 K。默认 `partial_drafts="fallback"`：不足 K 整轮回退；
  原有可选 `zero` 策略仍表示补零草稿并经 target verify，不等于变长验证。
  未接入 Arctic 的树状候选、动态 verify 宽度或 score 驱动的原生混合调度。
- `history_mode="cpu"`，显式要求 `--disable-overlap-schedule` 与
  `--chunked-prefill-size -1`。Arctic 读取 CPU 已确认历史，query/H2D 在 Graph 外
  完成；split Graph 内只执行 gate/共识/选择。CPU 历史版本必须等于 GPU
  `seq_lens+1`，否则该行统一回退。device 历史和 CPU overlap 组合明确拒绝。
- 自身 prompt 私有；开启 `enable_global_pool` 时只有已确认 response 进入
  Arctic 全局树，活跃 response 可供其他请求检索。完成时丢弃 local prompt，
  最多保留 `max_shared_requests` 个完整 response，按 FIFO 淘汰。
  `max_requests` 只约束活跃请求，完成 donor 不占活跃槽。
- 每次仅追加确认增量；改写/撤回历史会删除旧树并递增 generation。完成时
  从最终 CPU 输出重新对账，包括最后一轮尚未被下次 draft 消费的 token，
  排除停止条件裁掉的尾部。异常结束、弱引用失效、flush 和槽位复用清理旧状态。
- Arctic CPU 树不消耗 GPU SA 状态池；仍有 CPU 查询、候选上传和 split Graph
  的成本。GPU SA 的吞吐结果不能套用于 Arctic。

Triton draft attention 支持普通 EAGLE/MTP、EAGLE3 和 STANDALONE；DSV4 MTP
继续使用原生 dsv4 draft attention。其他模型、并行和采样限制同主 README。
`max_history_tokens` 限制整个请求历史，`arctic_max_tree_depth` 控制树深/查询上下文；
`shared_context_tokens` 属于 GPU SA donor 选择，Arctic 使用其自己的树查询规则。

## 安装

Arctic 为可选依赖，默认安装、导入和 GPU SA 运行不需要它。可安装
`pip install -e './experimental/hybrid-spec[arctic]'`，需保证 upstream 原生 suffix
扩展可导入。缺少扩展时启动明确报错，不会静默使用其他后端。

本轮验证固定官方源码
[`9d4643881608178f0f58a33d1b61919ca246997f`](https://github.com/snowflakedb/ArcticInference/tree/9d4643881608178f0f58a33d1b61919ca246997f)。
为保持现有 Torch/CUDA serving 环境，4060 仅编译官方 CPU suffix 扩展；
没有安装整个 Arctic/vLLM 栈或替换 Torch。完整依赖构建策略由 upstream 管理。
在服务使用的 Python 环境运行下列示例（`ARCTIC_SOURCE` 为新目录）：

```bash
ARCTIC_SOURCE=/实际目录/ArcticInference
# 已存在源码时使用独立 checkout，避免覆盖其他工作的修改。
git clone https://github.com/snowflakedb/ArcticInference.git "$ARCTIC_SOURCE"
git -C "$ARCTIC_SOURCE" checkout --detach 9d4643881608178f0f58a33d1b61919ca246997f
python -m pip install cmake nanobind==2.9.2 numpy
cmake -S "$ARCTIC_SOURCE/csrc/suffix_decoding" -B "$ARCTIC_SOURCE/build-suffix" \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$ARCTIC_SOURCE/arctic_inference/suffix_decoding"
cmake --build "$ARCTIC_SOURCE/build-suffix" -j 2
export PYTHONPATH="$ARCTIC_SOURCE:$PWD/experimental/hybrid-spec/src:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from arctic_inference.suffix_decoding import SuffixDecodingCache'
```

后缀模块来自上游 Apache-2.0 源码，本仓库不 vendor。官方 CPU 构建需要支持
C++20 的编译器、Python 开发头文件和 nanobind。

## 启动

在已可运行的原生 neural-spec 命令上设置：

```bash
export HYBRID_SPEC_CONFIG="$PWD/experimental/hybrid-spec/configs/arctic-suffix-split.json"
# 保留 --speculative-algorithm EAGLE / EAGLE3 / STANDALONE
# --disable-overlap-schedule --chunked-prefill-size -1
# --speculative-num-steps 8 --speculative-num-draft-tokens 9
# --speculative-eagle-topk 1
# --max-running-requests 8 --cuda-graph-max-bs-decode 8
# --speculative-draft-attention-backend triton
# DSV4 MTP 上一项保持 dsv4
```

仅把 `backend` 改成 arctic 而继续配置 `history_mode=device` 会报错，必须同步
设置上述 CPU 历史/调度选项。默认 GPU SA 配置不变。

模板选用 K=8/G=2，`arctic_max_spec_factor=2.0` 用于在匹配较短时容许更多候选；
这属于策略参数，需按负载校准。其他 Arctic 参数默认值：max depth 64、factor 1.0、
offset 0.0、min token probability 0.1。更改 K 时同步修改 JSON 与服务 steps/token
参数，并保持 G≤K。`enable_global_pool=false` 可单独评估请求内后缀树。

## 验证

4060 完整回归已通过 663 项，跳过两项需要 2/8 卡的 NCCL 用例；另增加
Arctic+dsv4 矩阵单独通过 46 项，两轮合计 **709 passed、2 skipped**。
原始日志、源码哈希和详细范围见
[4060 证据](arctic-suffix-4060/README.md)。

```bash
python -m pytest experimental/hybrid-spec/tests/test_arctic.py -q -rs
python -m pytest experimental/hybrid-spec/tests -q -rs
```

`test_arctic.py` 要求真实官方 C++ 扩展；没有安装时会跳过，不能把跳过计为通过。
4060 的验证环境已安装该扩展，运行时还明确检查它可导入。
Arctic compact CUDA 用例使用真实 CPU 查询和 H2D staging，覆盖 G=0/1/2/3/K、
K=1/2/3/8、B=1/8/16、混合命中和 padding；比较 full/split 草稿链及 tail 工作量。
神经模型和 child attention 执行仍是合成计算，不能据此声称真实权重/serving
或 TP8 NCCL、吞吐、质量已验收。
