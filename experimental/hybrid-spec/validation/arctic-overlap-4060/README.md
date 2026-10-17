# Arctic overlap / chunked prefill RTX 4060 验证

最终完整回归：**735 passed、2 skipped（245.46 秒）**。单卡环境跳过 NCCL
2/8 卡用例；测试结束显存为 0 MiB。

本轮基于 `bc44c2aa40ef88e3c7aed761744975c39d1511d1`，在
`/workspace/arctic-overlap-20261017/repo` 独立 worktree 验证。
先 git fetch 已发布分支，再上传本次补丁；沿用已编译的官方 Arctic suffix
扩展，固定 upstream `9d4643881608178f0f58a33d1b61919ca246997f`。

完整结果、环境与源码 SHA256 见 `metadata.json`，原始输出见 `regression.txt`。
测试前的探索性中断运行不计入通过数量；最终完整运行覆盖最终源码。

本次新增覆盖：

- CPU 尚无 prefill anchor / 尚未提交 verify 输出时，使用 GPU bonus 和接受路径；
  不等长接受、全局 node index、padding、batch 重排和 CPU 追赶后不重复追加。
- 分块中间行丢弃临时 bonus；同批其他 final 行正常索引，最后一块包含完整前缀。
- 重预填充、取消旧 ticket、异常快照、idle、结束回收、同 rid 新请求和容量回退。
- GPU 私有尾部与 CPU 确认共享树隔离；停止截断后只保留最终 response。
- 官方合并缓存与本实现分离 local/global 树的候选、匹配长度选优对照。
- 反馈版本/index/count/token 异常回退，GPU 进度计算剩余预算，历史超容量回退。
- MTP/EAGLE3/STANDALONE 的 overlap/chunked 配置组合；已有 full/split/no_skip
  gate、GPU 动态行号、compact tail、DSV4 形状模拟和 TP 回归保留；
  EAGLE3/STANDALONE compact 用例保留原生 Triton metadata/index kernel。

环境：RTX 4060 Laptop 8 GB，WSL2，Python 3.10.18，Torch 2.8.0+cu128，
Triton 3.4.0。命令（工作机仓库根目录）：

```bash
PATH=/workspace/dsv4-lease-validation-20261017/venv310/bin:$PATH \
PYTHONPATH=/workspace/arctic-suffix-20261017/ArcticInference:/workspace/arctic-overlap-20261017:experimental/hybrid-spec/src \
CC=gcc-12 CXX=g++-12 CUDA_HOME=/usr MAX_JOBS=2 \
TORCH_CUDA_ARCH_LIST=8.9 GLOO_SOCKET_IFNAME=lo \
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests \
  -q --tb=short -rs --junitxml=../regression.xml
```

轻量导入辅助沿用 [保存快照](../dsv4-lease-4060/lean_cuda_plugin.py.txt)。
真实服务不要加载该辅助。使用真实官方 CPU C++ 后缀树、GPU 路径打包、pinned
D2H/event、H2D 和 CUDA Graph；新增调度用例为构造的 prefill/verify batch，
神经模型和 child attention 仍为合成计算。不能据此声称真实调度器/权重 serving、
DSV4 TP8 NCCL、模型质量或吞吐已经验收。

内网移除 `--disable-overlap-schedule`，采用默认 chunk size 或显式设置
`--chunked-prefill-size 2048`；仍不可启用 `--enable-mixed-chunk`。
同样支持关闭两者作为对照。CPU 树每轮仍需接受路径 D2H、消费等待和查询，
是否有净收益需实测。完整验证矩阵建议见 [接入指南](../arctic-suffix.md)。
