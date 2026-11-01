# DSV4.1 探针：真实 logits 输出与 sampler 隔离

日期：2026-11-01；基线 `6bf9452493`。按用户要求仅做本地 CPU 测试，没有上传源码或运行4060。

## 修改

- DSV CUDA 探针统一经 `load_native_backends` 加载原生 full/BCG 定义，显式注入
  `dataclasses`，避免忽略 import 的源码加载方式在 BCG 中触发 `NameError`。
- 小模型从真实源码加载 `LogitsProcessorOutput`，返回 logits 和 hidden states，
  不再用单个 Tensor 代替模型输出；缓冲容量和行数查询同步读取 logits。
- 输出经过源码加载的原生 `DecodeCudaGraphRunner.execute` 及其输出处理，再模拟
  sampler 写入。保留 Local full graph 到 sampler 之间的原生 dataclass 包装行为，
  没有在探针里额外清空字段或自行实现输出拷贝来掩盖残留。
- 校验9个 sampler 字段在交付前为 `None`，然后写入非空值，覆盖重复同一 key、
  batch 切换、Remote→Local；另有负例确认脏字段即使 logits 正确也会被检测到。
- CUDA 探针保留 Remote batch 1/2/4 和四种 candidate variants，Local 改为依次回放
  4/1/2/4/1/2/4/8。结果记录输出类型、原生输出处理、sampler 检查、batch 序列，
  并增加 logits processor 和 decode runner 的源码哈希。

此次只改测试、探针支持代码和本记录，SMG、SGL 核心与插件运行逻辑没有修改。

## 本地结果

- **172项插件回归通过**，包括61项 DSV4.1 测试与真实 Gloo TP2 检查。
- **7项共享 BCG 输出回归通过**；共179项测试。
- 新增 CPU 回归使用原生 full/BCG 的捕获、缓冲、回放和原生 decode 输出处理方法，
  以及真实 DSV bank 选择。Remote 序列为 4/1/2/4/1/2/4，共7次；切为 Local 后序列为
  4/1/2/4/1/2/4/8/8，共9次。数值匹配、外壳独立、tensor 地址复用、sampler 隔离通过，
  Local 没有进入 Remote attention 分支。
- 将探针支持文件恢复到基线时，新增回归因输出仍为 Tensor 失败；保留新输出但去掉
  `dataclasses` 注入时，原生 BCG 报 `NameError: name 'dataclasses' is not defined`。
  正常代码下两项新增测试通过。
- 源码预检仍为 **44 hooks / 108 core interfaces**；isort、ruff、diff 检查通过。

```bash
python test/registered/unit/plugins/test_remote_attn.py \
  TestRemoteAttn.test_dsv41_probe_native_logits_survive_bank_batch_and_sampler_changes \
  TestRemoteAttn.test_dsv41_probe_rejects_dirty_sampler_output
```

CPU 测试以小模型重新计算并写入捕获结果，替代 CUDA graph 执行边界；没有执行真实
GPU capture/replay。输入 staging、timer、共享读完成信号不是该输出夹具的验证范围。
没有运行完整 DSV checkpoint、SMG/P/D、FlashMLA、NCCL 或真实性能测试。
CUDA 探针的 GPU 执行和新 batch 序列仍待目标机器验证。
