# D2D 分组与 token 区间

Qwen3-8B、DSV4 和 DSV4.1 的在线 D2D 已统一为 **layer group × token range**：

```bash
export HANDOFF_D2D_GROUP_SIZE=4       # 默认 1
export HANDOFF_D2D_TOKEN_RANGE=1024   # 默认 1024
```

配置在源 Decode 实例，两项均为正整数。完整的原生预留、冻结、区间到物理页的映射、
渐进切 Local、取消和回收规则，见 [二维搬运协议](D2D-LAYER-TOKEN-RANGE-ZH.md)。

当前结果见 [二维搬运 4060 验证](validation/d2d-layer-token-range-4060.md)。
此前仅合并 layer group 的实现与 28 项 CUDA 结果保留在
[历史 group-layer 验证](validation/d2d-group-layer-4060.md)，其源码哈希对应此前版本。
