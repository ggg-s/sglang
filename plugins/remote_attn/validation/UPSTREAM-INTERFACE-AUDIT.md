# 插件与上游 core 边界审计

日期：2026-11-07。基线：`9c8e60e25b`，分支 `codex/remote-attn-pre-layerwise`。

## 范围与方法

扫描 `plugins/remote_attn` 全部 48 个 Python 文件，包括控制面。先将插件的
2041 种属性名（含 getattr/hasattr/setattr 字符串）与 `cc0cc820` →
`eae838a180` 的 core 字段/方法删除记录交叉，再按当前对象声明和调用点核对。
静态告警是候选，不能把同名属性在另一个类上的删除当作插件回归。

重点对象：Req/ReqKvInfo、ScheduleBatch/ForwardBatch、Scheduler/request receiver、
ModelRunner/graph runner、native PD queues、KV manager/sender/receiver、
TransferInfo/KVArgsRegisterInfo/KVArgs、原生 pool/cache 及其状态/页索引接口。
区分插件自有标记、原生按需初始化字段、继承/重导出和同名局部变量。

- 56 个 core 导入均可从源码解析，包括 utils 的 star re-export。
- 79 处可静态绑定的导入函数/构造调用未发现参数不匹配；含动态 **payload 的
  路径另依赖现有原生预留、状态与 metadata ABI 回归。
- 对象方法扫描另产生 128 处调用候选。唯一的未知关键字告警为
  `_dsv41_canonical`，已确认由插件 canonical_track 消费，不传入 core。
- 删除记录产生 17 个属性名候选：2 个是已删除的旧 Decode backend 字段；其余
  为当前 Req 属性、搬迁/继承接口、配置 bag 字段或不同 owner 的同名方法。
- 手动检查了 IndexedReqList 的 Req / DecodeRequest wrapper 边界、native
  pop_transferred 的调用入口、注册消息/aux stride 和 GPU/receiver 退休接口。
  PP outcome 路径属于当前插件启动时明确拒绝的配置，不能据此扩展支持范围。

## 发现与修改

没有发现新的、类似 inflight_middle_chunks 的必现启动 AttributeError。
本轮确认并修复以下遗留访问，不将它们都归因为本次 merge 新引入的回归：

1. `dsv41_remote.validate()` 仍 getattr 已删除的
   `ModelRunner.decode_attn_backend` / `decode_attn_backend_group`。当前 core
   只发布 attn_backend；移除旧分支，保留原生 backend 类型、mega/TRTLLM 与 sparse
   eager 接口验证。测试夹具也不再补出这两个旧字段。
2. `debug_queue` 读不存在的 Req.extend_range / handoff_held，并默认为 None/False，
   会误导排查。现在由 prefix_len / extend_end / extend_len 构造区间；未准备
   extend 时仍为 None。held 来自原生 PD_HELD 列表及插件真实 held owner 列表。
   诊断不修改请求、队列或 KV 所有权。
3. 原预检过于集中于 hook 签名和少量传输字段。扩大为 145 项 core 契约：
   Req 构造状态、KV 行/长度/保护区、批次字段、DecodeRequest 的发布状态、
   传输注册信息和 Hybrid 状态迭代方法均在启动前校验。
   其中显式校验 71 个 dataclass 字段、16 个 KVArgs 注解和 54 个初始化状态名。
   KVArgs 本身不是 dataclass，单独校验其注解，未放宽已有任务类型检查。
   缺字段时源码与已加载模块检查均明确拒绝，不添加旧版本兼容默认值。

## 验证

Ubuntu / 4060 主机隔离源码快照，CPU 单元与原生 ABI 测试：

| 验证 | 结果 |
|---|---:|
| 完整 Remote/P4/预留/TP2 主套件 | 378 通过，194.470 s |
| native contracts | 23 通过，19.450 s |
| native ABI | 6 通过，0.511 s |
| 定向诊断、源码字段缺失、已加载模块注解缺失 | 3 通过，属于主套件子集 |
| hooks / source interfaces | 51 / 145 通过 |
| Ruff 定向检查、git diff --check | 通过 |

新增诊断用例覆盖未 held、native held、plugin held 及未准备 extend。独立负对照
仅恢复旧 handoff_control.py，三个 owner 子场景都复现错误的空区间；修复后通过。
预检新增负例删除 pending_bootstrap、kv_committed_len、pending_metadata、
dst_aux_item_lens、state_item_lens；源码检查均拒绝。已加载模块缺 KVArgs 注解
也被拒绝，未将检查限定为某个源码目录。

共 **407 项**通过，不重复计算 3 项定向子集。机器结果和源码 SHA-256：
[upstream-interface-audit.json](upstream-interface-audit.json)。
原始日志：`/private/tmp/remote-upstream-audit/results.tar.gz`。

## 验证边界

完整 CLI 预检没有通过：该隔离快照通过 PYTHONPATH 加载插件，没有安装插件的
entry-point 元数据。进一步绕开安装元数据检查、直接验证真实 loaded core 时，
原生 deepseek_v4 → deepseek_v4_mhc → mhc_mega 导入报缺少 deep_gemm。
因此没有将真实模型模块导入、完整 runtime preflight/hook 注册计为通过，也没有
用伪造 deep_gemm 模块把它变为绿灯。共享 venv 未修改。

现有 ABI/生命周期测试与源码扫描不能代替完整模型服务验证。没有 H20 内网访问，
本轮未运行 H20 权重起服、SM90 FlashMLA/DeepGEMM、跨机 RDMA 或性能测试。
Python 的动态 getattr、payload 和配置组合也不能仅靠静态扫描获得穷尽证明。
