# Flash-0731 C4/C128 状态、尾块及生命周期审阅

日期：2026-11-04。基于 d844c340c4。

## 发现并修复

native `_num_dsv4_physical_kv_pages` 按
`ceil((compressed_size + FULL_page_size) / compressed_page_size)` 分配 guard 页。
相同 FULL 容量下 C4 与 C128 压缩池的物理页数量不同（C1/C2 同样可能不同）。
原 `page_map` 按容量作为缓存 key，但拒绝同一请求建立第二种几何，因此逐层切换
压缩率会报 sparse page-map storage changed。之前每个模型比率单独建 fixture，
且 fixture 误用统一 guard 容量，未覆盖这个真实混合模型触发条件。

现在按 FULL table 身份、FULL page size 和 ratio 持有独立逆页表。每个 ratio 的
容量固定，容量漂移、FULL table 替换和 attempt/row 变化仍被拒绝。map 最多按
native 支持的压缩率增长，生命周期继续由同一个 Target 和退休证明管理；成功
Target.close 清理所有逆页表，没有新增 allocator free 或提前关闭 cache pin。
生产 fast attention 显式传 ratio；legacy reference 调用继续有固定容量检查。

## 审阅结论

- C4 compressed/indexer 状态页按 native SWA 页交接；C128 online 单行与 offline
  pending ring 的索引继续取 native planner，不从 global 页推算另一个状态索引。
  CPU 回归已用禁用 speculation 时真实 ring_size=1/128 校验整块边界无 pending，
  非整块返回对应 request row；状态注册 stride 与 request/page 两类所有权分离。
- C4/C128 只把 `< floor(prompt_len / ratio)` 的完整压缩行归 P；未完成块及接续归 D。
  接纳前必须完成所有 indexer/scales scatter 和 SWA/carry/metadata；canonical
  final 不再重复发送 D 已接管的状态。未改变最后 chunk 完整 forward 的事件边界。
- global 尾页的 data/scale 按有效行写入，非零区间起点同样遵守全局 token 边界；
  没有把尾页完整复制到 D 新完成的压缩块上。GPU 精确字节 oracle 校验 padding。
- overlap 的 forward proof、native outstanding/IO ledger、P reader ref、cache pin
  与 TP 退休仍是释放门槛；rank-local 完成不是释放证明。
- 未发现新的正常路径提前释放、重复释放或 KV 泄漏路径。unknown DMA / TP
  quarantine 继续保留未证实安全的 KV、metadata、ring 与 RPC 对象；这是故障隔离
  行为，不是本次新增自动恢复或故障后全部资源回收的承诺。

## 回归证据

TestRemoteAttn **273 项通过**，含真实 TP2 Gloo、native checkpoint/插入、SWA
窗口/预算、metadata 提前接纳、正常与取消退休、unknown DMA、TP quarantine。
新增混合 ratio 测试用 native 页数函数构造 C1/C2/C4/C128 不同容量，交替访问后
确认映射复用、未完成 C128 块保留 D、同 ratio 容量漂移与请求身份复用被拒绝。
49 hooks / 125 interfaces、Ruff E9/F63/F7/F82、git diff --check 通过。

[4060 Flash 验证](flash0731-review-4060.json)：

- 同一请求在 C4/C128 间交替，native guard 容量为 265/389，映射保持正确。
- C4/C128 × TCP tensor / native Mooncake GPU 四种实际 RPC 均通过；D 未就绪前缀
  刻意清零，输出匹配独立 reader/partial。C128 回传 520 行历史，验证超 top-k。
- 尾页覆盖 ratio±1、256+ratio±1 等边界，以及 start_page=0/1；data、scale、
  padding 逐字节匹配。有效行之外不写，未完成块留在 D。
- 真实 GPU reader 在途时提前 close 被拒绝，tombstone 等完成后才关闭 source pin。
  Target.close 后逆页表为空，正常路径 pins 归零，无 quarantined slot 残留，RPC
  teardown 后原生注册仍可传输。

[4060 overlap/cache 验证](flash0731-review-overlap-4060.json)：normal/cancel/revoke
均通过、无释放后写入。这里 cache/IO callback 为 ABI fixture；真实 hybrid cache
与 TP2 quarantine 由 CPU 套件覆盖。两份 JSON 的源码 SHA256 与最终源码一致。

4060 使用 Mooncake TCP，未加载 Flash 整模权重、未测 H20 RDMA/TP4 GPU、文本精度
或整模性能。GPU 验证为真实 kernel/event/RPC 加 source-loaded native ABI fixture。
