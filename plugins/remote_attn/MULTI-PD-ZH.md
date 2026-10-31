# 原生 SMG + 多 P × 多 D progressive

支持同机或不同机的多个独立 P/D 实例，复用 SMG 选择的每请求 P/D 组合。每个实例内部
仍为相同 TP、PP=DP=CP=1；TP ranks 位于同一主机。支持现有 Qwen3-8B 和 DSV4.1 路径，
模型、KV layout、TP 和能力检查继续执行。单请求的前缀由一个 P 提供，不做多 P 分片。
上线前仍需在目标硬件运行完整模型与原生 SMG/P/D 验收；本次验证范围见文末。

## 启动方式

在每个 P/D 原生启动命令的环境块中增加：

```bash
export HANDOFF_PD_NETWORK=1
export HANDOFF_INSTANCE_ID=P0  # P0/P1/D0/D1 各不相同；同一实例 TP ranks 相同
export HANDOFF_PD_DIR=/workspace/remote-attn/journal
export HANDOFF_CTL_DIR=/workspace/remote-attn/control
# 继续保留原来的插件加载、HANDOFF_PRE_LAYERWISE=remote、staging 等配置。
```

`HANDOFF_PD_DIR` 现在是**各主机本地目录**，无需共享盘，P 与 D 的路径可以不同。
同机实例可使用相同目录基址：插件自动隔离为 `instances/<instance>/<boot-epoch>/...`。
同一实例各 TP rank 的目录基址必须一致。控制文件位于
`$HANDOFF_CTL_DIR/$HANDOFF_INSTANCE_ID/ready.<rank>.json`，rank 0 持有实例锁，拒绝同一
控制根目录下重复启动相同 instance ID。重启生成新的 TP 共享启动代次。

控制通道和 Remote attention 默认绑定并通告原生 KV manager 的 `local_ip`。需要指定
网卡或区分绑定地址与通告地址时使用：

```bash
export HANDOFF_NETWORK_BIND_HOST=10.0.0.11
export HANDOFF_NETWORK_ADVERTISE_HOST=10.0.0.11
```

跨机时这些地址须能从对端访问，不能通告 `127.0.0.1` 或 wildcard。控制通道使用 D 的
动态 TCP 端口，Remote attention 使用 P 的动态 TCP 端口；网络需允许实例间访问这些端口，
以及原生 PD/Mooncake 已需的端口。不改变原生 PD 传输地址或传输参数。

P/D 启动仍直接执行各自既有 `python -m sglang.launch_server ...` 命令。保留线上完整的
optimistic prefill、chunk、transfer batch、graph buckets、缓存和 overlap 配置。模型已有
限制仍适用，例如 DSV4.1 的 D radix cache 与 DSpark 没有因此放开。

原生 SMG 在独立终端启动，不加载插件环境。下面示例为两个 P、两个 D：

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.0.0.11:31016 35409 \
  --prefill http://10.0.0.12:31016 35409 \
  --decode http://10.0.0.21:31017 \
  --decode http://10.0.0.22:31017 \
  --host 0.0.0.0 --port 31000
```

同机部署使用该主机可达地址及各实例不同的 HTTP/bootstrap 端口。原有 SMG 路由策略、
健康检查和请求头保持原样；插件不固定 pair，也不替换 SMG 的负载均衡。
不开启 `HANDOFF_PD_NETWORK` 时，仍使用原有同机共享 journal 的单 pair 模式。

## 请求隔离与安全

1. 原生 D receiver 根据 SMG 携带的 P bootstrap 信息完成 rank 映射。插件在原生
   10-frame Mooncake 请求 metadata 末尾附加一个带版本的控制能力描述；原生前十帧
   和 KV/state 传输不变。只为真实 mapped rank 建通道，dummy rank 不接管。
2. 能力绑定 D instance/boot、native session、bootstrap room、attempt、rank、随机 token
   及原生选中的 P rank endpoint。P 通过该能力连接 D；D 将连接绑定到一个 P instance/boot。
   同 room 再次发送 native metadata 使用新 attempt 和 token，文件目录不会复用。
3. P/D 各保留本地 journal。后台 TCP 线程按版本复制状态快照，断线重连仍绑定同一
   attempt；重复快照幂等、旧版本拒绝。调度循环只读本地文件，写入只唤醒本地发送线程。
   通道不会在每个 token 额外轮询网络，也不调用 scheduler、allocator 或 TP collective。
4. P 只能发布 P 完成事实，不能伪造 D 的 `done/admitted/local-*`。路径、generation、
   方向与 frame 大小均验证；远端不能指定本机任意文件路径。无有效能力、未连上或容量
   达到上限且尚未接管时，保持原生传输；接管后缺失证明则等待/报错并保留资源。
5. 传输完成与 GPU 完成继续分开：P TE completion 不替代 D scatter event；D 仍需
   native metadata、GPU forward/sampling/readback 证明和本实例 TP 有序退休。网络断线、
   对端消失或控制通道超时，都不能单独授权释放缓存页或 KV。
6. 同一 D 的每行 target 保存自己的 P route；MHA 按 endpoint 分组 RPC，DSV 按行寻址。
   每请求保留自己的 prefix/cache lease、epoch、Local mask 和退休身份。取消一条请求
   只关闭其 reader/租约；源 P 仍等待该请求 GPU reader、native IO 与 TP 回收条件。
7. P overlap 的证书指纹包含不可变网络能力，不包含 TCP/锁对象。延迟的 peer replacement
   继续经过原有资格撤销/重确认；稳定 middle→final 不增加逐 chunk 接管 all_reduce。
   网络配置先进入已有 TP 启动共识，再进行一次启动代次广播；运行期退休协议不变。

同一 D 含多个 P 的 Remote 请求时继续使用现有 batch 1/2/4 分段图；全部相关租约退休后
恢复 native full graph/overlap。没有新增图切分点、逐 token TP 投票或 serving 配置覆盖。
后台控制通道的终止/清理只清理通信对象，不执行 KV/cache 释放。

## 验证

本地测试、Gloo TP2、4060 两 P/two D 控制通道、混合来源分段 CUDA graph、取消/重连
和既有回归见 [验证记录](validation/multi-pd-4060.md)。4060 使用单 GPU 的小型模型/规划
夹具和不同 loopback IP、独立实例目录；没有部署四个完整模型服务或验证两台物理 GPU 主机。
因此这里实现跨机可寻址和无共享盘协议，尚未证明跨物理主机的生产时延/带宽或整模正确性。
