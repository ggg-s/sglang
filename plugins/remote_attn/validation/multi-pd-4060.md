# 多 P × 多 D：本地、TP2 与4060验证

日期：2026-10-31。基线 `0caedd006e`；生产改动仅在插件，SMG/SGL 核心无新增修改。
[启动与协议说明](../MULTI-PD-ZH.md)，[原始结果及逐文件 SHA256](multi-pd-4060-results.json)。

## 改动

- 新 `pd_mesh.py` 负责实例/启动代次/请求隔离、能力验证与本地 journal 的后台 TCP 复制。
  用三个 native function hooks 复用 receiver metadata、TransferInfo 解析、native replica
  resolution；metadata 原有十帧逐字保持，KV/state 仍走原生 PD/Mooncake。
- P 不从全局目录猜 D，D 不从全局目录猜 P：每条请求由原生 bootstrap 的真实 mapped
  rank 交换能力。每行 target 保留实际 P endpoint，同一 D 可混合来自不同 P 的请求。
- 网络线程不调用 scheduler、allocator 或 distributed collective。所有 GPU、native IO、
  cache lease、请求取消/retraction 和 TP 退休判断沿用既有协议；断线不构成完成证明。
- 实例的 graph/overlap/缓存/optimistic/chunk 配置不覆盖。网络设置进入已有 TP 启动共识；
  之后只新增一次实例启动代次广播，稳定 Remote、Local 和 middle→final 不增加同步。
- P 接管证书保留 capability 的身份指纹，排除 TCP/锁句柄；延迟 peer replacement 可在
  原有撤销流程提交后打开新通道。各 TP rank 的 sparse-ready response barrier 改为读取
  实际 source 所在的实例/peer/attempt 路径。

## 本地

**165 项完整插件测试通过，其中 59 项既有 DSV4.1 测试。** 新增五项覆盖：

1. 真实 TCP 的 2P × 2D，四个独立目录，同一 D 接收两个 P；取消一条请求不传播到其他请求，
   另一来源继续发布资源完成状态。
2. 断线后的同 attempt 重连与快照重放；丢连接不产生 `done`，后续合法完成事实仍可传达。
3. P 无法伪造 D retirement、目录穿越拒绝、同 room 新 attempt 的目录和 token 不复用、
   session/endpoint 不匹配不接管；接管指纹不序列化 TCP handle，能力变化仍能检测。
4. 加载实际 native `TransferInfo.from_zmq` 解析追加帧：原十帧和 KV indices 保持，非法
   能力不杀 bootstrap 工作线程、不能启用插件接管。
5. 相同 instance ID 的重复启动被锁拒绝；关闭后重启使用新 epoch，旧 proof 目录不复用。

真实 Gloo TP2 扩展：网络开关、实例标识、绑定地址不一致时双方在额外广播前拒绝；
一致配置安装网络服务后，两 rank epoch 相同。两 rank 使用独立 D capabilities 和本地
镜像，P 的 HTTP response snapshot 等全部 sparse-ready 后仅发布一次。原有慢 rank、
GPU pending、取消/retraction、资格重确认、事件驱动退休及稳态零新增 all_reduce 回归通过。

源码预检 **44 hooks / 104 core interfaces**。isort、ruff、diff 检查通过。

## RTX4060

用户授权的 ubuntu 工作机 RTX4060 Laptop 8GB，PyTorch 2.13.0+cu130；隔离目录
`/tmp/pd-mesh-20261031`，复用 venv，不安装依赖或修改部署。

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
PYTHONPATH=plugins/remote_attn PYTHONDONTWRITEBYTECODE=1 \
/workspace/remote-attn-cuda-20261028/venv/bin/python \
  test/manual/remote_attn/dsv41_segmented_graph_probe.py \
  --source-root . --mesh --output mesh-graphs.json
# 再跑 --fused-projection --bf16-kv-source。
```

- 两个 source hubs 绑定 `127.0.0.3/.4`，两个 D 控制实例绑定 `.2`，P 控制实例绑定 `.1`。
  四个 journal 目录独立，通过 native metadata 扩展协商后实际 TCP 同步，不依赖共享文件。
- 两个 P 存储不同的 native KV bytes，同一 batch 逐行交替来源；独立 reference 按来源
  读取不同 cache。默认与融合/BF16 两组均通过：每组 **18 次 Remote graph replay、42 次
  D0 sparse RPC、8 次 Local full replay**，最大绝对误差 **0**。
- 两组均保留 batch 1/2/4 三套 Remote banks（各8段）和四种 native candidate variants，
  捕获零 RPC、QKV 重放零 Python 投影调用，live SWA/carry 检查通过。
- D0 targets 关闭后，D1 仍分别向两个 P 发起实际 sparse RPC 并与 reference 一致；全部
  D0/D1 `remote-done` 正确回传对应 P 镜像，未关闭其他请求的连接。
- 旧单 pair 图路径、共享 Qwen 图探针、既有 DSV progressive CUDA 探针通过；后者覆盖
  cache/source pins、GPU reader 取消保护、native byte layouts、pinned H2D、末页/carry
  保护及逐资源退休，稳定 Remote/Local 新增投票均为0。

各组 **40 / 40 / 37 / 29 / 5** 份源码 SHA256 与本地最终实现一致。

## 范围

实际执行 native graph primitives/backend、core QKV/state 方法、native PD metadata
编解码、replica resolution、GPU tensor 运算、native-byte sparse reader、RemoteHub RPC
和网络 journal。小型模型的 linear/norm/store kernels、metadata/indexer/carry planner、
SMG 选源、ZMQ 递送及 native admission 仍是夹具边界；未启动四个完整模型服务。

本机与跨机使用同一可寻址、无共享盘协议；本次不同 IP 在同一物理4060工作机上验证。
没有双物理 GPU 主机、多 GPU NCCL、完整 checkpoint token/logprob、故障自动恢复或生产
TTFT/ITL/吞吐证据；上线仍需原生 SMG 与目标模型/网络的端到端验收。
