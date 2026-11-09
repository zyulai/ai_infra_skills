# RFC #51513: [RFC]: Unify functional P2P gating — one veto-only verdict for all P2P consumers (NCCL, custom allreduce, symm-mem)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51513](https://github.com/vllm-project/vllm/issues/51513) |
| 作者 | ldkhang1201 |
| 创建日期 | 2026-08-08 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Parallelism / Communication |

## 一句话结论
把 write-and-verify 的 `gpu_p2p_access_check` 定为唯一 P2P 功能裁决(veto-only),接入 NCCL/custom allreduce/symm-mem,消除坏 fabric 上的挂起与静默出错。

## 提案动机
在驱动报告 GPU P2P 可用、但 fabric 静默丢弃 peer 写的平台(PCIe ACS + translated IOMMU、BIOS 配置不当、部分 VM),vLLM TP=2 会在引擎启动时永久挂起。`cudaDeviceCanAccessPeer` 只是能力查询而非数据通路测试;PCIe peer 写是 posted 事务,被丢的写只出现在 host 内核日志(`AMD-Vi: IO_PAGE_FAULT`),GPU 侧毫无感知。作者在 2-socket AMD EPYC、8× PCIe GPU(无 NVLink)裸机上端到端复现了因果链:NCCL `ncclTopoCheckP2p` 的 AMD ≤2-GPU trust carve-out 恰好让 TP=2 走 P2P 挂起,而 TP≥4 走 SHM 正常——这正是「TP=4 没事」误导排障的原因。完整诊断需要 host dmesg、`NCCL_DEBUG=INFO`、py-spy 和约一天时间,而 vLLM 已拥有可秒级检测的 `gpu_p2p_access_check`,只是默认跳过、仅 custom allreduce 一个 consumer 使用,且其运行点晚于 NCCL 挂起的位置。

## 提出的方案
- 契约:把子进程 write-and-verify 检查 `gpu_p2p_access_check(src, tgt)`(JSON 缓存,已在 `all_reduce_utils.py` 实现)提为「P2P 在这两卡间是否功能可用」的唯一 canonical verdict,语义为 veto-only:`False` ⇒ 对所有 consumer 禁用 P2P(会丢 legacy cudaIpc 写的 fabric 也会丢 cuMem 写);`True` ⇒ 不强制任何启用(检查测的是 legacy cudaIpc,NCCL 2.28 与 torch symmetric memory 走 cuMem,通过并不认证那些路径)。
- 接入所有 consumer:CustomAllreduce 已接入;`SymmMemCommunicator` 增加同款 gate;MiniMax fused AR(#47544)改用共享 verdict;NCCL 由 vLLM 在首个 communicator 创建前于 worker 环境设置 `NCCL_P2P_DISABLE=1`(用户已设任何 `NCCL_P2P_*` 变量则绝不覆盖),并显著日志提示平台成因(ACS + translated IOMMU)与修复方法(BIOS 关 ACS 或 `iommu=pt`)。
- 成本:默认行为不变(检查仍默认跳过,信任驱动,依据 #21904);`VLLM_SKIP_P2P_CHECK=0` 时保护面从 custom allreduce 扩展到全部 consumer,#39249 的 async precompute 隐藏其启动时延;后续(单独 PR)可仅在风险拓扑(多卡、PCIe-only、无 NVLink)自动启用检查,NVLink 机器永远不付成本。

## 优化点与收益
正文未给出量化性能提升数字。价值在于:把这类故障的发现从「需 host dmesg + NCCL_DEBUG + py-spy、约一天」缩到秒级检测,并消除 TP=2 启动永久挂起与静默输出损坏两类后果。平台侧修复(`iommu=pt` 或 BIOS 关 ACS)已由作者实测消除了挂起与 fault。

## 现状与落地
仍 open。RFC 请求的是就「契约」达成一致而非直接落码;作者声明所有论断(dmesg fault 签名、nranks 2 vs 4/8 的 NCCL transport 翻转、`iommu=pt` 修复、各 consumer kill-switch 行为)均已在真实硬件复现验证,2026-08-09 做过重复工作检查(无既有 RFC 覆盖统一 P2P gating);处于等待维护者评审/讨论阶段。

## 关联
- #17676(vLLM hangs forever on waiting engine process to start)
- #2728(催生 `can_actually_p2p` 的原issue)
- #21904(将 P2P check 改为默认跳过的 PR)
- #47544(MiniMax fused AR,使用共享 verdict)
- #39249(async precompute)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
