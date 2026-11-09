# RFC #45872: [RFC]: A registry + CI ratchet so tuned Triton kernel configs cannot silently rot

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#45872](https://github.com/vllm-project/vllm/issues/45872) |
| 作者 | access2rohit |
| 创建日期 | 2026-06-16 |
| 状态 | ⚪ closed(关闭于 2026-10-19) |
| labels | stale |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Attention / Kernel |

## 一句话结论
提议为可调 Triton kernel 建统一注册表加 CI 棘轮检查,让调优配置的静默失效(如文件名笔误)从结构上不可能再次发生。

## 提案动机
vLLM 为少数计算密集、在热路径上的 kernel(MoE expert GEMM、block-FP8/INT8 GEMM、Mamba selective_state_update decode 步)随包发布离线调优的 Triton 配置(按 batch size M 键控的 JSON),运行时加载最优匹配,未命中则回退固定启发式。两个结构性缺口使其脆弱:一是没有统一的「使 kernel 可调」机制,4 个可调 kernel 各自手写文件名拼接+加载+回退,新 kernel 很容易上线时就没有调优路径而无人察觉;二是静默漂移有 14 个月的实证:block-INT8 加载器拼文件名时多了一个空格(`block_shape=[128, 128]` vs 发布的 `[128,128]`),导致 39 个配置 0 命中,自 #16366(2025-04)起一直静默回退启发式,无测试、无警告、无信号,所有 block-INT8 模型悄悄损失性能一年多。

## 提出的方案
- 注册表(机制):新增 `vllm/utils/tuned_config_registry.py`,含 frozen `TunableKernelSpec(name, applicability, key_fn, packaged_dir)`、`TUNABLE_KERNEL_REGISTRY`、注册/查询接口与通用前端 `get_tuned_config_for(name, M, **shape)`;提供 `tunable_triton_kernel(...)` 装饰器,import 时完成文件名构建+加载+回退并注册 spec;4 个现有 kernel 由 seed module 注册,key_fn 精确复现各自文件名。
- CI 棘轮(执行):`tests/test_tunable_kernels_ratchet.py` 用静态 AST 遍历(不导入模块)枚举所有 `@triton.jit` kernel;每个 kernel 必须注册为 tunable,或在 `exempt_kernels.py` 附一行理由豁免;基线一次性 seed 保证当天 GREEN,只对新增 kernel 报 actionable 错误;并对每个注册 kernel 做文件名 round-trip 检查,从结构上杜绝 INT8 类 bug;拒绝空/「TODO」豁免理由。
- 理由化 allowlist 是去强制手段:约 88% 的 kernel 是 memory-bound(rmsnorm、rope、activation 等)不应调优;没有 allowlist 该测试会 flag 当前树上 324/328 个 kernel。CI 只说「调优或用一行被评审的理由说明为何不调」。
- 非目标:自动判断哪些 kernel 计算密集且在热路径——这保持为显式的人工评审判断。
- 参考实现(注册表+装饰器+seed+发现+allowlist+棘轮+测试)已构建并通过;现有配置文件与 `VLLM_TUNED_CONFIG_FOLDER` 覆盖不受影响。

## 优化点与收益
正文未给出加速类量化数字;给出的量化事实均属问题规模:0 of 39 个已发布 block-INT8 配置在约 14 个月内命中;~88% kernel 为 memory-bound;无 allowlist 时 CI 会 flag 324 of 328 个 kernel。收益是防止该类 bug 静默复发,注册表兼作所有性能关键 kernel 调优状态的活文档,并为未来 tuning service 提供机器可读清单。

## 现状与落地
已关闭(2026-10-19),labels 含 stale。正文称参考实现在作者 fork 分支 `rfc/tunable-kernels-A3-B3` 构建并通过、愿意开 draft PR;是否合入 main 需查关联 PR。

## 关联
- #16366(block-INT8 启发式回退来源)
- 参考实现分支:https://github.com/access2rohit/vllm/tree/rfc/tunable-kernels-A3-B3

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
