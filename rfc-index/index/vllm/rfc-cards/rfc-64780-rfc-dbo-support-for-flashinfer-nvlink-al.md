# RFC #64780: [RFC]: DBO support for FlashInfer NVLink all-to-all backends (single-node H200/B200 first)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64780](https://github.com/vllm-project/vllm/issues/64780) |
| 作者 | algebra-MCX |
| 创建日期 | 2026-10-30 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | MoE |

## 一句话结论
为 FlashInfer NVLink all-to-all 后端补齐 DBO 支持(先做单节点 H200/B200),让这些 EP 用户也能把 all-to-all 与另一 microbatch 的计算重叠。

## 提案动机
`flashinfer_nvlink_one_sided` / `flashinfer_nvlink_two_sided` 是 vLLM 的 NVLink-native EP 路径,但今天 `--enable-dbo` 无法与之组合。在 commit `dc4dd5a2` 处的当前行为:CUDA 上 `use_ubatching` 只允许 `deepep_low_latency`、`deepep_high_throughput` 与 `nixl_ep`;两个 prepare/finalize 类保持默认 `supports_async() == supports_dbo() == False`,DBO 下 `_prepare`/`_finalize` 会 assert;per-step 状态(one-sided 把 `runtime_max_tokens_per_rank` 存在实例上,two-sided manager 持有单一 workspace)无法安全承载两个在途 microbatch——它们会共享缓冲。这些后端的用户因此失去 DeepEP 与 NIXL EP 用户已有的「all-to-all 与另一 microbatch 计算重叠」能力。作者基于对 FlashInfer 0.7.2 的静态阅读(未上机)论证首片不需要 GB200:`MnnvlMemory` 在无 fabric cluster 的设备上从 FABRIC handles 回退到 POSIX 文件描述符,单 NVSwitch 节点可用;默认 `backend="trtllm"`(legacy)内核无 Python 架构 gate,MNNVL comm JIT 模块面向 SM 9.x/10.x,只有 opt-in 的 `backend="cake"` 要求 SM 10.0/10.3;多节点才需要 FABRIC handles。

## 提出的方案
第一片范围:单节点、DP/EP over NVLink、CUDA、eager DBO(prefill 与 decode);full CUDA graph DBO、multi-node MNNVL 与 `backend="cake"` 留待后续。
1. 执行模型:所用 FlashInfer API 不暴露 send/receive hook 对,故沿用 `naive_dp_ep._comm_region` 的既有同步模式:DBO 下 yield 后在 comm stream 上运行 dispatch/combine,再切回;`supports_dbo()` 返回 True,`supports_async()` 保持 False 以使共享专家布局不变。
2. per-microbatch 状态:按 `dbo_current_ubatch_id()` 索引 per-step 状态(`runtime_max_tokens_per_rank`、dispatch handles/metadata);每个 microbatch 分配一个 FlashInfer workspace,使两个在途 collective 绝不共享缓冲或完成标志。
3. SM 预算:comm-stream kernel 限制在现有 DBO comm-SM 预算内,并确认 spinning receive 不会饿死 peer microbatch 的 sender(死锁检查)。
4. Config:仅当 (1)–(3) 通过后,才把两个 backend 加入 CUDA DBO allow-list。
完成标准:DBO off 时两个后端的代码路径与输出不变;单测覆盖状态索引、workspace 归属、`supports_dbo`、allow-list;2× 与 8× H200(有条件则加 B200)、DP/EP、eager DBO 下,一个 MoE 模型(如 DeepSeek-V2-Lite 与 DeepSeek-V3 级模型)DBO 开关输出/logprobs 一致、uneven microbatches 反复运行无 hang;`vllm bench serve` 吞吐/TPOT 对比 + trace 显示 all-to-all 与 peer microbatch 计算重叠。Out of scope:修改 FlashInfer 内核、FlashInfer 内 async hooks、SBO(#64776)。另注:`tests/distributed/test_mnnvl_alltoall.py` 有 4 个测试在无 `SYS_PTRACE` CapEff 的主机上会 skip,验证须在能执行它们的主机跑且 skip 不算通过。

## 优化点与收益
正文未给出量化数字——作者明言「No implementation, GPU run or performance result exists yet」。收益是定性的:把 DeepEP/NIXL EP 用户已有的 all-to-all 与 microbatch 计算重叠(DBO)能力带给 FlashInfer NVLink 后端用户。

## 现状与落地
仍 open,处于向维护者征集反馈阶段(feedback period 两周),列出 3 个待答问题:同步 comm-stream 模式(如 `naive_dp_ep`)是否可接受、per-microbatch workspace 应放在 all2all manager 还是 prepare/finalize 对象、单节点 H200/B200 认证对首片是否足够;无实现、无 GPU 运行、无性能结果;注明调查与起草使用了 AI 辅助。

## 关联
- #64776(SBO,另案)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
