# PR #61868: [Perf][DSv4.1] DeepSeek-V4.1-Flash specialized decode on GB300

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yaoyaoding |
| 合并 | 2026-10-21 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready, ci/build, deepseek, nvidia, DSv4.1, agents |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);另有 38 个文件因超长从 diff 中省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/61868 |

## 一句话结论

为 GB300 上 DeepSeek-V4.1-Flash 小 batch agentic decode 新增专用执行路径(默认关闭),DSpark 推测步 batch 1 快 18.3%,AgentX P90 交互性提升 16-24%。

## 动机

AgentX 类 agentic 负载发送长多轮对话(中位 350K token 上下文,几乎全被 prefix cache 命中),每台 server 同时只 decode 1-64 个请求,用户感知的是每输出 token 的时延。batch 1 时 stock vLLM 的一个推测步约 5.4 ms 且无单一热点:一步约 1600 个 kernel,多为几微秒的小 kernel;GEMM 只见 1-48 行却用大 batch 调优的 kernel;80 个 sublayer 边界的 all-reduce + mHC 仅在 16 行内融合;host loop 忙 4.95/5.37 ms,其中约 2.2 ms 在构建 attention metadata——纯 GPU 优化超过 0.4 ms 就会暴露 host。

## 优化点

模型持有 decode executor(FULL-graph 纯 decode、≤384 行即 64 请求 x 6 token 才接管;prefill、mixed 与更大 batch 走 stock forward),按 `--additional-config '{"decode_backend": "latency_specialized"}'` 显式开启。手段四类:(1) 减 kernel/去 gap——sublayer 边界融合 Lamport all-reduce + mHC 至 384 行(mHC 系数与注意力输入 MXFP8 量化放 side stream,`wo_b` GEMM epilogue 直写 mailbox,MoE finalize 延迟到融合 kernel);(2) 低 batch 专用 kernel——routed experts 用 CuTe DSL 的融合 front(routing+MXFP8 量化+排列)加两个 tcgen05 GEMM,`fused_wqa_wkv`/`wq_b`/`wo_b` 走 FlashInfer 低延迟 CuTe DSL MXFP8 GEMM 并按 shape 钉 tactic,skinny 投影用 vLLM BF16 split-K,LM head 用 CuTe DSL BF16 GEMM 加自定义 all-gather;(3) indexer/attention——同 index source 的层复用一张 sparse-index 表,sparse-MLA 用一次清零的持久 KV counter,candidate 用 `cooperative_topk` 只扫有效列,indexer query/表构建上 side stream,稠密 indexer 行按 TP rank 分片再 all-gather(每步 kernel 数 1578→1486);(4) host——attention metadata 构建录进 CUDA graph 回放,Engram KV 投影链在 prefetch stream 上提前执行(主 stream 占用 70→6 µs)。

## 关键改动

```text
vllm/models/deepseek_v41/nvidia/specialized_decode/(在省略 diff 中): runtime/ 执行器与 per-step gate、boundary/ 融合 all-reduce+mHC、attention/ 低 batch 投影与行分片 indexer、moe/ 延迟 finalize 与 CuTe DSL routed-expert、engram/ prefetch、logits/ 低 batch LM head
tests/models/deepseek_v41/nvidia/specialized_decode/test_kernels.py: 新增,每个 kernel 与其替代的 stock op 逐位/容差对比(eager + graph replay)
.buildkite/test_areas/models_basic.yaml: 新增 nightly 4xB200 specialized decode step(B200 跑 kernel/numerics,GB300-only 的 layers test 跳过)
```

关键设计:per-step gate 只读所有 TP rank 共享的 host 状态(行数、请求计数、graph mode),稳态不回读 GPU,避免 rank 分叉导致 collective hang;rank 间资格用 MIN all-reduce 合并,不满足要求时启动即报错并列出全部缺失项。

## 性能声明

DeepSeek-V4.1-Flash,单 GB300 tray(TP4),DSpark 5 draft tokens,MRV2 + FULL CUDA graph decode;switch-off 臂为同期 upstream main。Decode 步(含 draft,client 侧计时,同 tray ABBA):batch 1 5,355 → 4,375 µs(-18.3%),batch 8 -13.7%,batch 32 -11.3%。AgentX trace(每并发 3600 s,upstream `2c74fd1` vs 本 PR):P90 interactivity conc 1-64 提升 +15.7% 至 +23.8%,单 GPU 吞吐 +1.0% 至 +6.2%,TPOT p50 全线下降(如 conc1 1.46→1.14 ms),TTFT 不变(跨 tray 差异经同 tray A/B/A 排除)。精度:GSM8K strict-match c1/c8/c32 0.974/0.970/0.975(off:0.973/0.972/0.973);GPQA-Diamond 0.796→0.788(配对差 -0.79±0.56 点);AIME25 0.994→0.990;RULER NIAH 32K/128K/256K 全部 1.0;DSpark acceptance length 不变(约 3.9)。

## 验证与回归风险

kernel 测试 90 passed(每个 kernel 对比其替代的 stock op,含 graph replay);TP4 fused collectives 对 1,000 次 graph replay 与未融合参考逐位一致;100,000 次 replay 竞态压测通过;4xGB300 上五层随机权重模型 switch on/off 逐层激活比对;compute-sanitizer(memcheck/synccheck/racecheck)覆盖本 PR kernel,非本 PR 的 FlashInfer/CuTe DSL kernel 已知有报告并注明。CI 新增 nightly 4xB200 step(optional,无 GB300 agent)。范围限制严格:仅 GB300(SM103)、MRV2、TP4、无 PP/DP/EP/CP/micro-batching、DSpark 无 adaptive verification、`FLASHINFER_MLA_SPARSE_DSV41` + fp8 KV + MXFP4 indexer、仅 decode FULL graph、无 torch.compile/sleep mode/LoRA/KV transfer/batch invariance;三个路径依赖 FlashInfer 0.7.2 内部接口,其他版本回退 stock 路径并告警。开关关闭时模型行为与 upstream main 一致(有专门的 switch-off 计时验证,差异 ±1.8% 内)。

## 关联

同区域:PR #61867(同一 `decode_backend` 开关的先行集成)、#61869(DSpark draft model 放入同一 executor 的 stacked follow-up)、RFC #59665(fast-track 流程)、#57448(V4.1 tracking)、#59725(Cake kernels);DSpark 见 pr-46995;MRV2 见 pr-52228;同模型 FFN/attention 优化见 pr-56935(V4.1 FlashMLA mega attention)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
