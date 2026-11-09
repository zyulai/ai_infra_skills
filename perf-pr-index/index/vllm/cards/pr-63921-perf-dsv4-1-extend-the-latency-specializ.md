# PR #63921: [Perf][DSv4.1] Extend the latency_specialized decode on NVIDIA

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yaoyaoding |
| 合并 | 2026-11-04 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready, ci/build, deepseek, nvidia, DSv4.1, agents |
| Diff | diffstat 缺失(pack 未记录);diff 极大,pack 保留 4 个测试/CI 文件,38 个文件因篇幅省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/63921 |

## 一句话结论

把 DSv4.1 的 latency_specialized 专用 decode 路径扩展到更大 batch:GB300 上 decode step 时间降 5.9-17.0%,AgentX P90 交互性提升 4-26%。

## 动机

PR #61868 引入的 `latency_specialized` decode 此前只覆盖小 batch:超过约 8 行后 routed experts 回退 stock FlashInfer 链,MXFP8 投影、indexer、all-gather 等也走通用路径;indexer 的 dense 层每个 rank 对每行重复打分,top-k 依赖 FP32 upcast + cooperative_topk(慢est rank 上 182 µs/step);vLLM custom all-gather 的 Lamport kernel 不等待本 rank 的 multicast copy,存在跨调用字损坏隐患。扩展目标是让 batch 1-32(每请求 6 行,即 6-192 行)都留在低延迟路径并提高步内重叠。

## 优化点

一组内核与调度改动(正文逐项给出当时实测):(1) routed experts 低 batch 路径扩到 64/192 行——fused front(路由+量化+置换)、routing-record prefix、router padding mask,GEMM2 按 token tile 等待 GEMM1 就绪词;(2) MXFP8 attention 投影固定 CuTe DSL tactic(>8 行)与 low-latency 路径扩到 48 行;(3) CuTe DSL `decode_topk` 直接吃 BF16 logits(token 模式与 block 模式,1M 宽行用 16 CTA),替换 upcast+cooperative_topk;(4) CuTe DSL sparse MLA(每 token 一个 cluster,每 128 列一个 CTA,dsmem 合并)替换 TRTLLM-gen decode,支持至 8 请求;(5) dense indexer 从 2 请求起按行 row-shard 跨 TP,Engram WKV 提前跑,candidate expansion 重叠;(6) 64 行起 two-shot all-reduce + mHC(改编自 FlashInfer MNNVL);(7) executor 自带 CuTe DSL `LamportAllGather`(等本 rank 拷贝)用于 indexer 行与 logits;(8) 复制 embedding 表直入 hyper-connection 流、indexer metadata 分支化构建、Q/KV norm 与 fused `wo_a` 提前触发依赖等。

## 关键改动

```text
vllm/models/deepseek_v41/nvidia/specialized_decode/*: 路由专家、MXFP8 GEMM、
  decode_topk_cute、sparse MLA CuTe、metadata graphs、runtime(文件在省略 diff 中)
vllm/models/deepseek_v41/nvidia/ops/cute_dsl.py: LamportAllGather
.buildkite/test_areas/models_basic.yaml: nightly 4×B200 DeepSeek-V4.1 步纳入新测试
tests/.../specialized_decode/test_{decode_topk,sparse_mla_cute,all_gather}.py: 新增
tests/.../test_kernels.py: 覆盖扩展到 192 行与 padding/graph replay
```

关键机制摘录(test_all_gather.py 语义,Lamport all-gather 的容量与重放校验):

```python
TP = 4
CAPACITY = 4 << 20  # bytes a rank, as the executor builds it
EAGER_PACKS = (1, 37, 389, 128, 5000, 3, CAPACITY // 16, 2, 777, 1)
# CUDA graphs of three gathers of different sizes, replayed in a seeded
# random order; device-side mismatch count must stay 0
```

## 性能声明

全部为 DeepSeek-V4.1-Flash,单 GB300 tray(TP4),DSpark(5 draft tokens,合成 acceptance 3.51),Model Runner V2,decode FULL CUDA graphs:

- decode step(batch 1-32,ABBA 3 组 × 12 次测量):比 main 快 5.9-17.0%;
- AgentX(393-trace 语料,3600s/并发,闭环):P90 interactivity 并发 1/4/8/16/32 为 817.7→852.0(+4.2%)、666.3→808.7(+21.4%)、539.6→678.7(+25.8%)、431.1→536.2(+24.4%)、225.0→258.2(+14.8%) tok/s/user,总吞吐 +1.4~2.3%,TPOT p50 同步下降(如并发 32:2.210→1.640 ms);
- 逐项改动收益(Batch 8 定 pin tactic -7.3%、低 batch experts batch 2 -8.6% 等,见正文表,各自单独测、不可相加);
- 代价:复制 embedding 表 1.32 GB/GPU + all-gather mailbox 48 MiB,KV cache token 总量 -0.9%;启动时间 +最多 10 s。

## 验证与回归风险

精度:GSM8K strict 0.9735 vs main 0.9742(真实 verification),GPQA-Diamond 0.791, AIME25 0.985,RULER NIAH 九格(32K/128K/256K × 3 任务,并发 8/32)全部 40/40;`VLLM_GPU_SYNC_CHECK=error` 下 GSM8K 0.9719 无同步报错。测试:kernel 299 passed(routed experts 对 FlashInfer 192 行内 bitwise,500 次 graph replay 交替输入)、decode top-k 77 passed、sparse MLA 33 passed、AllReduceMHC 61 passed、compute-sanitizer memcheck/synccheck 0 错误(199 tests);all-gather 压力测试 6,000,000 次 gather 0 bad words(对照 custom all-gather 392 stale words,证实其缺陷)。新测试进入 nightly 4×B200 步(部分 GB300 专属用例在彼处 skip)。风险:依赖 #63914 的 FlashInfer 0.7.2.post1 pin(MoE finalize 求和顺序差异需放宽一个测试的 ulp 判定);阈值型分派(row 数、WIDE_LOW_LATENCY_MAX_ROWS)在非 GB300 形状上的表现未逐一验证;仅覆盖 NVIDIA SM100 家族。

## 关联

- 前序:PR #61868(latency_specialized decode 原始引入)、#63914(FlashInfer pin 保持,已并入本分支)
- 相邻:#62633(swap-AB CuTe decode attention)、#62172 / #57109(cooperative_topk BF16、FlashInfer gvr_2 indexer top-k)、#62635(finalize 与 shared-expert add 融合)
- 同区域:本索引 DSv4.1 系列(如 PR #56935 FlashMLA mega attention、PR #59327 Engram lookups、PR #56441 KV-only context insertion)

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
