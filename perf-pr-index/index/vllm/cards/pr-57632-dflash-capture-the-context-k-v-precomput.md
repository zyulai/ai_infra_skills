# PR #57632: [DFlash] Capture the context K/V precompute in the draft CUDA graph

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Juntian777 |
| 合并 | 2026-09-24 |
| 分类 | Speculative Decoding（tier: major） |
| Label | speculative-decoding, ready, deepseek, nvidia, mrv2, dflash, DSv4.1 |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾未见 "N file(s) dropped from diff" 标注，diff 段含 2 个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/57632 |

## 一句话结论

把 DFlash draft 的 context K/V precompute 纳入 FULL decode CUDA graph，4×GB200 / DSpark c=1 下 CPU/step 6.77→5.03 ms、TPOT/ITL 约 −15%。

## 动机

正文 Purpose：即使 query forward 以 FULL CUDA graph replay，DFlash/DSpark 的 draft step 仍会把 context K/V precompute（`precompute_and_store_context_kv`）放在 graph 之外 eager 执行，原因是每步的 context tokens 数量会变化。该 eager 部分带来额外的 host 开销。

## 优化点

把 context K/V precompute 提前捕获进 draft 的 FULL decode graph（在 query forward 之前）。每张 graph 为每个 request 存一次完整 verify 的 context rows；超出 batch 的 rows 用 `PAD_SLOT_ID` 填充使其不写 K/V；超出 graph rows 的 prefill context 在 replay 之前仍以 eager 方式存储。该 precompute 无 host sync，因此每个 DFlash drafter 都能捕获它；draft attention metadata build 保持 eager 不变。diff 中 `cudagraph.py` 的 `capture()` 新增 `precompute_context_kv` 回调参数，`create_forward_fn` 生成的 `forward()` 先调用 `precompute_context_kv(num_reqs)` 再执行原 `forward_fn`。

## 关键改动

```text
vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py: capture 新增 precompute_context_kv 回调，forward 内先做 context K/V 预计算再跑原 forward_fn
vllm/v1/worker/gpu/spec_decode/dflash/speculator.py: 移除 context_positions 上 "NOT captured by CUDA graphs" 的注释
```

```diff
+            def forward(cg_mode: CUDAGraphMode) -> None:
+                precompute_context_kv(num_reqs)
+                forward_fn(
+                    num_reqs,
+                    num_tokens,
+                    attn_metadata,
+                    slot_mappings,
+                    num_tokens_across_dp,
+                    cg_mode,
+                )
+
+            return forward
```

## 性能声明

测试条件：4× GB200、FlashInfer 0.6.18.post1、`VLLM_USE_V2_MODEL_RUNNER=1`；Model DeepSeek-V4.1-Flash（`dba1be0a40aa45a94ad051997016db3960a90277`）；DSpark 5 speculative tokens + full decode CUDA graphs；Topologies DEP4（MegaMoE + adaptive verification）/ TP4（auto MoE + fixed verification）。Serving benchmark：TP4、DSpark（5 spec tokens、synthetic acceptance length 5.5）、random dataset 128 in / 1024 out，Base=main、PR=`d8de1d3a2f`，CPU/step 为注入 per-step probe 测得的 host time。

| Concurrency | CPU/step | TPOT | ITL |
|---|---|---|---|
| 1 | 6.77 → 5.03 ms | 1.27 → 1.08 ms (−15.2%) | 6.95 → 5.88 ms (−15.3%) |
| 8 | 6.64 → 5.24 ms | 1.87 → 1.84 ms (−1.6%) | 9.96 → 9.90 ms (−0.6%) |
| 64 | 6.82 → 5.43 ms | 3.90 → 3.87 ms (−0.7%) | 19.97 → 20.01 ms (+0.2%) |

正文结论："The PR cuts host time per step by 1.4–1.7 ms. At concurrency 1 the step is host-bound, so this carries through to a ~15% lower TPOT and ITL; at higher concurrency the GPU dominates and latency is roughly unchanged."（perf claims 段标注 15% lower TPOT and ITL）

## 验证与回归风险

验证：Acceptance/GSM8K parity（全 GSM8K 1319 条 5-shot 贪心 completion，concurrency 32、1024-token 上限、strict extraction）对比 parent commit `c723a831a8` 与 `d8de1d3a2f`：DEP4 acceptance length 3.411→3.434 / GSM8K 1234→1228 (93.56%→93.10%)，TP4 3.471→3.463 / 1227→1219 (93.03%→92.42%)；正文指出 acceptance deltas 在 ±1% 内且两拓扑符号相反、GSM8K deltas 属 run-to-run noise（同配置两次 DEP4 parent 得 0.9219 与 0.9356），据此称捕获 precompute 不改变 drafter 的 proposals；`.venv/bin/python -m pytest tests/v1/attention/test_dspark_noncausal_sparse_mla.py -q`。可能影响路径：DFlash/DSpark drafter 的 FULL decode CUDA graph、context K/V 的 `PAD_SLOT_ID` 填充与超出 graph rows 的 eager prefill 路径；正文明确 draft attention metadata build 仍保持 eager、precompute 无 host sync。PR 未注明更多回归风险。

GSM8K server 的 common flags 为 `--reasoning-parser deepseek_v41 --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' --kv-cache-dtype fp8 --max-model-len 8192 --max-num-seqs 128`，DSpark 配 5 speculative tokens；DEP4 用 `--data-parallel-size 4 --enable-expert-parallel --kernel-config '{"moe_backend":"deep_gemm_mega_moe"}'` 且启用 adaptive verification，TP4 用 `--tensor-parallel-size 4`、auto MoE backend 且关闭 adaptive verification。Serving benchmark 的 serve 命令用 `FLASHINFER_MLA_SPARSE_DSV41`、拒绝采样为 synthetic、`synthetic_acceptance_length=5.5`，bench 为 random 128 in / 1024 out、`--ignore-eos`。正文明确 "These results establish parity with the parent; they do not claim an accuracy improvement"，并指出 per-position acceptance 曲线重合、acceptance deltas 在 ±1% 内且两拓扑符号相反。Validation 段还说明 acceptance length 的读取方式：DEP4 取自引擎 `vllm:spec_decode_*` counters 对整次运行的累计，TP4 则从引擎周期性 SpecDecoding 日志窗口在稳态聚合，每对比较两侧使用同一方法；并给出两次重复同一 DEP4 parent 配置得 0.9219 与 0.9356（18/1319 的离散），比表中任何 delta 都大，据此判定 GSM8K 差异属运行间噪声。

## 关联

无（PR 正文未给出其他 PR 编号；提到的 `c723a831a8` / `d8de1d3a2f` 为 commit hash，非 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
