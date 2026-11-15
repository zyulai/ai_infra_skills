# PR #63627: [Perf] Tuned batch-invariant fused MoE configs on Blackwell

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yifanFengg |
| 合并 | 2026-11-11 |
| 分类 | MoE(tier: major) |
| Label | ready, batch-invariance |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/63627 |

## 一句话结论

`VLLM_BATCH_INVARIANT=1` 下为 Blackwell Triton fused MoE 增加按架构的调优表(one config per shape,用于所有 M):Qwen3-235B-A22B-FP8 TP4 decode 吞吐 +44.7–47.6%,DeepSeek-V3.1 +107.5–108.8%,Qwen3-30B-A3B(bf16)+3.9–14.8%;batch-invariance 测试全部保持通过。

## 动机

开启 batch-invariance 后,`get_moe_configs()` 返回 `None`、`get_default_config()` 无 batch-invariant 表,Triton fused MoE kernel 只能跑固定 config(`BLOCK 64x64x32, GROUP_SIZE_M 8, SPLIT_K 1`),远慢于调优配置。需要一个既不破坏 batch 不变性、又接近最优 tile 的配置来源。

## 优化点

- 新增 `vllm/model_executor/determinism/batch_invariant_configs.py` 中的表 `_BATCH_INVARIANT_MOE_CONFIGS[(arch family, dtype, block_shape, E, N, K)] -> config` 与查找函数 `get_batch_invariant_moe_config()`;N/K 取 kernel 视角(per-rank `w2.shape[2]`、`w1.shape[2]`),arch family 复用 `_get_tuned_matmul_arch_family()`(major 10 → `blackwell`)。`fused_moe.py::get_default_config()` 在 flag 下查表,无条目的 shape 保持原固定 config。
- **为什么 one config per shape 而不是 M-bucketed tiles**:第一版按 M bucket 选 BLOCK_SIZE_M/N、num_warps/num_stages(固定 BLOCK_SIZE_K/SPLIT_K),kernel 级 checker(8 probe rows 插入随机 batch,M=1…16384,fp8 block-quant,GB300)显示每个 bucket 边界两侧 bitwise 一致,但模型级 needle test 仍失败(5/5 trials)——在 Blackwell 上即使固定 K reduction,Triton fused MoE 的 per-row 结果仍依赖 tile 配置;只有"每 shape 一个 config 用于所有 M"通过。数据上 tuned config 也显著优于固定 64×64×32(Qwen3-235B shape,M=32:381 µs 固定 vs 211 µs 单一 tuned config vs 132 µs bucketed)。
- **调参方法**:每 shape 扫 `BLOCK_SIZE_M ∈ {16,32,64,128} × BLOCK_SIZE_N ∈ {64,128,256} × num_warps ∈ {4,8} × num_stages ∈ {2,3,4} × BLOCK_SIZE_K ∈ {32,64,128}`(`GROUP_SIZE_M=8`、`SPLIT_K=1`),经 `fused_experts()` 在 `VLLM_BATCH_INVARIANT=1` 下 M=1…4096(CUDA-graph 计时,20 次中位)计时,剔除与固定 config 输出差超 3% 的候选,取全 M 总时间最低者。条目在 GB300(SM 10.3)按 Qwen3-235B-A22B-FP8、DeepSeek-V3.1、GLM-5.3 的 TP=4 fp8 shape 与 Qwen3-30B-A3B 的 bf16 shape 调出。

## 关键改动

```text
vllm/model_executor/determinism/batch_invariant_configs.py: _MoEConfig + _BATCH_INVARIANT_MOE_CONFIGS 表 + get_batch_invariant_moe_config() + 默认固定 config
vllm/model_executor/layers/fused_moe/fused_moe.py: get_default_config() 在 VLLM_BATCH_INVARIANT 下改调 get_batch_invariant_moe_config()
```

```python
+_BATCH_INVARIANT_MOE_CONFIGS: dict[
+    tuple[str, str | None, tuple[int, ...] | None, int, int, int], _MoEConfig
+] = {
+    # Qwen3-235B-A22B-FP8, TP=4
+    ("blackwell", "fp8_w8a8", (128, 128), 128, 384, 4096): _MoEConfig(
+        64, 128, 128, 4, 4
+    ),
+    # DeepSeek-V3.1, TP=4
+    ("blackwell", "fp8_w8a8", (128, 128), 256, 512, 7168): _MoEConfig(
+        32, 128, 128, 4, 3
+    ),
+    # GLM-5.3, TP=4
+    ("blackwell", "fp8_w8a8", (128, 128), 256, 512, 6144): _MoEConfig(
+        32, 128, 128, 4, 3
+    ),
+    # Qwen3-30B-A3B (bf16), TP=1
+    ("blackwell", None, None, 128, 768, 2048): _MoEConfig(64, 128, 64, 4, 3),
+}
```

## 性能声明

(4×GB300,本 diff 叠于 `2d40c0609f`,`VLLM_BATCH_INVARIANT=1`,FP8 MoE oracle 在 flag 下解析为 `TRITON`,每 workload 两轮,取均值增益)

- **Qwen3-235B-A22B-FP8, TP=4**:serve decode 128 并发 output tok/s 2434.69/2420.71 → 3587.11/3580.00(+47.6%);32 并发 814.74/814.18 → 1177.22/1179.34(+44.7%);serve prefill 4096→1 req/s 3.71 → 5.64(+52.0%);offline throughput 128→512 output tok/s +49.9%;offline 4096→1 total tok/s +52.3%。
- **DeepSeek-V3.1 (fp8), TP=4**:decode 128 并发 +108.8%(1185.46/1186.14 → 2480.46/2470.89);32 并发 +107.5%;prefill req/s +32.6%;offline 128→512 output +110.8%;offline 4096→1 total +32.8%。
- **Qwen3-30B-A3B (bf16), TP=1**(每臂 1 轮):decode 128 并发 +5.8%;32 并发 +4.2%;prefill req/s +14.6%;offline 128→512 +3.9%;offline 4096→1 +14.8%。
- 参照(同 server 关 flag、非 batch invariant,FlashInfer TRT-LLM FP8 MoE):Qwen3-235B decode@128 7350.64 tok/s,DeepSeek-V3.1 3806.61 tok/s——本表在不触碰非不变路径的情况下缩小部分差距。
- Kernel 级:Qwen3-235B shape、M=32 下 381 µs(固定)→ 211 µs(单一 tuned config)。
- 准确度:gsm8k(lm_eval,5-shot,flexible-extract,Qwen3-235B-A22B-FP8,flag 下)main 0.8802 ± 0.0089 vs PR 0.8772 ± 0.0090。

## 验证与回归风险

- Batch-invariance suite(`tests/v1/determinism/test_batch_invariance.py`,FLASH_ATTN/TRITON_ATTN ids,Qwen3-235B-A22B-FP8 TP=4):PR 树上 6/6 通过;DeepSeek-V3.1 TP=4 TRITON_MLA(needle、bs1-vs-bsN、ragged chunked prefill)2/3 通过,ragged case 在 main 上同样失败。
- Needle 对照表(Qwen3-235B-A22B-FP8 TP=4,flag 下):main PASS;M-bucketed tiles + 固定 BLOCK_K=128 FAIL 5/5;固定 tiles 只改 BLOCK_K=128 PASS;M-bucketed + BLOCK_K=32 FAIL 1/5;one config per shape(本 PR)PASS——证明 tile 配置本身影响 per-row 结果,只能按 shape 锁 config。
- 风险:仅覆盖表中 4 个 (arch family, dtype, block_shape, E, N, K) 条目,其余 shape 维持固定 config;M-bucketed 方案的更大潜在收益(132 µs vs 211 µs @ M=32)不可用;所有测量在 GB300(SM 10.3)。

## 关联

- 无直接关联 PR;正文提及 dense matmul 已有 `batch_invariant_configs.py` 的 per-M-bucket 表,本 PR 因 batch 不变性约束改用 per-shape 方案。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
