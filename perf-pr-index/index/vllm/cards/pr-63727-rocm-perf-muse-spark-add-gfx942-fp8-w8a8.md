# PR #63727: [ROCm][Perf][Muse Spark] Add gfx942 FP8 W8A8 fused-MoE configs for E=32,N=4096

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mjkvaak-amd |
| 合并 | 2026-11-12 |
| 分类 | MoE(tier: major) |
| Label | rocm, ready |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/63727 |

## 一句话结论

为 gfx942(MI300X/MI325X)新增 Muse Spark TP8+EP 每 rank 专家 shape(`E=32, N=4096, fp8_w8a8`,per-token FP8)的调优 Triton fused-MoE config(纯 config 文件):MI300X 上 64 并发 serving 吞吐 +11.6–24.8%,MI325X 上 64–128 并发 +20.4–21.4%,GSM8K 不变。

## 动机

该 shape 此前回落到 `get_default_config`;没有其他 PR 覆盖这个 shape。按 expert shape 提供按 batch 调优的 tile,可显著提高 EP 下 decode 大 batch 的 MoE 吞吐。

## 优化点

- 新增两个 config 文件:`E=32,N=4096,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json` 与 `...MI325X...json`,各 16 个 batch 档(8…8192)。
- **MI300X 调参**:batch 512–8192 来自 `benchmarks/kernels/benchmark_moe.py --tune --tp-size 8 --enable-expert-parallel --dtype fp8_w8a8`;但 EP 下该脚本让每个 token 路由到 32 个本地专家(每 expert 看到 8× serving 行),其 batch 8–32 的选择输给默认 tile——batch 16–256 用本地 patch(路由到全部 256 个 expert、经 `expert_map` 保留一个 rank 的 32 个,`num_stages ∈ {1,2}`)重调;batch 8 保持默认 tile,batch 1–7 也解析到默认 tile。
- **MI325X 调参**:每个 batch 都在 MI325X 上用 patched script 的 `benchmark_config`(serving 路由、per-token 激活)调,每组搜索的 top 8 重测;batch 8 和 24 保留 MI300X 条目(在 MI325X 上持平或更好);batch 128 取 MI300X tile 与 batch 96/256 top picks 在 batch 128 下的最优。
- **scope**:MI300X 与 MI325X、TP8+EP 无冗余专家(E=32);tile 按 tokens per step 键控,chunked prefill 上限 8192,16k 行已覆盖所有条目;16k 以上 context 未跑。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=32,N=4096,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json(新增)
vllm/model_executor/layers/fused_moe/configs/E=32,N=4096,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8.json(新增)
(正文另附 tuner patch 说明,注明该 patch 不属于本 PR)
```

```json
+{
+    "triton_version": "3.8.0",
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 2
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 1,
+        "num_stages": 1,
+        "waves_per_eu": 4
+    },
```

## 性能声明

- **Serving, MI300X**(8× MI300X,`vllm/vllm-openai-rocm:nightly-83c078d955`,`amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8`,每点 1 run、3 waves,两次 warm-up 后):Total throughput(本 config vs 默认)——1k/1k:c1 +1.1%、c8 −0.7%、c16 +3.7%、c32 +2.8%、c64 **+23.6%**;8k/1k:c64 +15.5%;16k/1k:c64 +11.6%;1k/8k:c64 +24.8%。c1–c8 平坦是设计使然(decode batch ≤8 走默认 tile)。c64 下 1k/1k total tok/s 2,455 → 3,034,TPOT 50.2 → 40.5 ms;8k/1k 6,700 → 7,739;16k/1k 8,768 → 9,781;1k/8k 1,484 → 1,852。
- **Serving, MI325X**(8× MI325X,`nightly-rocm72-3966924d`):1k/1k c1 −0.2%、c8 +0.3%、c16 +1.9%、c32 +5.8%、c64 **+21.4%**、c128 **+20.4%**;c64 TPOT 44.24 → 36.28 ms,c128 55.96 → 46.13 ms。GSM8K(64 并发,flexible-extract)0.9500 ± 0.006 默认 vs 0.9515 ± 0.006 本 config。
- **Kernel, MI300X**(单卡,路由覆盖全部 256 experts、一个 rank 的 `expert_map`,同 serving):batch 16–32 快 5–7%,batch 48–128 快 23–29%;batch 1–8 走默认 tile。
- **Kernel, MI325X**(MoE 调用 µs,5 次交错中位):batch 64 594 → 439、batch 1024 844 → 617、batch 8192 2557 → 2332;MI300X 的 tile 不能直接搬:在 MI325X 的 batch 512、2048、3072 上输给默认 tile。每个 MI325X 条目在同一输入下与默认 tile 输出 bit-identical(FP8 乘积在 fp32 中精确累加)。
- **GSM8K, MI300X**(5-shot,1,319 题,64 并发):0.9507 ± 0.006 默认 vs 0.9507 ± 0.006 本 config。

## 验证与回归风险

- 两臂同镜像同 flags,candidate 仅把 config 文件挂载覆盖 `vllm/model_executor/layers/fused_moe/configs/`,server log 确认加载。
- 风险:纯 config 变更,但覆盖面限定该 E/N/K shape 与两款 gfx942 设备;MI300X 的 tile 不可移植到 MI325X(batch 512/2048/3072 会回退);serving 表用后来被丢弃的 batch 1/2/4 条目测过(只影响 batch < 8,现在走默认 tile);context > 16k 未跑;tuner patch 不在本 PR 内。

## 关联

- 无直接关联 PR;正文注明"No other PR adds this shape"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
