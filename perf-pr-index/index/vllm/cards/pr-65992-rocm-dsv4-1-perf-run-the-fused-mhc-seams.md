# PR #65992: [ROCm][DSv4.1][Perf] Run the fused mHC seams as AITER's single-launch HIP Mega-MHC

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @davzhuAMD |
| 合并 | 2026-11-15 |
| 分类 | Speculative Decoding(tier: major) |
| Label | performance, rocm, ready, deepseek |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/65992 |

## 一句话结论

gfx950 上 DeepSeek-V4.1 的 fused delayed mHC seam 从 AITER 两 launch Triton kernel 换成单 launch HIP Mega-MHC(覆盖 #59802 侧流的 gates):microbench 每 seam 快 16–41%,C256 decode step 的 seam 时间 −25.5%(约 1.0% step time)、4,096-token prefill step −2.5–2.8 ms(1.6–1.9% step);serving A/B n=3 各项未超出噪声。

## 动机

#58655 引入的 Triton seam 每次两 launch(Triton main + reduce);#59802 把 gates 挪到侧流与 attention/FFN 重叠,但 post-collapse kernel 仍在关键路径。AITER 自 v0.1.27 起(#63774 pin,main 现 pin v0.1.29,#66596)提供单 launch 的 HIP Mega-MHC(`aiter.ops.mega_mhc`,ROCm/aiter#6520),可替换今天 Triton kernel 服务的每一条 seam。这是 kernel 级节省:microbench 每 seam 16–41%;在关键路径上按 profiled serving windows 度量约 1% of a C256 decode step、1.6–1.9% of a 4,096-token prefill step——这个量级预示 serving A/B 在 n=3 下不会分辨出来,正文如实给出该预期。

## 优化点

- **每 seam 一次 launch 替代两次**:gfx950 上每条 fused seam(每 decoder layer 两条 + DSpark draft 的)在 `fn_packed` 存在且 norm 权重为 bf16 时调用 `rocm_aiter_ops.mhc_mega_post_pre_delayed_rms_norm`(单 launch 的 AITER HIP Mega-MHC),否则回落 Triton seam;`fn` 由 `mega_mhc_pack_fn` 一次性 pack 成 bf16 hi/lo MFMF operand 布局。
- **门控与范围**:`is_mega_mhc_available()` 要求 `_AITER_ENABLED` 且 `get_cdna_version() == 4`(tuned kernel 仅 gfx950;gfx942 有通用版但 vLLM 不用)。Triton seam 保留场景:非 gfx950、`VLLM_ROCM_USE_AITER` 未设/为 0(默认)、DBO;无旧 AITER 回退——AITER < v0.1.27 现在在加载时以 `ModuleNotFoundError` 失败(可用 `VLLM_ROCM_USE_AITER=0` 规避,但会关掉其他 AITER 路径)。不新增 env var 或 config field;#59802 的 split 结构不变,只有其侧流 gates 换 kernel。
- **per-stream split-K scratch**:Mega-MHC 的 split-K scratch 按 stream 分配且不能在 CUDA graph capture 中分配,`mega_mhc_prepare_stream` 在模型加载期间(模型默认 dtype 下)以 zeros 各 token 数预分配;权重 reload/in-place update 走 `process_weights_after_loading` 的 `refill_derived`(#65914/#64692)原地 refill packs,先前 capture 的 CUDA graph 读到新权重。
- **数值**:与 Triton seam 不 bitwise(同样的 fp32 数学与 bf16 hi/lo split,求和顺序不同);单元测试断言 residual 在 float64 参考的一个 bf16 ulp 内加 fp32 求和界,其余输出相对误差 ≤ Triton seam 的两倍加小 floor。顺带修复 MI350X 验证发现的 loader bf16 默认 dtype 下 server 起不来的 model-load bug(带回归测试)。

## 关键改动

注:pack 的 diff 被截断(注明 2 个文件略去),以下仅覆盖已展示部分。

```text
vllm/_aiter_ops.py: is_mega_mhc_available 门控 + mega_mhc_pack_fn + mhc_mega_post_pre_delayed_rms_norm(单 launch 封装)+ mega_mhc_prepare_stream
vllm/model_executor/layers/mhc.py: MHCPreDelayedOp.forward_hip 增加 fn_packed 参数并在 Mega-MHC 可用时改派
vllm/models/deepseek_v41/amd/mhc.py: MHCOverlapStream.prepare_mega_mhc(侧流 scratch 预加载期分配)
vllm/models/deepseek_v41/amd/model.py: 权重加载时 pack fn(pack 注明该文件被略去)
tests/kernels/mhc/test_mhc_kernels.py: 新增 test_mhc_pre_delayed_rocm_aiter_mega(24 case)与 bf16 load 回归测试,decoder overlap 测试支持 triton/mega 双参数化
benchmarks/kernels/benchmark_mhc_seam_rocm.py(新增): triton/hip/flydsl/overlap_main 每 seam 延迟基准(HIP-graph replay,float64 参考)
```

```python
-            rest = torch.ops.vllm.mhc_fused_post_pre_delayed_rms_norm_aiter(
+            if fn_packed is not None and norm_weight.dtype == torch.bfloat16:
+                from vllm._aiter_ops import rocm_aiter_ops
+
+                seam, weight = (
+                    rocm_aiter_ops.mhc_mega_post_pre_delayed_rms_norm,
+                    fn_packed,
+                )
+            else:
+                seam, weight = (
+                    torch.ops.vllm.mhc_fused_post_pre_delayed_rms_norm_aiter,
+                    fn,
+                )
+            rest = seam(
                 residual,
-                fn,
+                weight,
```

## 性能声明

- **Microbench**(MI350X,layer-30 权重,HIP-graph replay,cold inputs/weights,rounds 取中位):HIP Mega-MHC 比 main 的 Triton seam(含 post block)快 16–34%,不含 post block 快 27–41%;含 post block 每 seam 2.7–5.3 µs(T 1–256)。
- **Serving trace**(C256 decode,192 rows/rank/step,seam 全部串行 100% exposed,8 ranks 均值):seam 2,069 ± 56 → 1,541 ± 68 µs/step(−528 µs,−25.5%),约 ~50.75 ms step 的 ~1.0%;每 step 85.5 launches、Triton seam kernel 从 171.1 降到 0;每 seam 中位 21.8 µs(13.8+8.0)→ 16.2 µs。C128(overlap 路径,96 rows/rank)只有 59% 的 Mega-MHC 时间 exposed,故 C128 decode 收益很小。
- **Prefill**(random 8192/256 at C32,4,096-token 步,86 seams/step 全串行):seam 7.1–7.5 ms → 4.6–4.7 ms/step(−2.5 至 −2.8 ms),为 140–177 ms step 的 1.6–1.9%,即 0.60–0.68 ms/1,000 prefill tokens;每 seam 中位 83.5 µs(Triton 68.8+14.7)→ 54.3 µs(−35%);serving 与 microbench 同比例(Triton 83.5 vs 161 µs,Mega-MHC 54.3 vs 107 µs)。decode(24 tokens)取 overlap 路径,exposed seam 每步仅降 5–17 µs。
- **Serving A/B**(DeepSeek-V4.1-Flash,DP8 + EP8,MI350X,n=3 per arm,0 failed requests):整体结论为无超出噪声的收益——C128 acceptance-normalized step −0.02%(p 0.92);C256 −1.0%(p 0.05,在噪声线上);低并发 C1–C32 各指标 ±0.5–1.1% bars 内;prefill-heavy 8k 两配置 TTFT p50 与 input throughput 变化 <3%(p ≥ 0.26),4k/C64 input throughput +1.45%(p 0.026)、TPOT p50 −2.7%(p 0.061)。正文注明 trace 测得的节省(~2% TTFT、~1% input throughput)与这些结果一致,n=3 分辨不出该量级。
- **准确度**:GSM8K(5-shot greedy,1,319 题)main 0.9227 ± 0.0074 vs 本 PR 0.9287 ± 0.0071(rebase 前 0.9272/0.9204),均在合并 stderr 内;GPQA Diamond(reasoning high,temp 0.6,8 samples/question)89.08 ± 1.35% vs 88.64 ± 0.76%,配对差 −0.44 ± 0.58 points(p 0.44)。

## 验证与回归风险

- 单元测试 `tests/kernels/mhc/test_mhc_kernels.py`(需 gfx950 + pinned AITER;`test_mhc_pre_delayed_rocm_aiter_mega` 24 case 覆盖 T ∈ {1,2,7,48,128,129,1024,4096} 与 post/no_post/identity_pre 三种 seam 形状,覆盖 decode/split-K/single-split 变体):MI350X Nov 12——main 272 passed/136 skipped,this head 305 passed/136 skipped,merge 306 passed/136 skipped;MI355X 单测亦通过。
- CI:该文件在 `:amd: (MI355) mHC Kernels` step 运行,但该 step 为 optional 且只在触发 AMD CI 时运行,本 PR 尚未跑过,CI 覆盖依赖该 step 被触发。
- 风险:数值非 bitwise(fp32 求和顺序);benchmark 自身输入下 `post_mix` T=4 出现 3.0e-5 相对误差(Triton 1.0e-6),单测在 T 4(MI350X)/T 3–8(MI355X)下通过,正文判断为 benchmark 输入特例、无单测可复现;旧 AITER(<v0.1.27)从"保 Triton seam"变为加载失败;model-load bug 的修复带 bf16 回归测试。Docker/lint:`pre-commit run --files` 六个改动文件全过。

## 关联

- Seam:#58655(Triton seam)、#59802(overlap)、#60153(draft,gfx942 seam);kernel:ROCm/aiter#6520(HIP Mega-MHC)、ROCm/aiter#6179(FlyDSL Mega-mHC)、ROCm/ATOM#2568;AITER pin #63774、#66596;follow-up:同 launch 写 `wqkv_a` 的 MXFP8 输入(基于 #59782/#57857);其他框架:SGLang AMD DeepSeek-V4.1 路径经 FlyDSL Mega-mHC(sgl-project/sglang#49283、#49284,open)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
