# PR #67906: [ROCm][Perf][DSv4.1] Fuse the MXFP8 input quant of fused_wqa_wkv and the shared expert into their producers

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shen-shanshan |
| 合并 | 2026-11-18 |
| 分类 | MoE(tier: major) |
| Label | rocm, deepseek, quantization |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数);diff 涉及 vllm/_aiter_ops.py、vllm/models/deepseek_v4/amd/model.py、vllm/models/deepseek_v4/amd/silu_and_mul_mxfp8.py(新增)、vllm/models/deepseek_v41/amd/{mhc.py,model.py,rocm.py}、tests/kernels/models/amd/test_dsv41_mxfp8_input_fusion.py(新增)等 |
| 链接 | https://github.com/vllm-project/vllm/pull/67906 |

## 一句话结论

gfx950 上把 DeepSeek-V4.1 三个 MXFP8 linear(`fused_wqa_wkv`、共享专家 `gate_up_proj`、共享专家 `down_proj`)的 bf16 输入量化(`mxfp8_e4m3_quantize`)移进其生产 kernel:每层去掉 3 次独立 quant launch,2× MI355X DEP2 前向步快 0.23–0.38%,输出与未融合路径 bit-identical。

## 动机

AITER FlyDSL MXFP8 路线(#67377)下,`RocmDotScaledMxfp8LinearKernel` 等 kernel 的 bf16 输入由单独的 `mxfp8_e4m3_quantize` launch 量化,且共享专家激活还要经历一次 bf16 round trip;这些 quant kernel 体量小、launch 开销占比高。该 kernel 已能直接读 row-major `[M, K/32]` E8M0 scale 的 `QuantizedActivation`,故由生产者直接交出量化副本即可,GEMM 侧无需改动。

## 优化点

1. ① `fused_wqa_wkv` / ② 共享专家 `gate_up_proj`:AITER Mega-MHC seam 以 `out_dtype="mxfp8"` 运行(`mhc_pre_delayed_mxfp8` → `mhc_mega_post_pre_delayed_rms_norm_mxfp8`),把 `QuantizedActivation` 作为属性挂在 bf16 层输入上(`attach_mxfp8_input`);任何 reshape/slice/gather 都会丢属性,consumer 只看到自己投影行的量化副本;bf16 输入仍供 router、compressor、indexer、MegaMoE 使用,forward 签名不变。
2. ③ 共享专家 `down_proj`:新 Triton kernel `silu_and_mul_mxfp8` 融合 `silu_and_mul_with_clamp` 与 MXFP8 量化——fp32 计算、先舍入到 bf16(与未融合路径一致)、再用 `_mxfp8_quant_triton_kernel` 同款算术量化,输出 e4m3 值与 row-major E8M0 scales。
3. 适用条件:gfx950 + AITER,consumer linear 在 `RocmDotScaledMxfp8LinearKernel` / `AiterFlyDSLMxfp8LinearKernel` 且无 sequence parallelism;①/② 需 Mega-MHC,② 还需 MegaMoE 且共享专家未融合;FULL 图内每 rank ≤128 token 的 decode 走 mHC overlap 路径时 ①/② 不生效。
4. 仅 ROCm 模块改动(`vllm/_aiter_ops.py`、`models/deepseek_v4/amd/`、`models/deepseek_v41/amd/`),CUDA/XPU/CPU 不受影响。

## 关键改动

```python
# vllm/models/deepseek_v4/amd/silu_and_mul_mxfp8.py(激活+量化单 launch)
+    y = (gate / (1.0 + tl.exp(-gate)) * up).to(x_ptr.dtype.element_ty)
+    y = tl.reshape(y.to(tl.float32), (BLOCK_M, BLOCK_N // 32, 32))
+    amax = tl.maximum(tl.max(tl.abs(y), axis=2), TINY)
+    sb = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
+    sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
```

```python
# vllm/models/deepseek_v4/amd/model.py(共享专家 down_proj 融合)
+        if self._down_takes_mxfp8 and gate_up.dim() == 2 and gate_up.is_contiguous():
+            # One launch for the activation and down_proj's MXFP8 input quant.
+            values, scales = silu_and_mul_mxfp8(
+                gate_up, getattr(self.act_fn, "swiglu_limit", None)
+            )
+            x = QuantizedActivation(
+                values, scales, gate_up.dtype, values.shape, kMxfp8Dynamic
+            )
```

## 性能声明

数字均照抄 PR 正文(未独立核实):

- 移除每层 3 次独立 MXFP8 quant launch 及共享专家激活的 bf16 round trip。
- Forward step(`vllm bench forward`,DeepSeek-V4.1-Flash,2× MI355X,DEP2, AITER MegaMoE, fp8 KV, DSpark k=5;时间为每 DP rank、单完整步的 graph-replay 中位,含 sampler 与 drafter):prefill-16k(1×16384-token):393.36 → 392.46 ms(−0.23%);prefill-8k(1×8192):214.41 → 213.62 ms(−0.37%);decode-128(128×verify, 1+5 drafts, KV≈16k):71.81 → 71.54 ms(−0.38%)。正文结论:"the forward step is 0.23-0.38% faster for 8k / 16k-token prefill and 768-token decode steps"。
- End to end on AgentX(c256, DEP2):fusion 前后 throughput 与 interactivity on par。
- Accuracy(GSM8K 8-shot 全量 1319,DEP2 server):strict-match Base 0.9257 → This PR 0.9265;flexible-extract 0.9272 → 0.9272。

## 验证与回归风险

- 单测(MI355X):`pytest tests/kernels/models/amd/test_dsv41_mxfp8_input_fusion.py` 22 passed——两个 producer 均与未融合路径 bit-identical(逐 uint8 码字与 scale 相等),`silu_and_mul_mxfp8` 参数化 1/17/300/5000 tokens × d∈{576, 2304} × clamp on/off,`mhc_pre_delayed_mxfp8` 各输出张量 rtol=0/atol=0 相等且 `QuantizedActivation` 只在融合侧出现。
- trace 前后对照图(PRID 正文截图)显示 ① 与 ②③ 融合后单独 quant launch 消失。
- 回归风险:仅量化本 bit-exact,但 greedy 采样下文本理论上可因 kernel 边界变化漂移(未测量);mHC overlap / mHC seam 不支持时回退原路径;`takes_rocm_mxfp8_input` 在首次使用时惰性解析(权重处理完成后),依赖 quant_method/kernel 就位;DeepSeek-V4 共用 `DeepseekV4MLP` 但其 linear 非 MXFP8,走原路径。
- 非重复:open PR #66572 / #65837(q-norm quant before wq_b)与 #63821 / #66856(wo_a epilogue quant before wo_b)融合的是不同的 quant。

## 关联

依赖 #67377(AITER FlyDSL MXFP8 输入量化路线);相关 open PR #66572/#65837/#63821/#66856;共享 forward 层 `DeepseekV4MLP` 由 DeepSeek-V4/V4.1 两侧共用。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
