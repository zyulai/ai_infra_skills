# PR #47052: [Perf] [AMD] qwen3: Fuse prefill QK-norm + RoPE on ROCm with the JIT qknorm_rope kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @fengshaoyi-amd |
| 合并 | 2026-11-17 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | run-ci, jit-kernel, run-ci-amd-extra |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 摘录含 7 个文件、其中 1 个新测试文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/47052 |

## 一句话结论

修复 HIP 上 JIT `fused_inplace_qknorm_rope` 的 64-bit mask shuffle 编译失败,让 ROCm Qwen3 prefill 把每层 5 次 launch 的 QK-norm + RoPE 融合为一次 in-place kernel:aiter 路径 TTFT 降 2.5% 到 6.4%,无 AITER 时 decode TPOT 降 13.3% 到 20.3%、吞吐升 12.7% 到 25.3%。

## 动机

ROCm 上 Qwen3 prefill 的 QK-norm 与 RoPE 每层 5 次 launch:`apply_qk_norm` 把 packed qkv 的 strided q/k 视图拷贝成连续张量、跑两个 aiter RMSNorm,再由 `rotary_emb` 启动 `rotary_embedding`。`apply_qk_norm` 中的 fused 路径是 CUDA-only。Qwen3-8B prefill profile(MI355X,batch 1/8、input 2048/8192,main `7fe5b127e4`)显示这些 kernel 占 GPU kernel time 4.8% 到 6.8%。diffusion JIT kernel `fused_inplace_qknorm_rope` 本可一次 in-place 完成(strided q/k、GQA、fp32 cos/sin cache),但裸 `__shfl_sync` 带 32-bit mask 触发 HIP 的 64-bit-mask static_assert 而无法编译,#35573 打开的 ROCm 路径于是静默回退到非融合算子。

## 优化点

- `qknorm_rope.cuh`:HIP 上用 `__shfl`/`__shfl_xor` 在 width 32(kernel 逻辑 warp,wave64 下宽度须为 kWarpThreads)替代带 mask 的 shuffle;CUDA 构建不变。
- `qknorm_rope_jit.py`:HIP 上对两个无数值覆盖的变体(KV packing、`round_norm_before_rope=False` 的 NeoX)`can_use_fused_inplace_qknorm_rope` 返回 False、直接调用抛 NotImplementedError。
- `models/utils.py`:新增 `can_use_jit_qk_norm_rope`(在 init 时构建 kernel,避免首次编译发生在 CUDA graph capture 中)与 `apply_qk_norm_rope`(use_jit 时 in-place 一次完成,否则回退 `apply_qk_norm` + `rotary_emb`)。门控:HIP、`--enable-fused-qk-norm-rope` 未被显式关闭、plain `RotaryEmbedding` + fp32 cache、plain RMSNorm 权重为激活 dtype、无 deterministic/on-policy/inductor prefill 模式。
- `qwen3.py`:`forward_prepare_native` 在 #46908 的 CUDA decode 分支之外改调 `apply_qk_norm_rope`;ROCm 上服务 prefill/extend,以及 aiter decode 融合(#44267)不适用时的 decode。

## 关键改动

```text
python/sglang/kernels/jit/csrc/diffusion/qknorm_rope.cuh:
+template <typename T>
+SGL_DEVICE T shfl_idx32(uint32_t mask, T v, uint32_t src_lane) {
+#ifndef USE_ROCM
+  return __shfl_sync(mask, v, src_lane);
+#else
+  return __shfl(v, src_lane, device::kWarpThreads);
+#endif
+}
python/sglang/srt/models/utils.py(新函数):
+def apply_qk_norm_rope(q, k, q_norm, k_norm, rotary_emb, positions, head_dim, alt_stream=None, use_jit=False):
+    if (use_jit and positions.dim() == 1 and q.dtype == q_norm.weight.dtype):
+        ...
+        fused_inplace_qknorm_rope(q.view(num_tokens, -1, head_dim), k.view(num_tokens, -1, head_dim),
+            q_norm.weight, k_norm.weight, rotary_emb.cos_sin_cache, positions,
+            is_neox=rotary_emb.is_neox_style, eps=q_norm.variance_epsilon,
+            head_dim=head_dim, rope_dim=rotary_emb.rotary_dim, round_norm_before_rope=True)
+        return q, k
python/sglang/srt/models/qwen3.py:__init__ 求 use_jit_qk_norm_rope;forward_prepare_native 改调 apply_qk_norm_rope。
test/registered/kernels/ops/attention/test_qk_norm_rope_hip.py(新,AMD CI):fused vs unfused、fp64 参考误差不大于 unfused、门控回退、mRoPE 拒绝等。
```

## 性能声明

(数字全部照抄 PR 正文,未独立核实。环境:MI355X(gfx950),torch 2.11.0+rocm10.0.0,sglang-kernel 0.4.10,aiter `285579b62`。)

单层微基线(32 q / 8 kv heads,head_dim 128,bf16,q/k 为 qkv strided 视图,CUDA events,median of 50):2048 tokens:66.6 → 23.8 µs(2.80x);8192:156.9 → 48.0(3.27x);16384:301.8 → 80.1(3.77x);65536:1194.5 → 436.0(2.74x)。

Prefill profile(`one_batch --attention-backend aiter --profile-stage prefill`,36 层 kernel time 求和):1×2048:28.770 → 27.876 ms(−3.1%),QK-norm+copies+RoPE 1.959 → 0.506 ms,launches 180 → 36;8×8192:986.627 → 960.753 ms(−2.6%),47.313 → 15.531 ms,180 → 36。

端到端(Qwen3-8B 单 MI355X,`SGLANG_USE_AITER=1`,`--attention-backend aiter --disable-radix-cache`,bench_serving random 数据集,blocks of 4 runs 取 median):"TTFT improves by 2.5% to 6.4% in every workload, above the base-to-base noise of at most 1.1%; throughput gains are 0.6% to 1.9%, and TPOT changes by at most 1.3%."(如 W3 1024/1024 c64:TTFT 608.5 → 569.5 ms,−6.4%。)`SGLANG_USE_AITER=0`(decode 也走融合):Qwen3-8B TP1 "Median TPOT drops 13.3% to 20.3% and throughput rises 12.7% to 25.3%. Mean TTFT drops 8.2% to 10.4% in every row ... the median rises 4.8% at W2 c64"(该行噪声 13.9%)。Qwen3-32B TP8(`SGLANG_USE_AITER=1`,decode 仍走 aiter 融合):"TTFT drops 2.1% to 2.6%, above the noise of at most 1.0%; TPOT moves by at most 0.4% and throughput by at most 1.1%."

## 验证与回归风险

精度:GSM8K(`run_eval --eval-name gsm8k --num-examples 1319 --temperature 0`,1314 questions)aiter backend:base 0.9444/0.9422/0.9452(mean 0.9439)vs 本 PR 0.9482/0.9475/0.9460(mean 0.9472);triton 0.9467 → 0.9437。MI325X(gfx942):0.9437 vs 0.9475。`SGLANG_USE_AITER=0`:base mean 0.9419 vs 0.9444。正文解释:融合 kernel 读 fp32 cos/sin 表,unfused HIP RoPE 先舍入到 bf16,输出差 1-2 ulp 且更接近 fp64。单测:AMD CI 上 `test_qk_norm_rope_hip.py` 130 passed、`test_qknorm_rope_out_of_place.py` 1 passed(12 skipped)、`test_qwen3_aiter_fused_qk_norm_rope.py` 6 passed。CI 状态:Base 通过,Extra 与 AMD ROCm 10 失败。回归风险:(1) `--enable-torch-compile` 在 main 上本就因 aiter decode op 的内存错误失败,torch.compile 不在覆盖范围;(2) 大量回退分支(YaRN/mRoPE/eps 不一致/fp32 norm 权重/deterministic/on-policy/inductor/显式关闭)需保持;(3) 与 #40585、#47001、#47785 都改 `forward_prepare_native`,后落地者需要小 rebase;(4) HIP 上 KV packing 与 unrounded NeoX 变体被门控拒绝,相关调用方(如 Cosmos3 video)保持分离算子。

## 关联

同区域:#40585(MiniMax-H3 的 shuffle 修复,本 PR 自带等价修复以免依赖)、#47001(XPU in-place 融合,open)、#47785(CUDA prefill 融合,open)、#44267(aiter decode 融合)、#46908(CUDA decode 分支)、#35573(打开 ROCm 路径但静默回退)。本批次内无直接依赖项。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
