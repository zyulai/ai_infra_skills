# PR #65521: [Perf][mHC] use FlashInfer mega mHC for MHCFusedPostPreOp (GLM-5.3-Flash)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Ayu190505 |
| 合并 | 2026-11-12 |
| 分类 | Attention / Kernel(tier: major) |
| Label | performance, ready, nvidia, glm |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/65521 |

## 一句话结论

GLM-5.3-Flash 的 fused mHC step(mHC post + 下一 sublayer pre + RMSNorm,hidden 4096、4 条 residual streams)在 kernel microbenchmark 显示更快的 shape 上改用 `flashinfer.mega_mhc` 单 launch 融合 kernel:R200 小 token 即启用(8–25%,≥1024 tokens 32–46%),GB300 从 512 tokens 起(≥1024 tokens 22–34%)。

## 动机

TileLang 路径在 ≤32 tokens 走两次 launch,更大 batch 走 post + TF32 projection GEMM + pre;FlashInfer 提供把整个 mHC step 融合为一次 cooperative launch 的 `mega_mhc` kernel(hidden-4096 程序需 flashinfer-ai/flashinfer#7024,PDL launch 需 #7885,均已在 0.7.3 pin 中)。FlashInfer PyPI wheel 缺 experimental kernel sources(#7977),此时无法 JIT 构建、模型保持 TileLang;nightly 或源码安装则启用。

## 优化点

- 不加新选项/env var:op 内按 compute capability 维护每架构 token 阈值 `_MEGA_MHC_MIN_TOKENS = {107: 5, 103: 512}`(R200 sm_107a 阈值 5,GB300 sm_103a 阈值 512,其他 GPU 保持 TileLang、未测量)。T 为每 GPU token 数(batch/DP 下的 batch、prefill 的 chunk),一个阈值覆盖所有并行方式;模型在 CUDA graphs 下 eager 运行,每个 capture size 记录自己的分支。
- `kernels/mhc/flashinfer_mega.py`(新增):custom op `mhc_fused_post_pre_flashinfer_mega` 只跑 FlashInfer;`mega_mhc_supported()` 判断 FlashInfer 是否有该 shape/device 的 program 且安装了 kernel sources。
- BF16 split:kernel 把 FP32 projection `hc_*_fn` [24, 16384] 读成三块 BF16 [3, 32, 16384],decoder layer 持有并在 `process_weights_after_loading` 生成、weight loader 就地重切,保证 CUDA graph 与权重更新(reload、RL weight sync)正确。
- Workspace:cross-CTA counters(每 launch grid 一对)作为 `WorkspaceManager` persistent resource,scratch 走 `get_simultaneous`,全部在首次 eager call、`lock_workspace()` 前分配。
- JIT warmup:`MegaMHCKernel`(`VllmJitKernel`)预构建 T ≤ min(max_num_batched_tokens, 64 × SMs) 的所有 program。
- Chunking:FlashInfer(#8199 之前,含 0.7.3)单 launch 拒绝超过 64 tokens/SM,更大 batch 分多次 launch(mHC 逐 token,切分精确)。

## 关键改动

```text
benchmarks/kernels/benchmark_mhc_flashinfer_mega.py(新增): TileLang vs FlashInfer mega 每 token 数计时
tests/kernels/mhc/test_mhc_flashinfer_mega.py(新增): float64 reference 对照 + CUDA graph / 权重更新 / warmup 覆盖
vllm/model_executor/kernels/mhc/flashinfer_mega.py(新增): custom op + mega_mhc_supported + MegaMHCKernel JIT warmup
vllm/model_executor/kernels/mhc/__init__.py: 导出新 op
vllm/models/glm5next/common/model.py: _MEGA_MHC_MIN_TOKENS / use_mega_mhc / hc_fused_post_pre 分支 / fn_split 权重管理
```

```python
        if self.use_mega_mhc and residual.shape[0] >= self.mega_mhc_min_tokens:
            assert fn_split is not None and norm_weight is not None
            return torch.ops.vllm.mhc_fused_post_pre_flashinfer_mega(
                x,
                residual,
                post,
                comb,
                hc_fn,
                fn_split,
                hc_scale,
                hc_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.mhc_post_mult_value,
                self.mhc_sinkhorn_iterations,
                norm_weight,
                norm_eps,
            )
```

## 性能声明

- Kernel microbenchmark(`benchmark_mhc_flashinfer_mega.py`:每 op 以 100 对 [predecessor → mHC] 在一个 CUDA graph 内计时、扣除 predecessor 单独耗时;每对独立权重拷贝使权重如真实 45 层一样来自 HBM;`--predecessor plain` 模拟 TP all-reduce 后、`pdl` 模拟 DP/DEP 可重叠;每模式每 GPU 跑两次、相差 <1%):
  - **R200**:fused kernel 为带 grid-wide barrier 的全 SM cooperative launch,≤~32 tokens 平坦 ~7.7 µs;TileLang ≤4 tokens 有专用小 batch kernel(6.3–7.0 µs)更快;≥5 tokens TileLang 8.3 µs 起,fused 在全部测得 T 上获胜:≤512 tokens 快 8–25%,≥1024(prefill chunks)快 32–46%。
  - **GB300**:小 T 两路都约 9.7 µs;512 以下最多 ±2% 平手、仅 T=12 与 24–64 明确获胜、T=1/96/256 反而慢 3–4%;≥512 每个 T 都获胜,≥1024 快 22–34%。
- E2E(GLM-5.3-Flash NVFP4 serving,7 configs:agg DEP2/TP4/TP1、disagg 1P TP4 + 1D TP4、3P/4P/6P TP1 + 1D DEP4/DEP8;fused vs production、same-day pairs、sa-bench 8192/1024;运行先于按 microbenchmark 定阈,当时 TP 从 5 tokens、TP1/DP/DEP 从 1 token 即启用):
  - **R200**:mean tok/s/user DEP2 +2.8%、disagg TP4 +2.6%、TP4 +1.2%、TP1 +0.9%;大 batch DEP4/DEP8 disagg −0.2% 至 +0.7%(噪声内)。最终规则只改 DEP2 C1–C8 与 TP1 C2–C4 回 TileLang,其余点与最终 kernel 选择一致。
  - **GB300**:旧规则下小 decode batch 亏(DEP2 C1–C32、TP4 C8/C16、TP1),按 512 阈值现跑 TileLang;仍用 fused 的点中 disagg TP4 decode C512–C1024 +2.6% 至 +2.8%;prefill 仍用 fused。
  - gsm8k 两 GPU 均无变化;27 个 R200 run 在 "JIT kernel warmup finished" 后无 FlashInfer JIT 构建(空 JIT cache)。
- 已知限制:CUDA-graph padding 行的 NaN 拖慢 FlashInfer 系数计算——R200 上 fused 升至 ~12.7 µs、最多输到 512 tokens;GB300 上 T=3–24 输 2–8%;仅影响非 capture size 的 batch。

## 验证与回归风险

- `pytest -v tests/kernels/mhc/test_mhc_flashinfer_mega.py`:R200 + FlashInfer main(f13c875eb)**13 passed**;覆盖:对 float64 reference(以 TileLang 误差为标尺)T=1–4096、单次与多次 launch(>64 tokens/SM,余 3 tokens)、in-place 权重更新传导到 BF16 split、`lock_workspace()` 后 CUDA graph replay 等于 eager、JIT warmup 覆盖所有选中 program、其他 shape 保持 TileLang。
- CI:B200 "mHC Kernels" job 收集该测试;CI 从 PyPI 装 FlashInfer 0.7.3(无 experimental kernel sources),GPU tests 在此 skip、只跑 shape test;R200 上同文件在移除 kernel sources 后 1 passed + 12 skipped,与 PyPI wheel 一致。
- 回归风险:PyPI 0.7.3 wheel 用户保持 TileLang;NaN padding 限制(见上);阈值仅 R200/GB300 实测,其他 GPU 未测量;GB300 在 12–64 token 区间内选阈值会在平/输尺寸旁追百分之几的收益,故保守取 512。

## 关联

- 依赖 flashinfer-ai/flashinfer#7024(hidden-4096 programs)、#7885(PDL launch)、#7977(PyPI wheel 缺 sources)、#8199(>64 tokens/SM 单 launch,0.7.3 尚未包含)。
- Follow-up(正文):把 fused kernel 的 NVFP4 输出直接传给 NVFP4 MoE(需 fused MoE 提供 pre-quantized input path)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
