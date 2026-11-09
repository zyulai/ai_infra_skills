# PR #34967: [MoE] Add FlashInfer SM90 MXFP4 W4A8 CUTLASS MoE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-09-01 |
| 分类 | MoE |
| Label | quant, deepseek, run-ci, hopper, flashinfer, run-ci-extra, release-highlight |
| Diff | diffstat 缺失;素材 diff 截去 5 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/34967 |

## 一句话结论

Hopper 新增 FlashInfer Humming MXFP4 W4A8 MoE(opt-in),内核较 W4A16 快 1.63-2.08x,GSM8K 无损。

## 动机

SGLang 已有 FlashInfer SM90 MXFP4 W4A16 CUTLASS MoE(#24816,基于 FlashInfer#3084);FlashInfer#3738 为 Hopper 增加了把激活动态量化到 FP8、跑 MXFP4 权重 × FP8 激活 GEMM 的路径(Humming 式 pre-MMA E8M0 scale fusion)。本 PR 把该路径接入为显式 opt-in,默认 W4A16 行为不变;要求 FlashInfer#4431 修正后的 API(FC1/FC2 residual scale 为 [num_local_experts] 的 FP32 张量),故需 flashinfer-python >= 0.6.18,SGLang 当前 pin 的 0.6.17 下选 fp8 会立即报错。

## 优化点

`--flashinfer-mxfp4-moe-precision` 新增 fp8,仅对 SM90 的 flashinfer_mxfp4 生效;post-load 用 `preprocess_moe_weights_for_sm90_mixed_gemm_humming` 处理 FC1/FC2,保存 folded interleaved 权重 scale 与 per-local-expert residual scale(residual×64,含固定 2^6 补偿);alignment padding 的 synthetic E8M0 scale 用同 expert 已有值填充,防止 padding 改变该 expert 的 min/max exponent 范围——对 FusedMoE 预先 round-up hidden 的场景(GPT-OSS 2880→3072),记住 checkpoint 的真实 K 并把 loader 从未写入的尾列排除在 Humming residual 之外。kernel invocation 构造 5 个 Humming quant-scale slot,并把 TP/EP topology 传给 FlashInfer 以映射 global→local expert;为兼容 0.6.17,既有 W4A16/MXFP8 路径完全不传 `use_wfp4afp8_humming` 关键字。

## 关键改动

```text
python/sglang/srt/layers/quantization/mxfp4.py: _use_sm90_humming 选择;_process_weights_for_sm90_cutlass 增加 preserve_expert_range 与 K_real 尾列排除
python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py: QuantInfo 增加 3 个 humming scale 字段;构造 5-slot quant_scales 与 humming_kwargs
python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py: DSv4 MXFP4 专家同样走 Humming 预处理
python/sglang/srt/server_args.py: flashinfer_mxfp4_moe_precision 增加 "fp8"
test/manual/layers/moe/bench_mxfp4_sm90_kernels.py: 基准加入 FlashInfer W4A8
```

```python
elif use_wfp4afp8_humming:
    quant_scales = [
        quant_info.w13_weight_scale.view(torch.int32),
        w13_humming_residual_scale,
        humming_fc2_act_scale,
        quant_info.w2_weight_scale.view(torch.int32),
        w2_humming_residual_scale,
    ]
```

## 性能声明

内核 benchmark(NVIDIA H20-3e,GPT-OSS-like shape:hidden 4096 / intermediate 2048 / 256 experts / top-k 6,每路径完整 autotune 后 5 warmup + 30 timed):CUTLASS W4A8 对 W4A16 在所有测过的 token 数更快,speedup 1.63x(256 tokens)至 2.08x(8192 tokens),2048 tokens 起约 2.0x;例:4 tokens 0.275→0.164 ms,8192 tokens 25.624→12.308 ms。E2E sanity:DeepSeek-V4-Flash,TP=4,GSM8K 200 例,同配置仅精度选项不同:W4A16 得分 0.985 / 37.301 s / 611.809 token/s;W4A8 得分 0.985 / 32.952 s / 683.574 token/s(评测延迟 −11.7%,输出吞吐 +11.7%)。

## 验证与回归风险

test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py 18 passed 1 skipped,新增 Humming 预处理对拍、padding 不改变 per-expert residual、pre-rounded hidden、FlashInfer<0.6.18 fail-fast 等用例。作用域:SM90-only、opt-in、不改默认;PR 正文称在 SM100/SM120 拒绝该 SM90 选项以免与既有 MXFP8 激活路径混淆;生产启用前需把 FlashInfer python/cubin/JIT-cache pin 一起升级到 0.6.18(checklist 未完成项)。

## 关联

#24816(前序 W4A16 集成);FlashInfer #3738/#4431/#4411;FlashInfer MoE 运行时 autotune 支持另见 #12306。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
