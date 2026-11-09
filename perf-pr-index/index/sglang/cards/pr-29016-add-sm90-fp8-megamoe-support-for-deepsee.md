# PR #29016: Add SM90 FP8 MegaMoE support for DeepSeek-V4

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @qiushixiaoyu |
| 合并 | 2026-07-30 |
| 分类 | Quantization(tier: major) |
| Label | documentation, deepseek, run-ci, jit-kernel, run-ci-extra, release-highlight |
| Diff | diffstat 未提供(pack 素材无增删行数字) |
| 链接 | https://github.com/sgl-project/sglang/pull/29016 |

## 一句话结论

为 DeepSeek-V4 FP8 在 Hopper SM90 上启用 DeepGEMM all-FP8 MegaMoE 路径,SLO 吞吐 +22.24%。

## 动机

此前 DeepSeek-V4 的 MegaMoE(W4A8/W4A4 fused kernel)只在 Blackwell 可用,Hopper H100/H200 上的 FP8 checkpoint 走不到该路径,长上下文/大 decode 吞吐受限。本 PR 在 SM90 上把 MoE 执行接入 DeepGEMM `mega_moe` runner 与 MegaMoE A2A 路径(含 pre-dispatch JIT kernel 与 FP8 expert 权重准备),目标是提升 DeepSeek-V4-Flash/Pro-FP8 的长上下文/大 decode 吞吐,同时保持实验路径由环境变量守护。

## 优化点

新增 `mega_moe_sm90.py` 三件事:(1) `run_sm90_mega_routed` 先调 `deep_gemm.mega_moe_pre_dispatch_sm90` 把 activations、scales、top-k indices/weights 打包进对称 buffer(group_size=128),再调 `deep_gemm.fp8_mega_moe`(recipe (128,128,128)、swiglu 激活、fast_math)一次完成 expert 计算;(2) `build_sm90_mega_moe_experts_weights` 做 FP8 expert 权重准备,校验 128x128 block scale 形状后,若 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1` 则用 gate/up 按 gran=8 interleave 的方式原位重排 w13 避免额外副本,否则走 deep_gemm 的 SF 布局变换与 `transform_weights_for_mega_moe_sm90`;(3) `should_use_mega_moe`/`_run_mega_routed` 挂上 `_device_sm == 90` 分支,要求 deep_gemm 提供 `fp8_mega_moe`/`mega_moe_pre_dispatch_sm90` 且权重已构建。fp8.py 的 block-quant 权重处理在 megamoe a2a + SM90 时直接构建并提前返回(断言非 FP4 expert)。启用靠 `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1`(或 `--moe-a2a-backend megamoe`)加 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1`、`SGLANG_OPT_USE_JIT_EP_ACTIVATION=1` 与 per-rank token 上限。依赖 DeepGEMM 的 FP8 MegaMoE 算子(#36)与其 small-batch 优化(#48)。

## 关键改动

```text
python/sglang/srt/layers/moe/mega_moe_sm90.py: 新增 SM90 FP8 MegaMoE 前向路径与 expert 权重准备
python/sglang/srt/layers/moe/mega_moe.py: should_use_mega_moe/_run_mega_routed 接入 SM90 分支
python/sglang/srt/layers/quantization/fp8.py: megamoe a2a + SM90 时构建 SM90 权重并提前返回
docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx: 新增 SM90 (Hopper) FP8 MegaMoE 配置说明
test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py: 新增 TestDSV4FlashFP8H200MegaMoE e2e 测试
```

SM90 前向核心(摘自 `run_sm90_mega_routed`):

```python
deep_gemm.mega_moe_pre_dispatch_sm90(
    hidden_states, topk_ids, topk_weights,
    buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights,
    num_tokens=num_tokens, group_size=128,
    routed_scaling_factor=routed_scaling_factor,
)
y = torch.empty(...)
deep_gemm.fp8_mega_moe(
    y, moe.experts.mega_l1_weights, moe.experts.mega_l2_weights, buf,
    recipe=(128, 128, 128), activation="swiglu",
    activation_clamp=getattr(moe.config, "swiglu_limit", None),
    fast_math=True,
)
```

## 性能声明

基准用内部脚本 `sglang_auto_bench.py`(包装 `bench_serving`,自动搜索给定 IO 长度下的最高请求率/并发),random prompts,input/output 3500/1500,SLO 目标 TTFT <= 2000 ms、TPOT <= 20 ms;服务端 DeepSeek-V4-Flash-FP8,`--tp 8 --ep-size 8 --chunked-prefill-size 4096 --moe-a2a-backend deepep --moe-runner-backend deep_gemm`,EAGLE 投机解码(steps 3 / topk 1 / draft 4),`--cuda-graph-max-bs 32 --max-running-requests 32 --mem-fraction-static 0.75`(`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096`)。GPU 型号 PR 正文未注明(新增 CI 测试为 H200 runner 配置)。

| Mode | SLO total throughput | SLO output throughput | TTFT | TPOT |
|---|---|---|---|---|
| MegaMoE ON | 7001.55 tok/s | 2100.46 tok/s | 683.94 ms | 13.83 ms |
| MegaMoE OFF | 5727.56 tok/s | 1718.27 tok/s | 825.53 ms | 17.05 ms |

ON vs OFF:SLO total / SLO output throughput 均 **+22.24%**。

准确性:GPQA(198 examples x 16 repeats,thinking)pass@1[avg-of-16] = 88.64% +/- 1.28%,pass@16 = 96.46%,majority@16 = 90.15%;GSM8K Accuracy 0.956、Invalid 0.000;GPQA Score 0.90625(32 examples)。

## 验证与回归风险

验证:新增 JIT pre-dispatch kernel 单测 `python/sglang/jit_kernel/tests/test_mega_moe_pre_dispatch_sm90.py`(10 passed,对照 PyTorch 参考实现);新增 e2e 测试 `TestDSV4FlashFP8H200MegaMoE`(TP=4/DP=4 dp-attention + DeepEP a2a + deep_gemm runner + EAGLE,阈值 gsm8k >= 0.93、accept_length >= 1.8、bs=1 speed >= 140),并把该文件 CI 预算从 280s 提到 560s;CI Base/Extra 通过。风险:实验路径,要求带 SM90 FP8 MegaMoE 支持的 `sgl-deep-gemm` 构建与最新镜像;`SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1` 会原位 interleave w13 权重布局;per-rank cap 不变量(`= chunked_prefill_size / dp_size`,测试注释中为 8192/4=2048)若不匹配,超限 prefill chunk 会回退;`--moe-runner-backend` 不得手动设置;不支持 FP4 expert。

## 关联

上游依赖:sgl-project/DeepGEMM #36(FP8 MegaMoE 算子)、#48(small-batch 性能优化);Blackwell MegaMoE 已有路径见 #26451(wired into high-throughput recipe)。同分类(Quantization)其他 DeepSeek-V4/FP8 相关 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
