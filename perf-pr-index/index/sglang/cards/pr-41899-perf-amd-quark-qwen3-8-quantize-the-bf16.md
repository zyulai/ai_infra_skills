# PR #41899: [Perf] [AMD] quark/Qwen3.8: Quantize the BF16 shared expert of Quark MXFP4 checkpoints to MXFP4 while loading for shared-expert fusion

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @amd-oshkarav |
| 合并 | 2026-10-26 |
| 分类 | MoE(tier: major) |
| Label | documentation, run-ci |
| Diff | diffstat 缺失(素材未统计行数);正文列出 9 个文件(4 个 srt、2 个 docs、3 个测试),diff 素材另省略 5 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/41899 |

## 一句话结论

把 Quark MXFP4 checkpoint 中 BF16 的 shared expert 在加载时量化为 MXFP4 并入 routed grouped GEMM,8x MI355X 输出吞吐 +4.5%。

## 动机

`amd/Qwen3.8-2.4T-A95B-Quark-MXFP4` 只把 routed experts 量化为 MXFP4,shared expert body 在 Quark 量化配置的 `exclude` 列表里、以 BF16 发布;而 shared-expert fusion 的机制是把 shared expert 追加为 `num_experts` 号 routed expert,要求 fused weight 所有 slot 同 dtype,于是 `QuarkConfig.can_fuse_shared_expert()` 因格式不匹配拒绝,模型退回"独立 BF16 MLP + routed grouped GEMM"并行执行。目前没有 shared expert 已 FP4 的 checkpoint,本 PR 在加载时量化 BF16 shared expert,让现有 checkpoint 走 fused 路径。

## 优化点

默认关闭,由既有 `--enforce-shared-experts-fusion` 参数显式开启(该 flag 本意即"在 gate 会拒绝时强制 fusion")。三层分工:量化配置层(`quark.py`)新 `shared_expert_online_mxfp4_supported()` 逐项检查 opt-in flag、ROCm build、`SGLANG_USE_AITER`、gfx95x FP4 硬件、prequantized checkpoint、model_type 白名单(qwen3_5_moe / qwen3_5_moe_text,防 DeepSeek 等未测量家族被静默启用)且所有 MoE 层都解析到 W4A4 MXFP4 scheme,任何拒绝都 warning 并回退 standalone 路径;scheme 层(`quark_w4a4_mxfp4_moe.py`)新增 `quantize_shared_expert()`,用 aiter 的 `dynamic_mxfp4_quant`(仅 ROCm)整张量量化后再由调用方分片,并对 hidden/intermediate size 是 MX block 32 倍数做校验;loader 层(`fused_moe_triton/layer.py`)新增 `_maybe_load_bf16_shared_expert_as_fp4` hook,在 expert id 映射到 fused slot 时拦截 BF16 权重、量化后写入 w13/w2 与 scale。依赖 #40754 的 `mlp.shared_expert.* -> mlp.experts.{num_experts}.*` 权重重映射。注意精度代价:fused 路径下 shared expert 的权重与激活都是 MXFP4(W4A4),替代原 BF16 x BF16。

## 关键改动

```text
python/sglang/srt/layers/quantization/quark/quark.py: online MXFP4 转换的准入 gate 与 can_fuse_shared_expert 改写
python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py: quantize_shared_expert(aiter dynamic_mxfp4_quant,ROCm only)
python/sglang/srt/layers/moe/fused_moe_triton/layer.py: _maybe_load_bf16_shared_expert_as_fp4 loader hook
python/sglang/srt/arg_groups/fields/exec_.py: 扩展 --enforce-shared-experts-fusion help 文本
```

## 性能声明

`amd/Qwen3.8-2.4T-A95B-Quark-MXFP4`,8x MI355X(gfx950)单节点 TP8,`sglang.bench_serve` 12 点 sweep 跑两遍,仅切换 `--enforce-shared-experts-fusion`(ISL 8192/OSL 1024 并发 4..256;ISL 60000/OSL 600 并发 4/16/64;ISL 128000/OSL 1024 并发 4/16)。geomean:输出吞吐 409.8 -> 428.3 tok/s(+4.51%,12/12 点改善),request throughput +4.51%,Mean ITL 46.2 -> 44.1 ms(-4.57%),Mean TPOT -4.57%,Mean E2E -4.24%,Median TTFT -1.53%;无指标 geomean 回退。单点最佳:c=4 时输出吞吐 280.5 -> 309.6 tok/s(+10.38%)。正文标题表述为 +4.5% 输出吞吐、-4.6% mean ITL。

准确性:GSM8K(nightly e2e 参数,greedy,1314 题)fusion ON 0.9764 vs OFF 0.9726(+0.38 pp);eval 墙钟 215.6 s vs 227.2 s(-5.1%),eval 输出吞吐 841.4 vs 806.1 tok/s(+4.4%)。GPQA 0.9242 vs 0.9192;RULER niah_single_2 在 65536/131072 均为 1.0。

## 验证与回归风险

新增/扩展 3 个测试:`test_quark_config.py`(+546)覆盖 gate 的排除识别(shared_expert_gate、draft stack 不计入)、每个前置条件缺一不可、model_type 白名单、per-layer 非 MXFP4 override 的 warning 回退;新文件 `test_quark_shared_expert_online_mxfp4_loader.py`(+474)覆盖 loader hook;`test_qwen38_mxfp4_eval_mi35x.py` 扩展 fused phase(暂 skip)。风险:W4A4 精度交换需要用户自行验证精度(help 文本已注明);不支持的 checkpoint 静默回退 standalone 路径(已有 warning);DeepSeek 家族通过 model_type 白名单显式排除。CI:Base 通过,Extra 与 AMD ROCm 10 失败(素材所示)。

## 关联

依赖:#40754(e2f4fed,shared-expert fusion 权重重映射,FP8 版先例)。同族路径上的 `_maybe_load_fp8_shared_expert_as_fp4` 为既有 FP8 hook;其他同分类 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
