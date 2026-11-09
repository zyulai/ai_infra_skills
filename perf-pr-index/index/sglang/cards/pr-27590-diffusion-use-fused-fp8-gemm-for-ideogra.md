# PR #27590: [diffusion] Use fused FP8 GEMM for Ideogram4 weight-only linears

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @qimcis |
| 合并 | 2026-06-11 |
| 分类 | Quantization(major) |
| Label | documentation, run-ci, diffusion |
| Diff | pack 未提供 diffstat(diff 正文可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/27590 |

## 一句话结论

给 diffusion 的 weight-only FP8 linear 接上 fused W8A8 FP8 GEMM,Ideogram4 e2e 延迟 5218 → 3050 ms(1.711x)。

## 动机

sglang-diffusion 的 weight-only FP8 linear(`WeightOnlyFP8Linear` 等)此前每次前向都把整个 FP8 权重张量按行反量化到激活 dtype,再调 `F.linear`。每步去噪都重复做整权重 dequant,既费带宽又费时间。而 SGLang 主线(srt)已有现成的 fused FP8 GEMM(cutlass 路径,activation 动态量化),本 PR 把 diffusion 的这条路径接到它上面。

## 优化点

`_apply_weight_only_fp8_linear` 成为统一入口:当实验开关 `SGLANG_DIFFUSION_ENABLE_W8A8_FP8_GEMM`(默认 false,opt-in)开启且输入/权重/scale 都在 CUDA、compute dtype 为 fp16/bf16 时,直接调用 srt 的 `apply_fp8_linear`(权重保持 FP8,激活动态量化为 FP8,W8A8 计算),并在首次命中时打一条一次性 warning 说明输出质量可能与官方 weight-only 路径不同;调用抛 ImportError/NotImplementedError 时静默回退到原"dequant + F.linear"路径。开关关闭时行为与之前完全一致。`WeightOnlyFP8Linear` / `WeightOnlyFP8ColumnParallelLinear` 增加 `enable_fused_w8a8` 构造参数(默认解析 env),Qwen3VL 文本编码器的两个封装点显式传 `enable_fused_w8a8=False` 保持原路径。附带更新 `environment_variables.mdx` 文档说明该开关的语义与质量风险。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/quantization/weight_only_fp8.py: _apply_weight_only_fp8_linear 路由 + 回退
python/sglang/multimodal_gen/envs.py: SGLANG_DIFFUSION_ENABLE_W8A8_FP8_GEMM 环境变量
python/sglang/multimodal_gen/runtime/models/encoders/qwen3vl.py: 文本编码器显式关闭 fused 路径
docs_new/docs/sglang-diffusion/environment_variables.mdx: 开关文档
```

## 性能声明

测试条件:1x H200,`ideogram-ai/ideogram-4-fp8`,backend `sglang`,attention backend `fa`,preset `V4_TURBO_12`,分辨率 1024x1024,seed 0。E2E 延迟 5218.44 → 3050.28 ms(1.711x,-41.5%);Denoising 延迟 5059.65 → 2944.99 ms(1.718x);峰值 reserved 显存 32,444 → 31,622 MB(低 822 MB)。PR 未注明重复次数等统计信息。

## 验证与回归风险

正确性验证为视觉对比:同一 prompt(带 style/compositional 结构化描述)下 baseline 与 fused 各生成一张 1024x1024 图像,PR 附两图对比,未给出量化指标。默认关闭,关闭时前向路径逐行等价于原实现。风险面:开启后激活被动态量化为 FP8,文档明示"may change output quality";fused 路径依赖 srt `apply_fp8_linear` 与 cutlass 支持检测,不支持平台走静默回退(仅 ImportError/NotImplementedError 被捕获);Qwen3VL 编码器显式排除,避免其文本侧走 W8A8。PR 未新增单元测试(unit tests 未勾选)。

## 关联

同区域:diffusion 量化路径(FlashInfer FP4 GEMM backend override `SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND` 为相邻机制);srt 侧 `apply_fp8_linear`/`cutlass_fp8_supported` 为其复用的底层。pack 未列出其他具体 PR 编号。
