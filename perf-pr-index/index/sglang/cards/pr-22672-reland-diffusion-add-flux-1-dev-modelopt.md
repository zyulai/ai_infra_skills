# PR #22672: reland [Diffusion] Add FLUX.1-dev ModelOpt NVFP4 support

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-04-14 |
| 分类 | Quantization |
| Label | documentation, quant, blackwell, run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);diff 含 10+ 个可见文件(文档、quant config、loader、FLUX DiT、平台层、测试),另有 3 个文件裁剪 |
| 链接 | https://github.com/sgl-project/sglang/pull/22672 |

## 一句话结论

为 SGLang diffusion 增加 FLUX.1-dev ModelOpt NVFP4 混合(BF16+NVFP4)transformer 支持,RTX 5090 4 卡上 denoise 37.69s→29.04s,快 22.95%。

## 动机

SGLang diffusion 已有 ModelOpt FP8 与 FLUX.2 NVFP4(packed-QKV)路径,但 FLUX.1-dev 的 NVFP4 checkpoint 需要"混合 BF16+NVFP4"布局(modulation/FF 部分 module 保留 BF16 fallback),且其 diffusers 导出的 weight 字节序与既有路径假定相反,直接加载会产出错误结果。此外 FLUX 的 attention/single-block 内 ColumnParallelLinear 未显式传 prefix,导致 exclude_modules 的量化前缀匹配不到目标 module;Wan2.2 这类双 transformer pipeline 的全局 `--transformer-weights-path` 会错误地套到 `transformer_2` 上。本 PR 为 reland。

## 优化点

四部分:(1) `ModelOptFp4Config` 新增 `swap_weight_nibbles`(从 flat/nested quant config 读取,默认 True),`_prepare_nvfp4_weight_bytes` 按 config 决定是否做 `(w>>4)|(w<<4)` 字节交换——经校验的 FLUX.1-dev 导出布局固定 `swap_weight_nibbles=false`,由 `build_modelopt_nvfp4_transformer`(`--pattern-preset flux1-nvfp4`)构建混合 checkpoint 并保留 BF16 fallback set(含新增的 `single_transformer_blocks.*.proj_mlp`)。(2) FLUX DiT 的 to_q/to_k/to_v/add_*_proj/proj_mlp/proj_out/attn 显式传入 prefix,使 NVFP4 exclude modules 按完整前缀匹配。(3) 加载侧:`_server_args_for_transformer_component` 对 `transformer_2` 屏蔽全局量化 override 标志;`fsdp_load` 的分片失败改为携带形状信息的 AssertionError。(4) torch.compile 前 `prewarm_nvfp4_jit_modules` 预热 NVFP4 JIT 模块避免 Dynamo trace JIT 初始化;`SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND` 环境变量可优先选 flashinfer `mm_fp4`(绕过 sm100 CUTLASS 对大 M 形状 `can_implement()` 拒绝的已知问题),并把 ModelOpt 校验矩阵固化到 `docs/diffusion/quantization.md`。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py: swap_weight_nibbles 配置 + 字节序处理
python/sglang/multimodal_gen/runtime/models/dits/flux.py: 子模块显式 prefix,修复 quant 前缀匹配
python/sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py: transformer_2 屏蔽全局量化 override
python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py: torch.compile 前 NVFP4 JIT prewarm
python/sglang/jit_kernel/nvfp4.py: prewarm_nvfp4_jit_modules(@torch.compiler.disable)
```

关键机制(可配置的 nibble 处理):

```python
def _prepare_nvfp4_weight_bytes(weight, *, swap_weight_nibbles):
    if not swap_weight_nibbles:
        return weight.contiguous()
    return ((weight >> 4) | (weight << 4)).contiguous()
```

## 性能声明

PR 正文数据(Remote RTX 5090 × 4,torch.compile 全程禁用,1024x1024 输出):

- BF16 benchmark denoise:37.6940 s;NVFP4:29.0421 s(**22.95% faster**)。
- BF16 端到端:38.2545 s;NVFP4:29.4954 s(**22.90% faster**)。
- 正确性(512x512 / 8 steps,vs BF16):trajectory cosine **0.9933**,final image PSNR **28.16 dB**;附 1024x1024 bf16/nvfp4 输出对比图。

## 验证与回归风险

`pytest python/sglang/multimodal_gen/test/unit/test_transformer_quant.py`:新增 nibble swap/跳过、flat/nested config 读取 `swap_weight_nibbles`、builder 写入 `quant_type: NVFP4` 与 fallback patterns、FLUX single-block exclude 前缀匹配等用例。风险点:nibble 交换默认值反转(True)意味着旧的非 FLUX.1 导出仍按原行为,而经校验的 FLUX.1 导出必须带 `swap_weight_nibbles=false` 标记,配置缺失会加载错;FlashInfer FP4 GEMM 优先是文档明示的 workaround(sm100 CUTLASS 大 M 形状拒绝),长期方案是补 CUTLASS fallback;`transformer_2` 屏蔽全局 override 属行为变化,显式 per-component override 仍生效。

## 关联

同区域(diffusion 量化):#27590(fused FP8 GEMM for ideogram)、#34619(GLM image retune)、#41830(Wan VAE decoder kernels)、#45304(AMD diffusion FP8)。配套工具:`tools/build_modelopt_nvfp4_transformer.py`、`tools/compare_diffusion_trajectory_similarity.py`;FP8 对应 builder 更名 `build_modelopt_fp8_transformer`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
