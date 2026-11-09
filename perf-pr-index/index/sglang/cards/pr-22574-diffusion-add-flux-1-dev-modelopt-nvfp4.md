# PR #22574: [Diffusion] Add FLUX.1-dev ModelOpt NVFP4 support

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-04-12 |
| 分类 | Quantization (tier: major) |
| Label | documentation, quant, blackwell, run-ci, diffusion, jit-kernel |
| Diff | pack 未提供 diffstat(diff 因超长截断,2 个文件未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/22574 |

## 一句话结论

为 SGLang diffusion 新增 FLUX.1-dev ModelOpt NVFP4 混合精度支持,4x RTX 5090 上 denoise 提速 22.95%。

## 动机

SGLang diffusion 的 ModelOpt NVFP4 此前只验证过 FLUX.2-dev 的 packed-QKV 加载路径,FLUX.1-dev 缺少可用的 mixed BF16+NVFP4 checkpoint 构建与加载方案。同时 NVFP4 加载对 nibble 交换不可配置,无法适配不同导出格式的布局;FLUX attention/single-block 的 quant prefix 也不完整,导致 FLUX.1 fallback exclude 匹配不到预期模块。本 PR 补齐这三点并给出验证数据。

## 优化点

机制改动分三层。其一,新增 `build_modelopt_nvfp4_transformer` 工具,以 `--pattern-preset flux1-nvfp4` 构建 mixed BF16+NVFP4 transformer,保留 `quant_type: NVFP4` 与 `swap_weight_nibbles: false`,选定的 modulation/FF 模块留在 BF16。其二,`ModelOptFp4Config` 从扁平与嵌套两种 quantization config 中读取 `swap_weight_nibbles`,加载时按配置决定是否对权重字节做 `(w>>4)|(w<<4)` 交换再 padding 进 runtime kernel(默认 True,兼容旧行为)。其三,denoising 阶段在 torch.compile trace 之前调用 `prewarm_nvfp4_jit_modules` 物化 NVFP4 JIT 模块,避免 Dynamo tracing 触发 JIT 初始化。另外给 FLUX 各线性层显式传入完整 prefix,使 exclude_modules 按全路径命中。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py: 新增 _prepare_nvfp4_weight_bytes 与 swap_weight_nibbles 配置解析
python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py: torch.compile 前预热 NVFP4 JIT 模块
python/sglang/multimodal_gen/runtime/models/dits/flux.py: FLUX 线性层补全 prefix 以匹配 exclude modules
python/sglang/jit_kernel/nvfp4.py: prewarm_nvfp4_jit_modules 入口
python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py: 由 convert_modelopt_fp8_checkpoint.py 重命名,新增 LTX2 keep-BF16 patterns
docs/diffusion/quantization.md: ModelOpt FP8/NVFP4 验证矩阵(canonical support matrix)
```

```python
def _prepare_nvfp4_weight_bytes(
    weight: torch.Tensor, *, swap_weight_nibbles: bool
) -> torch.Tensor:
    """Normalize serialized NVFP4 bytes before padding for the runtime kernel."""
    if not swap_weight_nibbles:
        return weight.contiguous()
    return ((weight >> 4) | (weight << 4)).contiguous()
```

## 性能声明

测试条件:Remote RTX 5090(4 GPUs),benchmark/profile/correctness 全程禁用 `torch.compile`;BF16 与 NVFP4 除 checkpoint 覆盖外命令完全一致,1024 分辨率、layer offload。

- BF16 benchmark denoise:37.6940s;NVFP4:29.0421s(快 22.95%)
- BF16 end-to-end:38.2545s;NVFP4:29.4954s(快 22.90%)
- Correctness @512x512 / 8 steps:轨迹余弦相似度 0.9933,最终图像 PSNR 28.16 dB

## 验证与回归风险

依赖 `pytest python/sglang/multimodal_gen/test/unit/test_transformer_quant.py`,新增用例覆盖 nibble 交换/跳过、扁平与嵌套 config 的 swap_weight_nibbles 读取、builder 写入 quant_type,以及 FLUX single-block prefix 与 exclude 匹配(验证 proj_mlp/proj_out/to_q 落到 UnquantizedLinearMethod)。回归面:NVFP4 加载路径(默认 swap=True 行为不变,显式 false 的 checkpoint 走新路径)、FLUX.1/FLUX.2 diffusion 模型加载、torch.compile 与 NVFP4 JIT prewarm 的组合;`fsdp_load.py` 仅将 weight_loader 的 AssertionError 包装为带 shape 上下文的更详细报错。

## 关联

pack 正文未列出关联 PR 编号。同区域:FLUX.2-dev NVFP4 packed-QKV 加载路径与 ModelOpt FP8 builder(见 docs/diffusion/quantization.md 同一验证矩阵)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
