# PR #28304: [perf] Use default torch compile mode for Wan2.2 T2V A14B

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-06-16 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | run-ci, diffusion, run-ci-extra |
| Diff | diffstat 缺失(素材未提供统计);diff 实际涉及 8 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/28304 |

## 一句话结论

Wan2.2 T2V A14B 的 DiT torch.compile 从 max-autotune-no-cudagraphs 换成 default,H100/B200 E2E 各快约 5%。

## 动机

sglang-diffusion 的 DiT 编译模式默认 `max-autotune-no-cudagraphs`,该模式编译期开销大且对 Wan2.2 T2V A14B 并无收益。#28050 已在其他模型上得出 default 模式更优的结论,本 PR 把同样的发现应用到 Wan2.2 T2V A14B,并在 H100 与 B200 上用干净编译缓存验证。

## 优化点

两层改动:(1) 配置层——`DiTConfig` 新增 `torch_compile_mode` 字段(默认仍为 `max-autotune-no-cudagraphs`)并暴露 `--dit-config.torch-compile-mode` CLI 参数;`Wan2_2_T2V_A14B_Config.__post_init__` 把它设为 `default`,LTX2Config 同样设为 `default`;(2) 消费层——denoising stage、Hunyuan3D paint、MoVA 的 `_maybe_enable_torch_compile` 改为优先级 `SGLANG_TORCH_COMPILE_MODE` 环境变量 → 各自 dit_config 的 `torch_compile_mode` → 全局默认,MoVA 还按 video_dit/video_dit_2/audio_dit 分别传入对应 config。搭车改动:新增 `LTX23PipelineConfig` 并注册到 Lightricks/LTX-2.3 模型探测。

## 关键改动

```text
python/sglang/multimodal_gen/configs/models/dits/base.py: DiTConfig 增加 torch_compile_mode + CLI 参数
python/sglang/multimodal_gen/configs/pipeline_configs/wan.py: Wan2_2_T2V_A14B 设 default 模式
python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py: 编译模式解析 env → dit_config → 默认
python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/mova.py: 按 config 逐模块编译
python/sglang/multimodal_gen/configs/pipeline_configs/ltx_2.py: 新增 LTX23PipelineConfig
```

```python
def __post_init__(self) -> None:
    self.dit_config.boundary_ratio = self.boundary_ratio
    self.dit_config.torch_compile_mode = "default"
```

## 性能声明

干净缓存重跑(每次运行新建 `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR`/`CUDA_CACHE_PATH`),diffusion benchmark preset,4 GPU,原生 SGLang 路径:

| 硬件 | 指标 | max-autotune-no-cudagraphs | default | 差异 |
| --- | --- | ---: | ---: | ---: |
| H100 | denoise | 217.8232 s | 206.0693 s | +5.70% |
| H100 | e2e | 220.6501 s | 208.8785 s | +5.64% |
| H100 | peak memory | 15.23 GB | 14.00 GB | -1.23 GB |
| B200 | denoise | 110.3138 s | 105.0979 s | +4.96% |
| B200 | e2e | 112.8339 s | 107.5709 s | +4.89% |
| B200 | peak memory | 19.04 GB | 18.00 GB | -1.04 GB |

## 验证与回归风险

`py_compile` 语法检查与 `git diff --check`;B200 干净缓存原生后端基准对比两种模式。PR 未提供输出质量/数值等价验证(torch.compile 模式切换理论上不改变语义,但 PR 未注明)。回归风险:default 模式取消 autotune,某些 shape 的 kernel 选择可能变慢,因此做成 per-model config 而非全局翻转;环境变量仍可覆盖;Hunyuan3D/MoVA 的编译模式解析路径一并改变,若其 config 缺 `torch_compile_mode` 属性则回退全局默认。

## 关联

前序:#28050(同样的 compile-mode 结论)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
