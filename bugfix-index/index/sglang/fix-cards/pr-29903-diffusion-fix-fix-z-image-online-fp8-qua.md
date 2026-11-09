# PR #29903: [diffusion] fix: fix z-Image online fp8 quantization crash with dit_cpu_offload

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29903](https://github.com/sgl-project/sglang/pull/29903) |
| 作者 | ppraneth |
| 合并日期 | 2026-07-04 |
| labels | quant, run-ci, diffusion |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论
dit_cpu_offload 使 DiT 权重留在 CPU,online fp8 量化失败后静默回退致 rotary_emb(None) 崩溃;改为量化期保持在设备并抛真实错误。

## 问题现象
执行 `sglang generate --model-path Tongyi-MAI/Z-Image-Turbo --quantization fp8` 时崩溃,报 `TypeError: 'NoneType' object is not callable`,位置在 `zimage.py::get_freqs_cis`(去噪第一步崩溃)。Z-Image-Turbo 的 online fp8 是文档中标注"validated"的路径,因此这是一次回归。closes issue #29833。

## 根因
正文披露:自动内存策略启用 `dit_cpu_offload`,DiT 权重被留在 CPU 上;而 online fp8 在加载时通过仅支持 CUDA/NPU 的 kernel 做量化,对 CPU tensor 会失败。量化失败后 loader 并不报错,而是静默回退到原生 diffusers 模型——该模型没有 `rotary_emb`,去噪时调用 `rotary_emb(None)` 即崩溃。真实错误被回退机制掩盖,症状表现为 `get_freqs_cis` 里的 TypeError。

## 修复方案
- `transformer_load_utils.py`:`TransformerQuantLoadSpec` 新增 `requires_device_weight_processing` 字段,由新函数 `_requires_device_weight_processing` 判定——`fp8` 非序列化 checkpoint 或 `mxfp4` 非序列化 checkpoint 时为 True(其加载后权重处理需要 CUDA/NPU tensor);
- `transformer_loader.py`:把 `dit_cpu_offload and quant_spec.requires_device_weight_processing` 作为 `defer_cpu_offload_until_after_weight_processing` 传入加载器,使 DiT 在量化权重处理期间保持在设备上;同时 `should_raise_customized_load_error` 增加 `quantization is not None` 条件,量化加载失败时抛出真实错误,不再静默回退到 unquantized native 模型;
- `fsdp_load.py`:新增 `defer_cpu_offload_until_after_weight_processing` 参数——延迟时先不加 CPU offload 完成加载与 `post_load_weights()`,随后 `model.to("cpu")` 再 offload(FSDP 加载下忽略该延迟并告警);
- `zimage.py`:`get_freqs_cis` 在 `rotary_emb is None` 时抛出明确的 ValueError(提示走了 native diffusers fallback,应查加载日志中的真实错误);
- 单测:验证 online fp8 需要 device weight processing、modelopt 适配器不改变 online fp8 的 offload 配置等。

正文将第一条概括为"请求 online 量化时禁用 dit_cpu_offload(不只 modelopt_fp8)";diff 的实际机制是延迟到权重处理完成后再 offload。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:改动集中在 multimodal_gen 的 transformer 加载链。online fp8/mxfp4 量化的 DiT 加载期间权重驻留设备(内存峰值升高;省内存仍走 `--dit-layerwise-offload`);量化加载失败改为显式报错,依赖静默回退行为的用户会开始看到错误;`get_freqs_cis` 的 None 防御性报错仅影响已处于错误状态的路径。

## 关联
- closes https://github.com/sgl-project/sglang/issues/29833

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
