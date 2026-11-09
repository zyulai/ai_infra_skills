# PR #23998: [V1] v1 engine + full CUDA graph support for PLaMo2

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @nopperl |
| 合并 | 2025-09-03 |
| 分类 | CUDA Graph / Compiler |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, llama, qwen, deepseek, gpt-oss |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面) |
| 链接 | https://github.com/vllm-project/vllm/pull/23998 |

## 一句话结论

让 PLaMo2 混合架构接入 v1 engine 与 FULL_AND_PIECEWISE CUDA graph,延迟降约 14%。

## 动机

PLaMo2(Mamba-2 + attention 混合架构)此前是 v0-only:实现继承 transformers 的 `PreTrainedModel`、mamba 状态管理走 `MambaCacheManager`,既不能上 v1 engine,也无法使用 full CUDA graph 与 torch.compile,decode 路径存在大量 kernel launch 开销;同时模型在 hybrid 单测中被跳过(issue #22999)。

## 优化点

核心是把 `Plamo2MambaMixer` 改造成 v1 编译体系的一等公民:mixer 注册为 `CustomOp` 并通过 `direct_register_custom_op` 注册自定义算子 `torch.ops.vllm.plamo2_mamba_mixer`(fake impl 为空、`mutates_args=["output"]`),使其可被 torch.compile 视为不透明算子;同时把 `"vllm.plamo2_mamba_mixer"` 加入 `CompilationConfig._attention_ops`,支持 (piecewise) CUDA graph。状态管理改为 v1 的 kv_cache 接口:`self.kv_cache` 挂 `(conv_state, ssm_state)` 元组、经 `Mamba2AttentionBackend` 的 metadata 提供 `state_indices_tensor`/`chunk_offsets` 等,输出改为写入选前分配的 `output` 张量(inplace 算子约定)。`Plamo2Model` 整体挂 `@support_torch_compile`(原来挂在 attention mixer 上),模型类改为普通 `torch.nn.Module`,移除 `SupportsV0Only`,并删除"不支持 prefix caching"的 assert。另吸收 #21075/#18218 的 MambaMixer2 改进与 #23520 的修复。

## 关键改动

```text
vllm/model_executor/models/plamo2.py: MambaMixer 改 CustomOp + 自定义算子,v1 状态接口,输出 inplace
vllm/config/compilation.py: _attention_ops 加入 vllm.plamo2_mamba_mixer
tests/models/language/generation/test_hybrid.py: plamo-2-1b 恢复并加入 V1/full CUDA graph 模型列表
```

```python
if not envs.VLLM_USE_V1:
    CustomOp.forward(self, hidden_states, output, ...)
else:
    torch.ops.vllm.plamo2_mamba_mixer(hidden_states, output, self.prefix)
```

## 性能声明

pfnet/plamo-2-8b,`--max-model-len 8192 --input-len 1000 --output-len 1000 --no-enable-prefix-caching`,3 iters:本 PR(v1 + FULL_AND_PIECEWISE)avg latency 26.398 s vs main(v0 + piecewise)30.821 s,约 -14.4%。吞吐(sharegpt 1000 prompts):12.00 vs 11.83 req/s,4619.67 vs 4532.89 total tokens/s,2144.02 vs 2104.20 output tokens/s,基本持平。质量:lm_eval gsm8k 5-shot flexible-extract 0.5163 vs 0.5125、strict-match 0.6603 vs 0.6581,无退化。硬件未注明。

## 验证与回归风险

`pytest tests/models/language/generation/test_hybrid.py` 通过;plamo-2-1b 进入 HYBRID_MODELS、V1_SUPPORTED_MODELS 与 FULL_CUDA_GRAPH_MODELS 三个单测列表。风险点:自定义算子与 v1 mamba cache 路径仅对该模型验证;PR 自述可能与 #21467 冲突;移除 transformers `PreTrainedModel` 继承并解除 registry 的 max_transformers_version 限制。

## 关联

前序:#21075、#18218(MambaMixer2 改进)、#23520(#22999 修复)、#21467(潜在冲突,cc @cyang49)。同分类下属 hybrid 模型 CUDA graph 支持系列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
