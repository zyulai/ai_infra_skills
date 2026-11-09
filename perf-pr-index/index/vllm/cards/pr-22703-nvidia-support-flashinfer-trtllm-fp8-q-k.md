# PR #22703: [NVIDIA] Support Flashinfer TRTLLM FP8-q/kv NVFP4-out Attention Kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @elvischenv |
| 合并 | 2025-08-22 |
| 分类 | Quantization (tier: major) |
| Label | performance, rocm, tpu, ready, ci/build, v1, llama |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 可用但被截断(15 个文件未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/22703 |

## 一句话结论

在 attention+FP8-quant 融合(#21716)之上新增 attention 输出直接量化为 NVFP4 的图融合,配合 TRTLLM-gen attention kernel 省掉单独的 FP4 量化 kernel。

## 动机

TRTLLM-gen attention 内核支持把 attention 输出直接写为 NVFP4(FP8 Q/FP8 KV 输入),此前 #21716 只做了 attention+FP8-output-quant 融合;NVFP4 输出路径还需要把 attention 输出再过一个 `scaled_fp4_quant` kernel,多一次全量张量读写。在 torch.compile 图中把该量化 op 融合进 attention 自定义 op,可直接消掉这一步,对 FP4 checkpoint(如 Llama-4-Scout-FP4)的 serving 吞吐有明显收益。

## 优化点

机制上是图融合 pass 的扩展:`AttentionStaticQuantPattern` 重构为 `AttentionQuantPattern` 抽象基类,派生出既有 `AttentionFp8StaticQuantPattern` 与新增 `AttentionNvfp4QuantPattern`——后者匹配 "attention → reshape → scaled_fp4_quant" 模式,替换为 attention op 直接携带 `output_block_scale` 输出 FP4 数据与 block scale,量化 op 从图中删除。配套:`unified_attention_with_output` 自定义 op 增加 `output_block_scale` 参数并纳入 mutates_args;`fused_output_quant_supported` 接口从 (dtype, static, group_shape) 改为接收 `QuantKey`;`QuantKey` 从 fusion.py 移入 quant_utils 并改为 dtype+`ScaleDesc` 描述(新增 `kNvfp4Quant` 到 QUANT_OPS);其余 attention 后端(flash_attn、xformers、MLA common、ROCm FA、dual-chunk、DFA)统一在 `output_block_scale` 非 None 时抛 NotImplementedError,明确不支持融合;kernel benchmark 脚本增加 (FP8 q, FP8 kv, FP4 out) 组合与 `o_sf_scale`/FP4Tensor 输出分配。

## 关键改动

```text
vllm/compilation/fusion_attn.py: AttentionQuantPattern 基类 + AttentionNvfp4QuantPattern 图融合
vllm/attention/layer.py: unified_attention_with_output 增加 output_block_scale 参数
vllm/attention/backends/abstract.py: fused_output_quant_supported 改为 QuantKey 接口,forward 增加 output_block_scale
vllm/model_executor/layers/quantization/utils/quant_utils.py: QuantKey/ScaleDesc 迁移并新增 kNvfp4Quant(diff 未直接展示)
```

融合替换核心:

```python
at2 = auto_functionalized(ATTN_OP, query=q, key=k, value=v,
                          output=output_attn, layer_name=self.layer_name,
                          output_scale=input_scale,
                          output_block_scale=output_scale_view)
```

## 性能声明

E2E:`nvidia/Llama-4-Scout-17B-16E-Instruct-FP4`,640 请求,最大并发 128(硬件未注明)。

- `--kv-cache-dtype fp8` + enable_attn_fusion:输出吞吐 5039.38 tok/s,Mean TPOT 24.74 ms,Mean TTFT 658.54 ms;对照 main 同配置 3986.50 tok/s / 31.36 ms / 751.94 ms,即吞吐约 +26%、TPOT 约 −21%。
- GSM8K(5-shot):fp8+fusion flexible-extract 0.9014 / strict-match 0.8863,与 main fp8(0.8969/0.8817)和 main auto(0.9037/0.8908)相当;kv=auto 下 PR 与 main 基本持平(4556.72 vs 4571.39 tok/s)。

## 验证与回归风险

测试:`tests/kernels/attention/test_flashinfer_trtllm_attention.py` 增加 (None, fp8, None) 与 (fp8, fp8, fp4) 组合,44 passed, 4 skipped;`tests/compile/test_fusion_attn.py::test_attention_quant_pattern` 12 passed。NVFP4 输出与 baseline 的容差放宽到 rtol 3e-1(prefill 4e-1)/atol 1e0。风险面:所有 attention 后端 forward 签名统一加 `output_block_scale`(ABI 变更,下游 fork 需同步);融合只对 `fused_output_quant_supported(QuantKey)` 返回 True 的实现触发,其他后端传入 block scale 会直接抛异常;prefill 混合 QKV dtype 用例跳过(限制)。

## 关联

前序:#21716(attention + FP8-quant 融合);同区域:pr-24197(trtllm fp8 kv prefill)、pr-21078(trtllm-gen MLA decode)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
