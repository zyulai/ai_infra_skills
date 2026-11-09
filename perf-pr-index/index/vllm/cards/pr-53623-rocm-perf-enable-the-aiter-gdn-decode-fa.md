# PR #53623: [ROCm][Perf] Enable the AITER GDN decode fast path for flat qkvz layouts

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mjkvaak-amd |
| 合并 | 2026-09-19 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/53623 |

## 一句话结论

让 flat qkvz 布局模型也走 AITER GDN decode fast path，Qwen3.8 输出 +2.1~3.2%。

## 动机

`_forward_core_rocm` 当前把 AITER GDN decode fast path 门控在 `self.gqa_interleaved_layout`，只有 Qwen3-Next 能走，flat-layout 模型（如 Qwen3.5、Qwen3.8）回退 generic path。#42880 加了这个 guard，因为 fused reshape+conv kernel 只理解 Qwen3-Next 的 packing，会从 flat tensor 读错列。ROCm/aiter#3251 新增了 `qkvz_layout` 参数，同时处理两种布局。

## 优化点

传 `qkvz_layout` 给 AITER 内核并去掉 layout guard：`__init__` 新增 `self.qkvz_layout = "interleaved" if gqa_interleaved_layout else "flat"`，`_forward_core_rocm` 的 decode-only 判定删掉 `self.gqa_interleaved_layout` 条件，`_forward_core_decode_aiter` 调用时传入 `qkvz_layout=self.qkvz_layout`。该参数只选择源索引映射，输出保持 canonical order，下游 `fused_rearrange_sigmoid_gated_delta_rule` 不变；`"interleaved"` 仍为默认故 Qwen3-Next bit-identical。旧 AITER 构建会 `TypeError`，故新增 `gdn_fused_conv_supports_qkvz_layout()` 探测签名一次（仿 `fused_moe_supports_gate_mode()`），flat layouts 在探测失败时保留 guard。fast path 在 generic block 前返回，每个 GDN layer 每 step 的 4 次 kernel launch 变 2 次：`core_attn_out.zero_()` 与 `z_out[:] = z` 被纯移除（AITER kernel 写两个 buffer 含 padding rows）。另把 `vllm/model_executor/layers/mamba/gdn` 加入 Kernels Mamba 的 `source_file_dependencies`。

## 关键改动

```text
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py: 新增 self.qkvz_layout；_forward_core_rocm 去掉 gqa_interleaved_layout guard；_forward_core_decode_aiter 传 qkvz_layout；新增 gdn_fused_conv_supports_qkvz_layout() 能力探测。
tests/kernels/mamba/test_gdn_rocm_layout_dispatch.py（新增 124 行）: host-side CPU 测试，覆盖 flat dispatch、Qwen3-Next non-regression、必须走 generic 的 batch shapes 与 capability probe。
.buildkite/test-amd.yaml、.buildkite/test_areas/kernels.yaml: 把 vllm/model_executor/layers/mamba/gdn 加入 Kernels Mamba 的 source_file_dependencies。
```

```text
        self.qkvz_layout = "interleaved" if gqa_interleaved_layout else "flat"

        if (
            attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
```

## 性能声明

Serving benchmark：`amd/Qwen3.8-2.4T-A95B-Quark-MXFP4`，8x MI355X，TP8，固定 input/output lengths（`--random-range-ratio 0`），one run per point，对比 a stock build of the same commit。Output tok/s / Mean TPOT (ms)：ISL 1024 / OSL 1024 / Conc 4 = 250.6 → 258.6（+3.2%）/ 15.66 → 15.16（-3.2%）；Conc 16 = 674.0 → 688.3（+2.1%）/ 22.95 → 22.45（-2.1%）；Conc 64 = 1,560.7 → 1,594.7（+2.2%）/ 38.82 → 37.94（-2.3%）。GSM8K 精度（5-shot，seed 4321，`batch_size=128`）：Stock (main) flexible-extract 0.9318 ± 0.0069 / strict-match 0.9333 ± 0.0069，This PR 0.9348 ± 0.0068 / 0.9363 ± 0.0067，"No accuracy regression."。(Decode) trace：GDN decode window **73.7 ms → 38.0 ms**；`core_attn_out.zero_()`（3933 / 16.3 ms）、`direct_copy_kernel`（3920 / 17.0 ms）、`_causal_conv1d_update_kernel`（3933 / 16.7 ms）、`fused_recurrent_gated_delta_rule_packed_decode_kernel`（3933 / 23.7 ms）变为 absent，新增 `_reshape_causal_conv1d_update_single_token_kernel`（3933 / 17.3 ms）与 `fused_rearrange_sigmoid_gated_delta_rule_update_kernel`（3933 / 20.7 ms）。

## 验证与回归风险

验证：`pytest tests/kernels/mamba/test_gdn_rocm_layout_dispatch.py -q`，14 passed；host-side CPU 测试（无需 GPU / AITER）覆盖 flat dispatch with/without kernel support、Qwen3-Next non-regression、batch shapes that must stay generic、capability probe。Accuracy 用 #42880 的 lm_eval invocation，performance 用 `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4` on 8x MI355X 的 decode traces。风险：旧 AITER 构建无 `qkvz_layout` 会 `TypeError`，靠 `gdn_fused_conv_supports_qkvz_layout()` 探测并为 flat layouts 保留 guard；仅 decode-only（无 spec、无 prefill）批次走 fast path，其余（prefill / no decode / spec decode）仍走 generic；Qwen3-Next bit-identical。

## 关联

#42880（加了 layout guard，本 PR 去掉；Accuracy 沿用其 lm_eval invocation）、ROCm/aiter#3251（新增 `qkvz_layout` 参数）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
