# PR #43050: feat: MLA prefill enable FA4 fp8 output

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @carlyou |
| 合并 | 2026-06-16 |
| 分类 | Quantization(tier: major) |
| Label | performance, rocm, ready, ci/build, v1, nvidia |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 全文可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/43050 |

## 一句话结论

MLA prefill 在 FA4 后端把 static FP8 输出量化直接融合进 attention kernel,省掉独立的 post-quant kernel(限 Blackwell SM100/110)。

## 动机

对应 issue #35792 "FlashAttn x Static FP8" 的收尾:MLA 模型使用 static FP8 输出量化(kFp8StaticTensorSym)时,原路径是 FA4 prefill kernel 先写出 BF16 结果,再由单独的 QuantFP8 kernel 量化拷贝到 quant_output,多一次全量读写的 post-quant pass。上游 flash-attention#135 已让 FA4 支持在 kernel 内直接写 e4m3 输出,本 PR 把该能力接入 vLLM 的 MLA prefill 抽象层。

## 优化点

核心机制是给 MLA prefill 后端加一个量化输出能力协商接口:`MLAPrefillBackend.supports_quant_output(quant_key)` 默认返回 False,FA4 后端覆写为"FA4 版本 + vLLM 自带 FA + SM100/SM110 + static per-tensor FP8"四条件同时满足才融合。满足时 `mla_attention.forward_impl` 直接把 `quant_output` 缓冲与 `output_scale` 传给 `forward_mha`,FA4 kernel 就地写 FP8,后续对 MHA 段的 post-quant 拷贝被跳过(仅剩 decode/MQA 段如有需要再量化);不满足时走原 BF16 + 独立量化路径。接口上 `run_prefill_new_tokens` 与基类 `forward_mha` 统一新增 `out`/`output_scale` 参数,FlashInfer、TokenSpeed、TRTLLM ragged、ROCm AITER 等其他 prefill 后端先补签名并显式 assert 不支持,保证兼容。融合仅限非 chunked-context 路径且 DCP world size ≤ 1。

## 关键改动

```text
vllm/v1/attention/backends/mla/prefill/base.py: supports_quant_output() 能力门 + run_prefill_new_tokens 增 out/output_scale
vllm/v1/attention/backends/mla/prefill/flash_attn.py: FA4 支持 SM100/110 + static FP8 的融合判定
vllm/model_executor/layers/attention/mla_attention.py: forward_impl 选择融合输出,跳过 MHA 段 post-quant
vllm/vllm_flash_attn/flash_attn_interface.py: flash_attn_varlen_func 新增 output_scale(仅 FA4)
tests/v1/attention/test_mla_prefill_quant_output.py: 能力门与数值等价性测试
benchmarks/attention_benchmarks/configs/mla_fa4_fp8_output.yaml: fused vs post-quant 微基准配置
```

核心判定代码:

```python
def supports_quant_output(self, quant_key: "QuantKey") -> bool:
    device_capability = current_platform.get_device_capability()
    return (
        self.vllm_flash_attn_version == 4
        and self._is_vllm_fa
        and device_capability is not None
        and device_capability[0] in (10, 11)
        and quant_key == kFp8StaticTensorSym
    )
```

## 性能声明

PR 正文未含 kernel 微基准数字(新增的 benchmark.py 支持 `--fp8-output-scale`/`--fuse-quant-op` 对比 fused 与 post-quant,但结果未贴出)。正文给出 B200 上的 gsm8k 端到端 eval(DeepSeek-Coder-V2-Lite-Instruct-FP8,TP1,max-model-len 4096,cudagraph NONE,`fuse_attn_quant: true`,5-shot 1319 题,CUDA 13.0/PyTorch 2.11.0+cu130):main 为 accuracy 0.785、587.03 s、2.25 Q/s、287.5 tok/s;本分支为 accuracy 0.769、invalid 0.011、600.43 s、2.20 Q/s、278.5 tok/s——该 eval 中精度与吞吐均略低于 main,PR 未对此附分析。Test Plan 提到在 H100 与 B200 上做过 FP8 benchmark 与 eval,但 H100 数字未在正文给出。

## 验证与回归风险

新增 `tests/v1/attention/test_mla_prefill_quant_output.py`:覆盖基类默认不支持、FA4 各 (版本, 架构, quant_key) 组合的门控真值表、签名兼容(Liskov)检查,以及 SM100/SM110 上融合 FP8 写与 BF16+独立量化参考的数值等价(rtol=0.125,要求 >90% 元素落入相同 FP8 bucket)。回归风险:`forward_mha`/`run_prefill_new_tokens` 签名变更触及所有 MLA prefill 后端(FlashInfer/TokenSpeed/TRTLLM/ROCm AITER),靠新增参数默认 None 保持向后兼容;chunked-context 路径遇 output_scale 会 assert 失败,属于显式不支持而非静默错误;ROCm 上游 FA 走 `return_attn_probs` 分支并断言禁用融合。误开融合会改变输出精度(e4m3 量化提前),由 quant_key 门控约束在 static FP8 模型上。

## 关联

上游依赖: flash-attention#135(FA4 FP8 output);总纲 issue: vllm#35792(FlashAttn x Static FP8)。同区域相关:本 PR 补齐 MLA prefill 的 quant-output 抽象,后续其他 prefill 后端(如 FlashInfer)可覆写 `supports_quant_output` 接入。
