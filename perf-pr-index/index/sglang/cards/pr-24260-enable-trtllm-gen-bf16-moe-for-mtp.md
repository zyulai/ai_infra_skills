# PR #24260: Enable trtllm-gen BF16 MoE for MTP

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @b8zhong |
| 合并 | 2026-05-09 |
| 分类 | Speculative Decoding | tier: major |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 含 2 个文件:`flashinfer_trtllm.py`、`server_args.py` |
| 链接 | https://github.com/sgl-project/sglang/pull/24260 |

## 一句话结论

解除 MTP draft MoE 对 flashinfer trtllm-gen BF16 后端的禁用,draft MoE 时间提速约 2.5x,accept length 不变。

## 动机

上游 flashinfer#2803 之后,trtllm-gen BF16 MoE 开始支持 DSV3 routing,因此对未量化(unquantized)MoE 层,MTP draft 层也可以用该后端。此前 SGLang 在 `server_args` 的 speculative 解码处理里显式回避:原注释给出的理由是"FlashInfer trtllm moe bf16 只支持 RenormalizeNaive 与 Deepseek routing method,而 draft model 的 routing method 难以判断,且 draft model 的 MoE 层在端到端中不是瓶颈",于是直接用 assert 拒绝 speculative MoE runner 使用 flashinfer_trtllm,并在主 backend 为 flashinfer_trtllm 时把 speculative 后端强制回落到 auto。上游 routing 支持补齐后,这道禁令只剩成本没有收益。

## 优化点

三处联动改动。(1) 删除 `_handle_speculative_decoding` 中的 assert 与 auto-fallback 分支:`speculative_moe_runner_backend` 未显式设置时直接继承 `self.moe_runner_backend`,即主模型选了 flashinfer_trtllm 时 draft MoE 也走同一后端。(2) `fused_experts_none_to_flashinfer_trtllm_bf16` 的激活约束从"仅 silu"放宽为集合 `{silu, relu2}`,并新增 `activation_type = get_activation_type(runner_config.activation)` 传入两处 trtllm MoE 调用(routed 与 bypassed 两个分支),使激活语义显式下传 kernel。(3) `_handle_model_specific_adjustments` 中自动选 `moe_runner_backend="flashinfer_trtllm"` 的条件追加 `quantization is None`,把未量化模型也纳入该后端的适用范围。机制上没有新 kernel,只是把 BF16 trtllm-gen 路径对 draft 模型与未量化配置开放。

## 关键改动

```text
python/sglang/srt/server_args.py: 移除 speculative MoE 对 flashinfer_trtllm 的 assert/fallback;speculative 后端默认继承主后端;quantization is None 纳入适用条件
python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py: BF16 路径激活支持 {silu, relu2},activation_type 传入 trtllm MoE 调用
```

```python
if self.speculative_moe_runner_backend is None:
    self.speculative_moe_runner_backend = self.moe_runner_backend
```

## 性能声明

- draft MoE 时间约 **2.5X speedup**;作者注明这只是端到端的一小部分,所以端到端提升很小("small portion of E2E, that is why the speedup is little")。该 2.5x 对应的微基准条件 PR 未注明(未给出硬件/模型/批量)。
- 端到端对照表(benchmark 具体条件未注明):Before latency 4.835 s / 1024 tokens / accept length 3.230 / 209.77 token/s;After 4.815 s / 1024 tokens / 3.230 / 212.68 token/s——accept length 无变化,速度约 +1.4%。
- 准确性:GPQA on DSV3 NVFP4(`run_eval` 198 examples、repeat 8、thinking-mode deepseek-v3):0.834→0.836,作者标注为 noise。

## 验证与回归风险

验证:GPQA 评估如上;acceptance length 前后一致(3.230),说明 draft 质量未受影响。PR 未列单测或确定性检查。回归风险:原禁令保护的两个假设被移除——draft model routing method 现在依赖 flashinfer#2803 的 DSV3 routing 支持兜底,非 DSV3 routing 的 BF16 MoE draft 模型若不满足 RenormalizeNaive/Deepseek routing 可能出错(运行时才暴露);`speculative_moe_runner_backend` 默认语义变化:此前主 backend 为 flashinfer_trtllm 时 spec 后端回落 auto,现在直接继承,量化模型(如 fp8/nvfp4)的 draft 路径行为随之改变;relu2 激活路径为新增面,此前会被 assert 拒绝。

## 关联

flashinfer#2803(上游前序:trtllm-gen BF16 MoE 支持 DSV3 routing,本 PR 的启用前提)。无正文列出的其他 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
