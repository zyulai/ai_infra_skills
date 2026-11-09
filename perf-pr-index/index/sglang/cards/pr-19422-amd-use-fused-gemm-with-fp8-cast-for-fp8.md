# PR #19422: [AMD] Use fused GEMM with FP8 cast for FP8 prefill

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @1am9trash |
| 合并 | 2026-02-26 |
| 分类 | Quantization (tier: major) |
| Label | amd, run-ci |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/19422 |

## 一句话结论

AMD gfx95 上用 aiter fused kernel 把 kv_b_proj GEMM、nope/v 切分与 k_pe 拼接合一并直接输出 FP8 k/v,FP8 prefill 的 TTFT 提升约 2-3%。

## 动机

使用 FP8 prefill 时,q、k、v 张量需要在 prefill attention kernel 之前做一次独立的 element-wise BF16→FP8 cast,引入额外开销;同时 `kv_indices`(`torch.arange`)在每个 layer 的 forward 中重复创建。

## 优化点

两条路径:(1) 当 `_use_fp8_prefill_attn`(env `SGLANG_AITER_FP8_PREFILL_ATTN`,默认 True)且 `_use_aiter_gfx95`、`kv_b_proj.weight.dtype == torch.uint8`(MXFP4 主模型权重)时,给 `kv_b_proj` 传入 5 元组 `(x, y, S1, S2, out_dtype)`,走 aiter Triton kernel `fused_gemm_afp4wfp4_split_cat`,把 GEMM、nope/v 切分(split)与 k_pe 拼接(cat)融合为单 kernel,直接以 FP8 输出 k 和 v,免去独立 cast;`quark_w4a4_mxfp4.apply_weights` 相应支持该 5 元组格式并返回 `(k, v)`。(2) `kv_indices` 从 per-layer forward 移到 `init_forward_metadata`,按 `extend_seq_lens.sum()` 每 batch 只算一次,存入 `ForwardMetadata.fp8_prefill_kv_indices`。另外 q/k/v 的 cast 从 `.float().to(fp8)` 简化为 `.to(fp8)`(q 在 RoPE 之后 cast)。非 MXFP4 权重(FP8/BF16,如 MTP draft 模型)仍走原 `view/split/concat` 路径并逐个 cast。

## 关键改动

```text
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py: forward_normal_prepare 新增 fused_gemm_split_cat 分支与 _use_fp8_prefill_attn 开关
python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py: apply_weights 支持 (x, y, S1, S2, out_dtype) 5 元组并调用 fused_gemm_afp4wfp4_split_cat
python/sglang/srt/layers/attention/aiter_backend.py: fp8_prefill_kv_indices 移入 init_forward_metadata;cast 简化
```

## 性能声明

- TTFT 提升约 2-3%;条件:input length 70k、output length 200、batch size 1-16。具体 GPU 型号 PR 正文未注明(代码路径限定 aiter gfx95 / ROCm)。
- 精度(2000 题 GSM8K、198 例 GPQA repeat 8):fp8 Prefill 得 GSM8K 0.946,GPQA mean 0.669 / max 0.697;PR 未给出改动前的对照数字。

## 验证与回归风险

验证:gsm8k(`benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000`)与 GPQA(`sglang.test.run_eval`)命令见正文;未新增单元测试,checklist 各项合并时未勾选。风险面:仅 DeepSeek MLA 的 MHA forward、MXFP4 kv_b_proj 权重与 gfx95 aiter 环境进入新 fused 路径;FP8/BF16 权重(MTP draft)仍走旧路径;`kv_indices` 缓存进 metadata 后依赖 `init_forward_metadata` 与 forward 的 batch 一致性;quark scheme 的 tuple 输入协议扩展(2/3/5 元组)可能影响其他调用方。

## 关联

同区域:aiter attention backend 的 FP8 prefill attention 路径、quark W4A4 MXFP4 量化方案、DeepSeek MLA prefill fused quant 链路(`fused_rms_fp8_group_quant` 等);pack 内未给出其他具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
