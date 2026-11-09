# PR #58569: [ROCm][Perf][GLM-5.3-Flash] Remove redundant copy after ragged sparse MLA

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @simondanielsson |
| 合并 | 2026-10-02 |
| 分类 | Attention / Kernel(major) |
| Label | performance, rocm, ready, glm |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,涉及 2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/58569 |

## 一句话结论

让 ragged sparse MLA prefill kernel 原地写入输出,省掉每次 MLA layer 约 5us、每次 forward 约 55us 的冗余拷贝。

## 动机

attention 输出目标是 `[s, h, kv_lora_rank]`(不含 rope 维),而 kernel 此前按 query 形状 `[s, h, kv_lora_rank + rope_dim]` 分配输出,算完后在 host 侧 slice 出前 `kv_lora_rank` 列再 `copy_` 进真正的 output 张量。这一步拷贝纯属格式转换开销,在 GLM-5.3-Flash 这类每 forward 有多个 MLA layer 的模型上逐层累积。

## 优化点

把 slice+copy 下沉进 Triton kernel:给 `_rocm_sparse_attn_prefill_ragged_triton` / `_rocm_sparse_attn_prefill_triton` 增加可选 `out` 参数,调用方(`rocm_sparse_attn_prefill`)直接传入目标 output,不再接收 chunk 后 `copy_`。kernel 内新增 constexpr `OUT_DV`(取 `out.shape[-1]`),store 掩码从 `dim_mask` 改为 `dim_offsets < OUT_DV`,使 head 维上负责 `>= kv_lora_rank` 列的 workgroup 走 diverge 分支、不写 rope 列。`out` 为空时保持旧行为(`empty_like(q)`);传入 `out` 时断言其形状为 `[s, h, kv_lora_rank 或 head_dim]`,支持非连续 view 与不同 out dtype(内含 cast)。

## 关键改动

```text
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py: kernel 增加 OUT_DV 掩码与 out 参数,caller 去掉 copy_
tests/kernels/attention/test_rocm_triton_attn_dsv4.py: 新增 out= 语义的 4 组测试
```

关键片段(kernel store 掩码):

```python
tl.store(
    out_ptr + ... + dim_offsets[None, :] * out_stride_d,
    out,
    mask=head_mask[:, None] & (dim_offsets[None, :] < OUT_DV),
)
```

## 性能声明

PR 正文:每次 MLA layer 约 5us、每次 forward 合计约 55us;相对每次 forward 约 13ms 的量级,作者明确表示在 vllm bench serve 噪声内不可测,故跳过完整 perf sweep,只提供正确性结果与非回归 benchmark(before/after 截图,数字未包含在素材文本中)。正确性:gsm8k 5-shot exact_match,本分支 flexible-extract 0.9181±0.0076 / strict-match 0.9174±0.0076;main 为 0.9219±0.0074 / 0.9196±0.0075。测试条件 PR 未注明完整硬件/精度信息(仓库名含 ROCm/GLM-5.3-Flash 语境)。

## 验证与回归风险

`tests/kernels/attention/test_rocm_triton_attn_dsv4.py` 新增:分配式 `out` 与返回值逐位一致(atol=0)、非连续 view 写入不越界不改邻居元素、out dtype 为 fp32/fp16 时正确 cast、head_dim 全宽 `out` 时 rope 行仍保留;dense fallback 路径签名同步改为接收 `out`。风险面:仅影响 ROCm AITER/triton sparse MLA prefill 路径;`OUT_DV` 掩码改变 kernel 编译特化,需覆盖 nope-only(rope_dim=0)、512+64(V3.2)与 448+64(V4)等 head 布局(测试已参数化)。

## 关联

同区域:GLM/DeepSeek 系 MLA 稀疏 attention 的 ROCm 路径优化(同分类 Attention / Kernel);PR 未注明具体前序/后续 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
