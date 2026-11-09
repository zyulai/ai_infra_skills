# PR #60102: [ROCm][DSv4.1][Perf] Route gfx942 sparse MLA prefill to AITER pa_prefill_sparse

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @frida-andersson |
| 合并 | 2026-11-02 |
| 分类 | Attention / Kernel \| tier: major |
| Label | rocm, ready, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明（diff 完整呈现） |
| 链接 | https://github.com/vllm-project/vllm/pull/60102 |

## 一句话结论

gfx942 ≥512 行 sparse MLA prefill 走 AITER；MI325X TP4 TTFT 降 11.8%、输出 tok/s +26.7%。

## 动机

PR 正文 Summary 说明，在 gfx942 上，fp16/bf16 sparse MLA prefill 且 query 行数至少 512 时改走 AITER `pa_prefill_sparse`，更短调用仍留在 in-tree kernel；import 与 "Using AITER pa_prefill_sparse for sparse MLA prefill on gfx942" 日志仅在 kernel 真正 launch 时发生。Gating 条件：gfx942、AITER enabled、`q` 为 fp16 或 bf16、`kv.dtype == q.dtype`、`q.shape[0] >= 512`（`_GFX942_AITER_PA_PREFILL_SPARSE_MIN_QUERIES`）。512 cutoff 来自一块 MI325、H=16、D=512、bf16 对 in-tree ragged prefill 的对比。

## 优化点

在 `rocm_aiter_mla_sparse.py` 的 gfx950 OPUS 尝试之后，gfx942 gate 通过时调用 `pa_prefill_sparse`（单 KV pool）；caller output 与 `q` 匹配时作为 `out` 传入，较窄输出则从 kernel 结果 copy。gate 用 `get_cdna_version() == 3`、`is_mla_enabled()`、行数、dtype 与 kv dtype 一致性判断（见下方摘录）。`pa_prefill_sparse` 不切分 KV，512 行以下仍走 in-tree ragged prefill（同样不切分）；DeepSeek-V4.1 decode 用 `rocm_sparse_attn_decode`，不进入此函数；2 行与 16 行 shape 不走 `pa_prefill_sparse`。测试把 dense Triton fallback 的 getter stub 掉以留在 in-tree，并把 `get_cdna_version()` pin 到物理芯片再 patch `on_gfx942()`，使 MI350 不误入 `pa_prefill_sparse`。

## 关键改动

```text
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py: 新增 _GFX942_AITER_PA_PREFILL_SPARSE_MIN_QUERIES = 512、_can_use_aiter_pa_prefill_sparse() gate 与 _rocm_sparse_attn_prefill_aiter()，在 rocm_sparse_attn_prefill 中 gate 通过时调用 pa_prefill_sparse
tests/kernels/attention/test_rocm_triton_attn_dsv4.py: 新增 gate 字面值参数化测试与 routing 测试，并 stub dense Triton fallback 的 getter
tests/v1/attention/test_sparse_mla_backends.py: 长 BF16 prefill 测试先 pin get_cdna_version() 到物理芯片
```

```python
def _can_use_aiter_pa_prefill_sparse(q: torch.Tensor, kv: torch.Tensor) -> bool:
    """Cheap gate. The AITER import runs only after this passes."""
    from vllm._aiter_ops import rocm_aiter_ops
    return (
        get_cdna_version() == 3
        and rocm_aiter_ops.is_mla_enabled()
        and q.shape[0] >= _GFX942_AITER_PA_PREFILL_SPARSE_MIN_QUERIES
        and q.dtype in (torch.bfloat16, torch.float16)
        and kv.dtype == q.dtype
    )
```

## 性能声明

正文 DeepSeek-V4.1-Flash 表条件：Nightly `0.31.1rc1.dev1+gbb87d227d`、4× MI325X、TP4、spec off、no FlyDSL QuickReduce、no decoder replay、Seed 7；为 Oct 7 gate=1024 时的运行，未针对本 commit 重跑（正文注明各行长度都 >512，prefill kernel 相同，on arm 打了 AITER 日志）。Median TTFT (ms) 与 Output tok/s（→ 表示 in-tree → AITER）：8192/128/2 为 1121.58 → 988.77（−11.8%）与 73.25 → 92.81（+26.7%）；8192/128/8 为 3876.07 → 3453.47（−10.9%）与 144.75 → 157.37（+8.7%）；20480/128/2 为 2648.53 → 2228.60（−15.9%）与 50.96 → 58.26（+14.3%）；65536/128/2 为 7966.74 → 7082.31（−11.1%）与 21.30 → 23.56（+10.6%）；262144/1024/4 为 62228.10 → 56206.90（−9.7%）与 36.68 → 40.09（+9.3%）。本 commit 只重跑 8192/128：Median TPOT 13.88 ms @conc2（1024 gate 时 13.88）、23.83 ms @conc8（23.89）。512 cutoff 表（MI325、H=16、D=512、bf16，AITER/in-tree 时间比）：512 行 top-k 128/512/2048 = 0.64/0.49/0.46，1024 行 = 0.43/0.32/0.30。GLM-5.3-Flash（`zai-org/GLM-5.3-Flash`、4× MI325X、TP4、spec off、4 prompts、8192 in/128 out、conc1、seed 7、Oct 7 gate 之前）：Output tok/s 65.53 → 66.66（+1.7%），Mean TTFT 411.65 → 384.89（−6.5%），Median TTFT 506.14 → 469.44（−7.3%）；GSM8K 1319 题 5-shot strict/flexible 0.9227 ± 0.0074 → 0.9257 ± 0.0072。

## 验证与回归风险

PR 引入/依赖的测试：Test plan 为 `python3 -m pytest -q tests/kernels/attention/test_rocm_triton_attn_dsv4.py::test_sparse_attn_prefill_preserves_dense_triton_fallback`，结果 "1 passed, 14 warnings in 3.80s"；diff 另新增 gate 与 routing 的参数化测试。正文 "Depends on" 标注 ROCm/aiter#6002：符号只在 gate 后 import，缺失时 eligible call 会抛错，短 call 不 import。回归风险：PR 未明确说明；正文注明 DeepSeek-V4.1 decode 用 `rocm_sparse_attn_decode`、不进入此函数，因此 "does not show a GLM decode change"；可能受影响路径为 gfx942 fp16/bf16 sparse MLA prefill ≥512 行及其他 CDNA 设备（测试已 pin 规避）。

## 关联

正文 "Depends on" 标注 https://github.com/ROCm/aiter/pull/6002（外部仓库 PR ROCm/aiter#6002）；未给出 vllm 仓库内的其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
