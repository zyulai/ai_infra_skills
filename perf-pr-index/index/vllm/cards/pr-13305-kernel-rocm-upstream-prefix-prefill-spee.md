# PR #13305: [Kernel][ROCM] Upstream prefix prefill speed up for vLLM V1

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @maleksan85 |
| 合并 | 2025-04-23 |
| 分类 | Attention / Kernel |
| Label | rocm, ready, ci/build, v1 |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面);diff 因超长截断,另有 2 个文件未收录 |
| 链接 | https://github.com/vllm-project/vllm/pull/13305 |

## 一句话结论

重写 ROCm Triton prefix-prefill attention 内核,以向量化加载、在线 softmax 重构与更宽的 autotune 空间加速 V1 prefill。

## 动机

vLLM V1 在 AMD GPU 上 prefix prefill 偏慢:原 `_fwd_kernel` 的 context 循环对每个 KV block 都做带 mask 的 K/V cache 加载,online softmax 每步重复做额外的 p_scale/acc_scale 缩放与除法,且 autotune 可选配置谱窄,难以按 shape 选到最优配置。PR 正文给出的改进方向是 context 循环向量化、online softmax 计算重构、kernel 结构化以便 autotune 按 shape 选优,并补充新的 unrolling/staging 配置。

## 优化点

从 diff 看,新 `_fwd_kernel` 有四处机制性变化。其一,context 循环以 `BLOCK_SIZE` 为步长组织,通过 `tl.range(..., loop_unroll_factor=num_unroll_cache)`(request 循环对应 `num_unroll_request`)引入编译期可控展开;`block_size`、`num_queries_per_kv`、`stride_k_cache_bl` 等改为 `tl.constexpr` 便于按 shape 特化。其二,当整个 KV block 都落在 context 范围内且 head size 无 padding(`start_n + BLOCK_SIZE <= cur_batch_ctx_len` 且 `BLOCK_DMODEL == BLOCK_DMODEL_PADDED`)时,K/V cache 走无 mask 的 `tl.load` 快路径,其余情况才回退到 mask 加载。其三,online softmax 重构为 `m_ij = tl.maximum(m_i, tl.max(qk))` 后 `acc` 只乘 `alpha`、`l_i` 初始化为 1.0 并在循环外统一 `acc / l_i`,减少每块的缩放次数。其四,文件头保留了一个注释掉的 autotune 配置示例(含 `kpack`/`waves_per_eu` 等 ROCm 参数),注明其提升吞吐但显著增加 triton 3.2 首次调用延迟。

## 关键改动

```text
vllm/attention/ops/prefix_prefill.py: 重写 _fwd_kernel(context 循环向量化 + softmax 重构 + unroll 参数化)
```

```python
for start_n in tl.range(0, cur_batch_ctx_len, BLOCK_SIZE, \
                        loop_unroll_factor=num_unroll_cache):
    ...
    if start_n + BLOCK_SIZE > cur_batch_ctx_len or \
        BLOCK_DMODEL != BLOCK_DMODEL_PADDED:
        k_load = tl.load(K_cache + off_k, mask=..., other=0.0)
    else:
        k_load = tl.load(K_cache + off_k)
```

## 性能声明

PR 正文未提供具体 benchmark 数字,仅定性描述 "Speed up prefix prefill with vLLM V1 on AMD GPU",并链接 AMD ROCm 文档的 Triton kernel tuning 指南;性能数字 PR 未注明。

## 验证与回归风险

`prefix_prefill.py` 是 vLLM 通用的 Triton attention ops(并非 ROCm 专属文件),改动覆盖所有走该 kernel 的平台与路径,需关注非 ROCm 平台、fp8 KV cache 分支、sliding-window 分支及 alibi 变体的数值一致性;pack 中 diff 被截断(2 个文件未收录),autotune/launcher 侧的完整改动以 PR 页面为准。

## 关联

无同区域 PR 记录(vLLM V1 早期 ROCm attention kernel 优化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
