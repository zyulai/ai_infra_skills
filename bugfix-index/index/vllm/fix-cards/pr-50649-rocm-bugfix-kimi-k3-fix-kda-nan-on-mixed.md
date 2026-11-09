# PR #50649: [ROCm][Bugfix] Kimi-K3 Fix KDA NaN on mixed batches and racy autotune config

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50649](https://github.com/vllm-project/vllm/pull/50649) |
| 作者 | kliuae |
| 合并日期 | 2026-08-05 |
| labels | bug, k3, kimi, ready, rocm |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
ROCm Kimi-K3 KDA 两处正确性 bug:混合 decode+prefill 批的 decode 走 chunk kernel 返回 NaN,改为分流到 `fused_recurrent_kda`;`recompute_w_u_fwd_kernel` autotune 会选中竞争配置产生 ~1e38 垃圾值,从 autotune 中剔除 `num_stages=4`。

## 问题现象
Kimi-K3 在 ROCm KDA 路径上有两个正确性 bug:1) 混合 decode+prefill 的非投机批中,decode 序列被送入 KDA chunk kernel `chunk_kda_with_fused_gate`,该 kernel 把每个 decode 当成长度为 1 的序列,对它们返回 NaN;2) `recompute_w_u_fwd_kernel` 的 autotune 可能选中 `num_stages=4`(尤其搭配 `num_warps=4`)的竞争配置,静默产生约 1e38 量级的垃圾值并污染下游计算。按正文测试计划(`moonshotai/Kimi-K3` + gsm8k,MI355X,TP=8)验证:修复前 5-shot gsm8k 仅 0.8901,修复后 5-shot 升至 0.9689(flexible-extract)/0.9697(strict-match),20-shot 达 0.9697/0.9704。

## 根因
正文披露两条:1) chunk kernel `chunk_kda_with_fused_gate` 把 decode 当 length=1 序列处理并返回 NaN,故混合批中的 decode 不能走该 kernel;2) `recompute_w_u_fwd_kernel` 在 `num_stages=4`(尤其与 `num_warps=4` 组合)时对该 kernel 是有竞争(racy)的配置。diff 注释补充机制(由 diff 推断):`u` 循环 trip count 为 2(V/BV),其上再叠 4 级流水会发射第三个无消费者的异步拷贝进 LDS,被 `w` 循环复用;在 gfx950 上 `num_warps=4` 时发生竞争,批一旦达到 4096 tokens,`u` 便带回非确定的 O(1e38) 垃圾值。

## 修复方案
- 新增 `vllm/models/kimi_k3/amd/kda.py` 的 `KimiK3DeltaAttention`(ROCm 专用 KDA 层,要求 `use_full_rank_gate`):其 `_forward` 在非投机混合批(decode 在前)时把 decode token 切出,送 `fused_recurrent_kda` 递推 kernel,prefill 尾部才交给 `chunk_kda_with_fused_gate`,输出再按原顺序 `torch.cat` 合并:

```python
core_attn_out_decode, _ = fused_recurrent_kda(
    q=q_ns[:, :nd_tok], ..., cu_seqlens=non_spec_query_start_loc[: m.num_decodes + 1],
    ssm_state_indices=non_spec_state_indices_tensor[: m.num_decodes])
...
core_attn_out_non_spec = torch.cat(
    [core_attn_out_decode, core_attn_out_non_spec], dim=1)
```

- `vllm/models/kimi_k3/amd/linear.py`:KDA 层按 `use_full_rank_gate` 分派——Kimi-K3 用新的 `KimiK3DeltaAttention`,Kimi-Linear 保留共享低秩 gate 实现;
- `vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py`:

```diff
-        for num_stages in [2, 3, 4]
+        for num_stages in _RECOMPUTE_W_U_NUM_STAGES
```

其中 `_RECOMPUTE_W_U_NUM_STAGES = [2, 3]`,即从 autotune 空间剔除 `num_stages=4`,使搜索不再落在出错配置上。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 ROCm Kimi-K3 KDA 路径——混合批的 decode/prefill 分流逻辑、KDA 层类型分派、`recompute_w_u_fwd_kernel` 的 autotune 配置空间(移除 num_stages=4 后已缓存该配置的用户需重选);Kimi-Linear 共享层与 NVIDIA 路径不受影响。gsm8k 分数显著回升且 20-shot 无回退。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
