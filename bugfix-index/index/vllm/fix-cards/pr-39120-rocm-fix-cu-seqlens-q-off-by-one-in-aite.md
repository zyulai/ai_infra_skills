# PR #39120: [ROCm] Fix cu_seqlens_q off-by-one in AITER FA speculative decode path

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39120](https://github.com/vllm-project/vllm/pull/39120) |
| 作者 | Bortlesboat |
| 合并日期 | 2026-04-19 |
| labels | rocm, ready, v1 |
| 热度 | 👍 1 · 💬 9 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论

AITER FA 投机解码路径 `cu_seqlens_q` 少切一个元素,破坏累积长度数组约定;改为切 `num_decodes + 1` 个并修正 `descale_shape`。

## 问题现象

触发条件:`AiterFlashAttentionImpl` 的投机解码路径(`decode_max_query_len > 1`)。正文指出此时 `cu_seqlens_q` 被切成 `query_start_loc[:num_decodes]`,只有 `num_decodes` 个元素,而 unified_attention 需要含前导 0 的 `num_seqs + 1` 个条目;正确的 `[:num_decodes + 1]` 写法在 33 行之后的 fallback 路径(1228-1229 行)已经在用。正文未描述具体用户可见现象(未附错误日志),仅提到此前一次 CI 失败只要求该行补上匹配的 ruff format 空白修改,本 follow-up 提交应用了该 diff。

## 根因

正文披露:投机解码路径中 `cu_seqlens_q` 以 `query_start_loc[:num_decodes]` 切片得到 `num_decodes` 个元素,但 unified_attention 的 `cu_seqlens_q` 是需要 `num_seqs + 1` 个条目(含前导 0)的累积长度数组,切片因此少 1。同时 `descale_shape` 在错误长度的切片上计算 `shape[0] - 1`,得到 `num_decodes - 1` 而非 `num_decodes`。正确模式在同函数 fallback 路径(第 1228-1229 行)已存在,上游 AITER unified_attention 实现也确认了正确切片方式。

## 修复方案

修改 `vllm/v1/attention/backends/rocm_aiter_fa.py` 的 `forward`:`cu_seqlens_q` 改为 `query_start_loc[: num_decodes + 1]`;`descale_shape` 不再从切片长度倒算,直接用 `num_decodes`,与第 1232 行 fallback 一致。

```diff
-                        cu_seqlens_q=attn_metadata.query_start_loc[:num_decodes],
+                        cu_seqlens_q=attn_metadata.query_start_loc[: num_decodes + 1],
```

```diff
-                        attn_metadata.query_start_loc[:num_decodes].shape[0] - 1,
+                        num_decodes,
```

## 影响与回归风险

- 影响窗口:正文未说明受影响版本范围(仅说明正确写法已存在于同文件 fallback 路径)。
- 回归面:仅 `rocm_aiter_fa.py` 中 AITER FA 投机解码分支的两处参数计算;fallback 路径与其余 attention 路径不变。风险面为 ROCm AITER 投机解码的 attention 调用入参。

## 关联

正文引用上游 AITER unified_attention 实现佐证:https://github.com/ROCm/aiter/blob/bc5ea32c19c604cda2f4781a97cb04d5fc494543/aiter/ops/triton/_triton_kernels/attention/unified_attention.py#L426

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
