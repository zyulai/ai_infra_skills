# PR #31705: [DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31705](https://github.com/sgl-project/sglang/pull/31705) |
| 作者 | alisonshao |
| 合并日期 | 2026-07-19 |
| labels | high priority, deepseek, run-ci, bypass-fastfail |
| 热度 | 👍 1 · 💬 3 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | crash |

## 一句话结论
DP attention + breakable prefill CUDA graph 下 idle rank 被填充成 dummy EXTEND,metadata 却按 stale IDLE 构建,触发 sparse-prefill 断言崩溃;`_get_logical_forward_mode` 改为 EXTEND 时返回 EXTEND。

## 问题现象
`test_deepseek_v4_flash_fp4_b200.py`(base-c b200 deepep,BCG recipe: TP4/DP4/DeepEP,`--enable-mixed-chunk --cuda-graph-backend-prefill breakable`)在 idle DP rank 上确定性崩溃:

```
_forward_prefill_sparse: assert core_attn_metadata.c4_sparse_raw_indices is not None
AssertionError: sparse-prefill c4 path requires c4_sparse_raw_indices
```

崩溃自 7-12 起潜伏,#31487(避免过度的 prefill CUDA graph padding)使 DP prefill 落到小的 capture bucket(4/16)、dummy extend 真正触发后,任务由绿变红(bisect:最后一个绿 commit `0ad0ff2e9e`,第一个红 `216b750c`)。

## 根因
PR 正文披露:#30898 引入的潜伏缺陷,由 #31487 暴露。DP attention + breakable prefill CUDA graph 下,idle rank 会被填充成构造的 dummy EXTEND:`forward_batch_info` 把真实模式存入 `_original_forward_mode`(IDLE),同时把 `forward_mode` 设为 EXTEND。`_get_logical_forward_mode` 返回 stale 的 IDLE,于是 metadata 走 decode 分支构建(不含 `c4_sparse_raw_indices`,该字段被 gate 在 `is_prefill=True`);而 forward 依据 `forward_mode = EXTEND` 执行 `_forward_prefill_sparse`(#29775 起默认开启),断言失败。

## 修复方案
`python/sglang/srt/layers/attention/deepseek_v4_backend.py` 的 `_get_logical_forward_mode`:为 forward 实际运行的模式构建 metadata——`forward_mode == EXTEND` 时直接返回它,而不是 stale 的 `_original_forward_mode`,从而构建 extend metadata 并分配 `c4_sparse_raw_indices`。正文说明构造出的 batch 已携带有效的 extend 字段;TARGET_VERIFY 覆写(MTP hybrid SSM)仍优先 `_original_forward_mode`。

```diff
     if forward_batch.forward_mode.is_idle():
         return forward_batch.forward_mode
+    if forward_batch.forward_mode == ForwardMode.EXTEND:
+        return forward_batch.forward_mode
     return (
         getattr(forward_batch, "_original_forward_mode", None)
         or forward_batch.forward_mode
```

## 影响与回归风险
- 影响窗口:正文未给出受影响版本范围(潜伏窗口自 7-12、由 `216b750c` 起转红)。
- 回归面:仅 DeepSeek-V4 attention backend 的 `_get_logical_forward_mode` 模式解析。EXTEND 场景不再回落 `_original_forward_mode`;IDLE 早退与 TARGET_VERIFY 优先 `_original_forward_mode` 的语义保持不变。正文验证:4×B200 devbox、`216b750c`、完全相同的 BCG 参数下,修复前 idle ranks DP1/2/3 在负载下崩溃;修复后 204/204 请求成功、0 断言;崩溃/存活用合成负载验证,gsm8k 精度留给 CI。

## 关联
- #30898(引入潜伏缺陷的 PR)
- #31487(暴露该缺陷的 PR)
- #29775(sparse prefill 默认开启的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
