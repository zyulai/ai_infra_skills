# PR #21691: [AMD] fix performance regression issue when run gpt-oss with "--context-length 13824"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21691](https://github.com/sgl-project/sglang/pull/21691) |
| 作者 | kkHuang-amd |
| 合并日期 | 2026-03-30 |
| labels | (none) |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
AMD aiter decode 的 kernel 选择把页表列数当 KV 长度,漏乘 page_size 致误选 kernel,E2E 掉 40%;补上乘法修复。

## 问题现象
运行 gpt-oss 模型且 `--context-length` 小于 32768(如标题中的 13824)时,E2E 性能下降约 40%。正文附有修复前后的性能对比图(前后截图链接)。

## 根因
正文披露:`unified_attention` 以 `max_kv_len` 决定启动哪个 kernel——条件 `max_seqlen_k <= 512` 选 `kernel_unified_attention_2d`,否则选 `kernel_unified_attention_3d`。现有代码没考虑 page_size 因素,`max_kv_len` 取的是 `context-length/64`(即页表列数/页数),不是真实 KV 长度;以 context-length 13824 为例,13824/64 = 216 ≤ 512,于是 decode attention 被误选为 2d kernel,性能大幅下降。修复即把 page_size 计入,算出真实 KV 长度再选 kernel。

## 修复方案
改 `python/sglang/srt/layers/attention/aiter_backend.py` 的 `forward_decode`,共 1 行:
```diff
-                max_kv_len = page_table.shape[1]
+                max_kv_len = page_table.shape[1] * self.page_size
```
页表列数乘以 `self.page_size` 得到真实 max KV 长度,使 `unified_attention` 的 kernel 选择(`kernel_unified_attention_2d` vs `kernel_unified_attention_3d`)基于正确的长度判断。

## 影响与回归风险
- 影响窗口:正文未说明;回归窗口为引入该误算的提交之后至本 PR 前。
- 回归面:AMD aiter 后端 `forward_decode` 中 unified_attention 的 kernel 选择路径;page_size=64 时,凡是 `page_table.shape[1] <= 512` 但真实 KV 长度 > 512 的 decode 场景(如 context-length < 32768 的 gpt-oss)从误选 2d kernel 恢复为 3d kernel;真实 KV 长度确实 ≤ 512 的短上下文场景选择不变(2d)。仅影响 kernel 选择决策,不改变数值计算语义。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
