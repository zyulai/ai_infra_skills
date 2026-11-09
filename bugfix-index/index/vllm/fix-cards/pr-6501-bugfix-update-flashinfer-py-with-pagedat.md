# PR #6501: [Bugfix] Update flashinfer.py with PagedAttention forwards - Fixes Gemma2 OpenAI Server Crash

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#6501](https://github.com/vllm-project/vllm/pull/6501) |
| 作者 | noamgat |
| 合并日期 | 2024-07-18 |
| labels | ready |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
FlashInfer backend 的 swap_blocks/copy_blocks 直接 raise NotImplementedError,调用即崩溃;改为转调 PagedAttention 的实现。

## 问题现象
正文未描述具体现象。标题指明修复 Gemma2 在 OpenAI Server 上的崩溃;正文仅写 "FIX #6477" 并链接该 issue(https://github.com/vllm-project/vllm/issues/6477),PR 描述部分保留模板占位符 "FILL IN THE PR DESCRIPTION HERE",未给出复现步骤或日志。

## 根因
正文未披露根因。(由 diff 推断)`vllm/attention/backends/flashinfer.py` 中 FlashInfer backend 的 `swap_blocks` 与 `copy_blocks` 两个静态方法此前实现为 `raise NotImplementedError`,任何走到这两个操作的路径(如 KV cache 块换入换出/拷贝)都会直接抛异常导致服务崩溃;Gemma2 场景会触达该路径。修复将二者委托给已有的 `PagedAttention` 实现,机制上即「缺失的功能实现导致调用即崩」。

## 修复方案
改 `vllm/attention/backends/flashinfer.py`:新增 `from vllm.attention.ops.paged_attn import PagedAttention` 导入;将 `swap_blocks` 从 `raise NotImplementedError` 改为 `PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)`,将 `copy_blocks` 从 `raise NotImplementedError` 改为 `PagedAttention.copy_blocks(kv_caches, src_to_dists)`。

```diff
+from vllm.attention.ops.paged_attn import PagedAttention
...
-        raise NotImplementedError
+        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)
...
-        raise NotImplementedError
+        PagedAttention.copy_blocks(kv_caches, src_to_dists)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 FlashInfer attention backend 的 `swap_blocks`/`copy_blocks` 两个入口;此前调用必崩,修复后委托给 PagedAttention 的既有实现,行为与 vLLM 其他使用 PagedAttention 的 backend 一致;不改动其他 backend 或内核逻辑。

## 关联
- #6477(正文 "FIX #6477" 指向的 issue:https://github.com/vllm-project/vllm/issues/6477)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
