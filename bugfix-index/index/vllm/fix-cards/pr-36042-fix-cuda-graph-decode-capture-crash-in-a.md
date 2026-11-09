# PR #36042: Fix CUDA graph decode capture crash in AITER FlashAttention

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36042](https://github.com/vllm-project/vllm/pull/36042) |
| 作者 | iseeyuan |
| 合并日期 | 2026-03-07 |
| labels | rocm, ready, v1, nvidia, meta-exported, fb-exported |
| 热度 | 👍 0 · 💬 1 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | crash |

## 一句话结论
sliding_window 条件误把非投机负载路由到非图安全的 unified_attention,decode FULL 捕获 91% 处显存访问崩溃。

## 问题现象
ROCm AITER FlashAttention 后端在 CUDA graph decode FULL 捕获阶段崩溃,报错:

```
Memory access fault by GPU node-X on address (nil)
```

出现于 "Capturing CUDA graphs (decode, FULL): 91% (10/11)"。正文测试环境为 8x MI300X + 内部模型。触发条件:即使是非投机(non-speculative)负载,只要启用 sliding window,decode FULL 捕获也会走 `unified_attention` 路径从而崩溃。

## 根因
正文披露:上游 vllm-project/vllm#32877 为 ROCm AITER FA 后端加入投机解码支持时,把 `decode_max_query_len > 1` 的负载路由到 Triton kernel `unified_attention`,但同时在同一条件里追加了 `self.sliding_window[0] != -1`,导致非投机负载在 CUDA graph decode FULL 捕获期间也进入该路径。而 `unified_attention` 不具备 CUDA graph 捕获安全性:它在 wrapper 内部做动态内存分配(`torch.empty`)与运行时 kernel 选择,捕获期执行即触发显存访问错误。

## 修复方案
修改 `vllm/v1/attention/backends/rocm_aiter_fa.py` 的 `forward`,把 sliding window 条件从 `unified_attention` 的门控中移除,仅保留投机解码条件:

```diff
-                # Use unified_attention for speculative decoding (multi-token)
-                # or when sliding window is enabled
-                if self.sliding_window[0] != -1 or decode_max_query_len > 1:
+                if decode_max_query_len > 1:
```

(相应调整了 shuffle KV cache 的 assert 文案。)sliding window 注意力继续走已有的 `paged_attention_v1` 路径,该路径原生支持 sliding window。正文验证:修复前在 91% (10/11) 处 Memory access fault;修复后 decode FULL 捕获完成 100% (11/11),`onStartServing done` 达到,8 卡健康运行(VRAM 87%)。

## 影响与回归风险
- 影响窗口:#32877 合入到本修复之间、使用 ROCm AITER FA 后端且启用 sliding window 的版本。
- 回归面:仅改 ROCm AITER FA 的 decode 路由分支。投机解码(`decode_max_query_len > 1`)行为不变,仍走 `unified_attention`;sliding window 非投机负载改回 `paged_attention_v1`,其性能/行为以该路径为准(正文称该路径正确支持 sliding window)。shuffle KV cache 与投机解码互斥的 assert 语义不变。

## 关联
- vllm-project/vllm#32877(引入错误路由条件的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
