# PR #6382: [Bugfix] Fix illegal memory access in FP8 MoE kernel

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#6382](https://github.com/vllm-project/vllm/pull/6382) |
| 作者 | comaniac |
| 合并日期 | 2024-07-12 |
| labels | ready |
| 热度 | 👍 3 · 💬 1 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
#6029 后 FP8 MoE 频发 illegal memory access,疑似与单 chunk 场景下对 intermediate cache 的切片有关;在 chunk>0 时才走该分支规避。

## 问题现象
FP8 MoE 运行中频繁出现 illegal memory access(CUDA 非法内存访问)错误,在 [#6029](https://github.com/vllm-project/vllm/pull/6029) 合并后高频出现。正文未给出具体复现脚本或日志摘录。

## 根因
正文披露:作者怀疑问题出自 fused_experts 中对中间缓存按 `tokens_in_chunk` 切片的三行(`intermediate_cache1/2/3 = ...[:tokens_in_chunk]`),但机制未查明——chunk 大小为 `64*1024`,绝大多数情况只有一个 chunk,该切片本应无任何效果,作者原话 "So I have no idea why this causes the issue"。因此本 PR 未解释确切机制,采取的是规避性修复:单 chunk 时完全不进入该分支。

## 修复方案
改 `vllm/model_executor/layers/fused_moe/fused_moe.py` 的 `fused_experts`,给"最后 chunk 缩缓存+重载配置"的分支加 `chunk > 0` 条件:

```diff
-        if tokens_in_chunk < CHUNK_SIZE:
-            # will only happen in the last chunk
+        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
+            # Adjust the intermediate cache size and config for the last
+            # chunk. Note that in most cases we only have one chunk
+            # so the cache size and config are already set correctly and
+            # do not need to be adjusted.
             intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
             intermediate_cache2 = intermediate_cache2[:tokens_in_chunk]
             intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
-            # reload config to get better performance on the last chunk
             config = get_config_func(tokens_in_chunk)
```

单 chunk(最常见的情形)时不再对缓存做切片,也不重取 config。

## 影响与回归风险
- 影响窗口:#6029 合并之后、本 PR 之前的版本;正文以 PR 链接为界,未给出具体版本号。
- 回归面:fused MoE 的分 chunk 执行路径——单 chunk 场景跳过切片与 `get_config_func` 重取(后者对最后 chunk 的性能调优配置不再生效,多 chunk 最后一块的调优配置也随之跳过?按 diff 条件,`chunk>0` 的最后一块行为不变);多 chunk 非最后一块不受影响(由 diff 推断)。

## 关联
- #6029

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
