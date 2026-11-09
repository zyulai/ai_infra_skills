# PR #34832: Revert "[Bugfix] Disable TRTLLM attention with KV transfer enabled (#33192)"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34832](https://github.com/vllm-project/vllm/pull/34832) |
| 作者 | ZhanqiuHu |
| 合并日期 | 2026-03-01 |
| labels | bug, nvidia, ready, v1 |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
撤销 #33192:其要修的连续性断言问题已被 #33241 修复,而它全局禁用 TRTLLM attention 反而在 B200 上引发 query dtype 不匹配回归。

## 问题现象
PR #33192 合入后,在 B200 上对 FP8/NVFP4 模型跑 P/D disaggregation(KV transfer 启用)即崩:`AssertionError: Query dtype mismatch: expected torch.bfloat16, got torch.float8_e4m3fn`(复现:`nvidia/Qwen3-32B-NVFP4`,P/D on B200),P/D 分离服务不可用。

## 根因
正文披露两层根因:
1. #33192 要修的原 bug 已不存在:该问题由 PR #30207 引入——它使 `prefer_cross_layer_blocks` 返回 True,跨层分配在为 TRTLLM attention 做 permute 时产生非连续 KV cache 视图,触发 `is_strictly_contiguous(kv_cache_permute)` 断言失败;而 #30207 后来被 PR #33241 回滚,原 bug 已随之修复。
2. #33192 自身引入回归:它在 KV transfer 启用时全局把 `can_use_trtllm = False`(于是 `q_data_type = bfloat16`),但每层的 `FlashInferImpl` 会在 Blackwell GPU 上独立启用 FP8 query 量化,两者判断不一致,导致 query dtype 断言失败。

## 修复方案
回滚 commit `bbe0574d8e51c1c5935aeff9e92040c61d1d59c5`(#33192),恢复 `vllm/v1/attention/backends/flashinfer.py` 中两处被禁用的 TRTLLM attention 逻辑:
1. 删除 `__init__` 中「KV transfer 启用时全局置 `can_use_trtllm = False`」的拦截及 `self._kv_transfer_enabled` 记录。
2. 删除 `build()` 中「`self._kv_transfer_enabled` 时强制 `prefill_use_trtllm = False`」的拦截。

关键 diff 摘录(删除部分):
```diff
-        self._kv_transfer_enabled = vllm_config.kv_transfer_config is not None
-        if can_use_trtllm and self._kv_transfer_enabled:
-            logger.info_once(
-                "TRTLLM attention is disabled because KV transfer "
-                ...
-            )
-            can_use_trtllm = False
...
-        if self._kv_transfer_enabled:
-            prefill_use_trtllm = False
```

## 影响与回归风险
- 影响窗口:#33192 合并后至本回滚之间,KV transfer(P/D)启用且满足 TRTLLM attention 条件的部署,尤其 Blackwell(B200)上 FP8/NVFP4 模型必崩。
- 回归面:恢复 TRTLLM attention 在 KV transfer 场景的使用。作者用三个场景验证:带 #33192 的 dtype mismatch(崩)→回滚后通过;原 contiguity bug 场景(`Qwen/Qwen3-0.6B`, P/D on B200)回滚后通过、无连续性报错。若未来 KV transfer 的 permute 再次产生非连续 KV 视图,原断言失败可能复现,但依赖 #30207 类改动不再合入。

## 关联
- [#33192](https://github.com/vllm-project/vllm/pull/33192)(被回滚的 PR)
- [#30207](https://github.com/vllm-project/vllm/pull/30207)(原连续性 bug 的引入者)
- [#33241](https://github.com/vllm-project/vllm/pull/33241)(回滚 #30207、修复原 bug 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
