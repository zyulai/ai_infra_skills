# PR #11827: Revert "Fix: Dynamic RoPE Cache Expansion to Prevent Position-ID Out-of-Bounds in EAGLE + Long-Sequence Workloads"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11827](https://github.com/sgl-project/sglang/pull/11827) |
| 作者 | hnyls2002 |
| 合并日期 | 2025-10-19 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | regression |

## 一句话结论
回滚 #10788 的 RoPE 缓存动态扩容,因其破坏 EAGLE 的 test_gsm8k 单测;待作者修复后再议。

## 问题现象
正文说明:#10788(动态 RoPE cache 扩容,防 EAGLE + 长序列 position-id 越界)破坏了 `python -m unittest test_eagle_infer_b.TestEAGLEServer.test_gsm8k`,故整体回滚,并 @ YAMY1234 要求先修复。正文未附具体报错输出。

## 根因
正文未披露 #10788 为何会使该测试失败,只给出事实:该改动导致 test_gsm8k 失败。(由 diff 推断)#10788 曾引入 `_ensure_cos_sin_cache_length`(按需扩容 cos_sin_cache)、载入模型后调用的 `reserve_rope_cache_for_long_sequences`(按投机步数/安全系数/对齐参数预留扩容)及 3 个环境变量(SGLANG_SPEC_EXPANSION_SAFETY_FACTOR=2、SGLANG_ROPE_CACHE_SAFETY_MARGIN=256、SGLANG_ROPE_CACHE_ALIGN=128);本回滚整体移除这些逻辑,恢复原静态 cache 行为。

## 修复方案
按 revert 逐处移除 #10788 新增代码:`python/sglang/srt/environ.py` 删除 3 个 RoPE cache 环境变量;`layers/rotary_embedding.py` 删除 `_ensure_cos_sin_cache_length`;`model_executor/model_runner.py` 删除 `load_model` 中的 `reserve_rope_cache_for_long_sequences` 调用与 import;`utils/common.py` 删除该函数本体。

```diff
-    # RoPE cache configuration
-    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR = EnvInt(2)
-    SGLANG_ROPE_CACHE_SAFETY_MARGIN = EnvInt(256)
-    SGLANG_ROPE_CACHE_ALIGN = EnvInt(128)
-
     # Overlap Spec V2
     SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)
```

## 影响与回归风险
- 回归面:EAGLE/长序列场景恢复到 #10788 合入前的 RoPE cache 行为,不再动态扩容 cos_sin_cache;3 个环境变量不复存在,依赖它们的配置会失效。若 #10788 标题所称「EAGLE + 长序列 position-id 越界」问题真实存在,回滚后该问题可能重新出现(正文未说明测试失败与该问题之间的关系);test_gsm8k 恢复通过是本回滚的直接目的。

## 关联
- [#10788](https://github.com/sgl-project/sglang/pull/10788)(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
