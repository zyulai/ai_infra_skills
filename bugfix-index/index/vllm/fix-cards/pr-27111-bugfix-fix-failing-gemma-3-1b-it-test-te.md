# PR #27111: [BugFix] Fix failing gemma-3-1b-it test: `test_lm_eval_accuracy_v1_engine[google/gemma-3-1b-it]`

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#27111](https://github.com/vllm-project/vllm/pull/27111) |
| 作者 | LucasWilkinson |
| 合并日期 | 2025-10-18 |
| labels | ready, ci/build |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
gemma-3-1b-it 精度测试挂掉,升级 vllm-flash-attn 依赖到含 flash-attention 修复的新提交解决。

## 问题现象
CI 中的精度测试 `pytest tests/entrypoints/llm/test_accuracy.py::test_lm_eval_accuracy_v1_engine[google/gemma-3-1b-it]` 失败;应用本 PR 后该测试通过("Now passes")。正文未描述具体的精度指标或失败日志细节。

## 根因
正文未直接披露根因,仅说明本 PR 是 flash-attention 侧修复(vllm-project/flash-attention#102)在 vLLM 仓库的对应改动。(由 diff 推断)修复手段是把 cmake 外部项目 `vllm-flash-attn` 的 `GIT_TAG` 从 `8f468e7da54a8e2f98abfa7c38636aac91c0cba1` 升级到 `a893712401d70362fbb299cd9c4b3476e8e9ed54`,即引入包含 flash-attention#102 修复的构建;据此推断 gemma-3-1b-it 的精度回归源于旧版本 vllm-flash-attn 中、由该上游 PR 修复的 attention kernel 问题。

## 修复方案
仅改一个文件 `cmake/external_projects/vllm_flash_attn.cmake`,把 FetchContent 声明的 vllm-flash-attn 源码版本指针升级:

```diff
   FetchContent_Declare(
           vllm-flash-attn
           GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
-          GIT_TAG 8f468e7da54a8e2f98abfa7c38636aac91c0cba1
+          GIT_TAG a893712401d70362fbb299cd9c4b3476e8e9ed54
```

vLLM 仓库内无任何代码改动,修复完全由上游 flash-attention 提交带入。

## 影响与回归风险
- 影响窗口:正文未说明;可确认从旧 `GIT_TAG` 被引入起、到本次升级前的构建受影响,具体提交范围正文未给出。
- 回归面:vllm-flash-attn 是 vLLM 的 attention 后端依赖,升级后所有使用该 kernel 构建的平台/模型都会换到新版本;若新提交引入其他行为变化,会全局生效。修复作者称目标测试已通过。

## 关联
- https://github.com/vllm-project/flash-attention/pull/102

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
