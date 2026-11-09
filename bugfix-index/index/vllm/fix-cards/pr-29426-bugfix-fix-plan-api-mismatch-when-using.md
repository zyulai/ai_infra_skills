# PR #29426: [BugFix] Fix `plan` API Mismatch when using latest FlashInfer

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#29426](https://github.com/vllm-project/vllm/pull/29426) |
| 作者 | askliar |
| 合并日期 | 2025-11-27 |
| labels | ready, ci/build, v1, nvidia |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
FlashInfer 0.5.3 给 plan API 新增参数,导致 vLLM fast_plan_decode 调用签名不匹配;升级依赖至 0.5.3 并补传 num_colocated_ctas=0。

## 问题现象
正文未描述具体运行时报错信息,仅说明:最新 FlashInfer(0.5.3)更改了 `plan` API,导致 vLLM 的 `fast_plan_decode` 被破坏(breaks)。

## 根因
正文明确披露:FlashInfer 0.5.3 变更了 plan API,`plan` 调用新增了一个参数;vLLM 的 `fast_plan_decode` 调用 `self._cached_module.plan` 时仍按旧签名传参(tensor core 版本恰为 18 个参数),缺少新增参数,导致 API 不匹配。

## 修复方案
- `vllm/v1/attention/backends/flashinfer.py` `fast_plan_decode`:在 `self._cached_module.plan(...)` 调用末尾追加实参 `0`(对应新增参数 `num_colocated_ctas=0`),注释由 "pass exactly 18 arguments" 改为 "exactly 19 arguments";

```diff
-        # Make sure we pass exactly 18 arguments for tensor core version
+        # Make sure we pass exactly 19 arguments for tensor core version
         self._plan_info = self._cached_module.plan(
             ...
             disable_split_kv,
+            0,
         )
```

- `docker/Dockerfile`:`flashinfer-cubin` 与 `flashinfer-jit-cache` 从 0.5.2 升级到 0.5.3;
- `requirements/cuda.txt`:`flashinfer-python` 从 0.5.2 升级到 0.5.3(注释注明需与 Dockerfile 同步更新)。

## 影响与回归风险
- 回归面:使用 FlashInfer backend 的 fast plan decode 路径(V1,NVIDIA);升级后环境强制要求 FlashInfer 0.5.3——若仍在旧版 FlashInfer(0.5.2)上运行,plan 调用会因参数个数不匹配而失败;依赖文件与 Dockerfile 需成对更新。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
