# PR #32413: [ROCm][Bugfix] Disable hip sampler to fix deepseek's accuracy issue on ROCm

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32413](https://github.com/vllm-project/vllm/pull/32413) |
| 作者 | ganyi1996ppo |
| 合并日期 | 2026-01-15 |
| labels | bug, deepseek, ready, rocm, v1 |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
ROCm 上 deepseek-r1 采样输出重复内容,源于 hip/aiter sampler 精度问题;在 `forward_hip` 中临时禁用 aiter sampler,回退 native 实现修复。

## 问题现象
ROCm 平台上 `deepseek-r1`(作者环境用 deepseek-v3.1 权重)倾向生成重复(repetitive)回答。例如发送 max_tokens=1000 的 chat 请求("How to 20 pound lose weight in one week"),响应中多次出现 `20 pounds (9 kg) in one week is not a safe, sustainable...` 的重复段落。修复后同样请求的响应不再有重复内容(正文附修复前后两份完整响应 JSON 对照)。

## 根因
PR 正文未深入披露 aiter sampler 内部的数值错误机理,仅定位到问题出在 ROCm 的 hip/aiter 采样路径存在精度(accuracy)问题,采取"暂时禁用"策略;diff 中新增的 FIXME 注释亦写明 "Fix aiter_sampler's accuracy issue and remove this flag"(具体数值 bug 由后续修复)。

## 修复方案
改 `vllm/v1/sample/ops/topk_topp_sampler.py` 的 `forward_hip`:新增临时开关 `DISABLE_AITER_SAMPLER = True` 与 FIXME 注释,在调用 `self.aiter_sample(...)` 之前先判断该开关,若开启则直接 `return self.forward_native(logits, generators, k, p)` 回退到 native 采样实现;aiter 路径代码保留,待精度问题修复后再移除该 flag。

```diff
+        # FIXME: Fix aiter_sampler's accuracy issue and remove this flag
+        DISABLE_AITER_SAMPLER = True
         ...
+        if DISABLE_AITER_SAMPLER:
+            return self.forward_native(logits, generators, k, p)
         return self.aiter_sample(logits, k, p, generators), None
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:ROCm 上所有走 `forward_hip` 的 top-k/top-p 采样改走 native 实现,可能带来 ROCm 采样性能回退(作者视为临时措施);aiter sampler 代码仍保留但不再被执行。按 diff 顺序,原有的 "aiter sampler does not support returning logits/logprobs" 断言仍位于回退判断之前。验证结果:gsm8k on deepseek-v3.1 flexible-extract exact_match 0.9507±0.006、strict-match 0.9492±0.006。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
