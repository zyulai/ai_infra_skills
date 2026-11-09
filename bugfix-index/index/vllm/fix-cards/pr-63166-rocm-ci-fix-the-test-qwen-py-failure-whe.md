# PR #63166: [ROCm][CI] Fix the test_qwen.py failure when AITER is on

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#63166](https://github.com/vllm-project/vllm/pull/63166) |
| 作者 | qli88 |
| 合并日期 | 2026-11-07 |
| labels | rocm, multi-modality, qwen, glm |
| 热度 | 👍 0 · 💬 25 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
AITER RMSNorm 实现与 HF 有差异导致 test_qwen.py ppl 测试失败;PPL 只按答案 token 计并对 aiter_on 开 emulate_precision_casts。

## 问题现象
`models/multimodal/generation_ppl_test/test_qwen.py` 在 AITER 开启时失败。根因敏感性来自 PPL 计算方式:参考答案之外的 prompt token(系统提示、问题、chat-template 标记)几乎不可预测,其 log-prob 位于 softmax 尾部,微小数值差异就会大幅移动它们;这些 token 占评分 token 的 20% 以上,却贡献约三分之二的总 loss,基本只是数值噪声。

## 根因
正文披露:AITER 的 RMSNorm 与 HF 版本实现不同,且 AITER 开启时 Inductor 生成不同的融合 kernel;更关键的是 PPL 计算本身不够准确(见上),把数值噪声放大成了测试失败。

## 修复方案
正文说明:将 PPL 计算改为仅基于答案(answer)token,参考 PPL 也相应更新;`test_glm.py` 同步改用新计算方式。diff 可见部分还包括:
1. 新增 `tests/models/multimodal/generation_ppl_test/conftest.py`:定义 `aiter_config` fixture,参数化 `aiter_off` / `aiter_on` / `aiter_on_rmsnorm_off` 三种环境配置(通过 MonkeyPatch 设置 `VLLM_ROCM_USE_AITER`、`VLLM_ROCM_USE_AITER_RMSNORM` 并刷新 env 缓存;不支持 aiter 时仅跑 `aiter_off`);
2. `test_qwen.py`:`aiter_on` 配置下注入编译配置,保留 Inductor 的 bf16 casts 使周边算子的舍入与 HF 一致:

```diff
+    if aiter_config == "aiter_on":
+        # AITER RMSNorm rounds once, unlike HF; keep Inductor's bf16 casts so
+        # the surrounding ops round like HF (see ROCm/aiter#2236).
+        extra_kwargs = {
+            **extra_kwargs,
+            "compilation_config": {
+                "inductor_compile_config": {"emulate_precision_casts": True}
+            },
+        }
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本号。
- 回归面:`generation_ppl_test` 下 qwen 与 glm 两个 ppl 测试的评分口径(只按答案 token 计 PPL)与 AITER 环境参数化;`aiter_on` 路径新增 `emulate_precision_casts` 编译开关,可能掩盖 AITER RMSNorm 与 HF 的真实数值差异而非消除(diff 注释所述用途);conftest 在测试内 monkeypatch 全局 env 并刷新 aiter 环境变量,同进程内其他依赖 env 缓存的用例可能受影响(由 diff 推断)。

## 关联
正文未列出(diff 注释中引用了 ROCm/aiter#2236)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
