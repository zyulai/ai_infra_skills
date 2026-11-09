# PR #31597: [ROCm][CI] Fix language generation test accuracy by disabling HF flash_sdp and mem_efficient_sdp

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31597](https://github.com/vllm-project/vllm/pull/31597) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-01-05 |
| labels | rocm, ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
ROCm 上 HF Transformers 的 flash/mem_efficient SDP 数值不准导致语言生成测试对比失败;测试侧禁用这两个后端、改用 math_sdp。

## 问题现象
在 ROCm 上,对比 HuggingFace 与 vLLM 输出的语言生成测试因数值发散而失败,正文给出复现断言:

```
AssertionError: Test5:
hf:   'fact' (logprob=-7.794)
vllm: 'upset' (logprob=-8.045)
```

影响面是 ROCm CI 的语言生成精度测试(HF 基线对比类)。

## 根因
正文披露:根因是 ROCm 的 flash attention 与 memory-efficient SDP 实现在 HuggingFace Transformers 中存在精度问题(关联 issue #30167),导致 HF 基线输出与 vLLM 输出数值发散。

## 修复方案
新增 `tests/models/language/generation/conftest.py`,在 `pytest_sessionstart` 钩子中,仅当 `current_platform.is_rocm()` 时配置 PyTorch SDP 后端:禁用 `flash_sdp` 与 `mem_efficient_sdp`,启用数值上更稳定的参考实现 `math_sdp`,并发出 UserWarning 提示。非 ROCm 平台直接返回,不受影响。

```diff
+    torch.backends.cuda.enable_flash_sdp(False)
+    torch.backends.cuda.enable_mem_efficient_sdp(False)
+    torch.backends.cuda.enable_math_sdp(True)
```

测试验证:`pytest -v -s tests/models/language/generation/test_common.py::test_models[False-True-5-32-TitanML/tiny-mixtral]` 通过。注释中留有 TODO:待 ROCm SDP 精度问题解决后移除。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅影响 `tests/models/language/generation/` 下的测试会话,且只作用于 ROCm 平台;生产推理路径不涉及。math_sdp 更慢,可能轻微增加该目录测试时长。

## 关联
- 首次应用于 [#30270](https://github.com/vllm-project/vllm/pull/30270)
- 上游 issue: https://github.com/vllm-project/vllm/issues/30167

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
