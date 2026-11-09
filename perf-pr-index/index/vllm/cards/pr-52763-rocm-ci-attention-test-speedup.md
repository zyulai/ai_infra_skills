# PR #52763: [ROCM][CI] Attention test speedup

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @stefankoncarevic |
| 合并 | 2026-08-18 |
| 分类 | Attention / Kernel(tier: major) |
| Label | rocm, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/52763 |

## 一句话结论

ROCm attention 测试放开 2592 个误跳过用例并跳过无谓全局清理，MI300 上整组快 4.6x。

## 动机

正文指出 `tests/kernels/attention` 远慢于其实际工作量，源于两个独立问题。(1) `test_merge_attn_states.py` 把全部 2592 个参数化用例挡在「只有 CUDA 支持 custom merge_attn_states kernel」的检查后；这已不准确——kernel 源码在构建里没有 CUDA gate，ROCm 上也编译、op 已注册且结果正确，同文件的 `both_empty` 用例本就在 ROCm 上跑原生 kernel。且该检查写在测试体内，每个被跳过的用例仍付完整 setup/teardown，整组花几分钟只为决定什么都不测。(2) 纯 kernel 模块每个测试后都跑一个 autouse fixture 拆分布式环境、清空 allocator caches，而这些模块不建 engine、不初始化分布式状态，这次 GC 与 cache flush 反而主导耗时。

## 优化点

(1) 把平台检查从测试体移到 module 级 `pytestmark`，条件改为 `current_platform.is_cuda_alike()`（CUDA 或 ROCm），从而解锁 2592 个用例在 ROCm 上的真实 kernel 覆盖，并删掉测试体内的 `pytest.skip`。(2) 给纯 kernel 模块打上既有的 `skip_global_cleanup` marker（如 `test_prefix_prefill.py`、`test_triton_decode_attention.py`、`test_triton_unified_attention.py`、`test_triton_unified_attention_diffkv.py` 及 merge 测试本身）。(3) 模块选择刻意保守：ROCm 上全局清理还会刷新 `monkeypatch` 无法恢复的 AITER class-level state，凡涉及环境变量/AITER 的模块以及一个 patch 平台类的模块都不动；正文说明最大的几个模块已占整组绝大部分耗时，故不放弃什么。

## 关键改动

```text
tests/kernels/attention/test_merge_attn_states.py: module 级 pytestmark（skip_global_cleanup + is_cuda_alike skipif），删体内 CUDA skip
tests/kernels/attention/test_prefix_prefill.py: 加 skip_global_cleanup marker
tests/kernels/attention/test_triton_decode_attention.py: 加 skip_global_cleanup marker
tests/kernels/attention/test_triton_unified_attention.py: 加 skip_global_cleanup marker
tests/kernels/attention/test_triton_unified_attention_diffkv.py: 加 skip_global_cleanup marker
```

关键片段（merge 测试的 module 级检查与移除体内 skip）：

```diff
+pytestmark = [
+    pytest.mark.skip_global_cleanup,
+    pytest.mark.skipif(
+        not current_platform.is_cuda_alike(),
+        reason="merge_attn_states kernels require CUDA or ROCm.",
+    ),
+]
```

## 性能声明

Whole group、单 MI300、相同 flags 与同一 GPU 的前后对比：Wall time 46:34 → 10:08；Passed 3207 → 5808；Skipped 3703 → 1111；Failed 0 → 0。正文结论「The group is 4.6x faster while running 2601 more tests, with no failures」。`test_merge_attn_states.py` 单独：由 7:23（6 passed、2592 skipped）变为 0:30、全部 2598 用例通过（含 fp8 variants）。正文说明以上数字均来自本地 MI300，并用默认环境复测一次以排除测量 flags 的影响；CI 上也预期有加速但「the magnitude there is not measured here」。

## 验证与回归风险

Test Plan 为整组 `tests/kernels/attention` 前后全跑、单 MI300、相同 flags/GPU，并用默认环境二次确认；解锁的用例另对照 Triton reference path 独立验证，`test_merge_attn_states.py` 单独计时。结果 0 failed。风险：模块选择刻意保守，涉及环境变量/AITER 或 patch 平台类的模块未加 marker；ROCm 上全局清理还负责刷新 AITER class-level state；CI 侧的加速幅度未测量；本 PR 只改测试/marker，不改生产 kernel。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
