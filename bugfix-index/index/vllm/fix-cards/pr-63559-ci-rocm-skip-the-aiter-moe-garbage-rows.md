# PR #63559: [CI][ROCm] Skip the AITER MoE garbage-rows test for fp8 per-tensor until the AITER fix

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#63559](https://github.com/vllm-project/vllm/pull/63559) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-10-26 |
| labels | rocm |
| 热度 | 👍 0 · 💬 2 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
AITER `fused_moe_2stages` 的 `topk_ids=-1` 行未写入加上 per-tensor a2 量化导致整行归零,CI 稳定假失败;临时跳过 fp8 per-tensor 用例,等上游 ROCm/aiter#6568。

## 问题现象
`test_aiter_moe_token_padding_garbage_rows[single_inf-fp8_tensor_tensor]` 与 `[single_nan-fp8_tensor_token]` 在 MI300 上失败(amd-ci build 15340),报 `mean |mk_out| = 0`;自 2026-10-08 起约每 10 个 nightly 有 1 个因此失败。MI350X + AITER v0.1.26 上用 1e30 污染 caching allocator 复现:恰好是被跳过的 4 个参数失败(3/3 次),其余 9 个通过。加上跳过后该测试组 9 passed、7 skipped。

## 根因
正文披露:bug 在 AITER 而非 vLLM——AITER 的 `fused_moe_2stages` 用 `torch.empty` 分配 `a2`,stage 1 从不写入 `topk_ids == -1` 行对应的 slot,动态 per-tensor a2 量化便在这段 stale 内存上取 scale,可能把每个真实行都压成 fp8 零。per-token 与 per-block 量化按行取 scale,不受影响,因此只跳过 fp8 per-tensor 权重的配置(测试运行时无静态 a2 scale)。

## 修复方案
`tests/kernels/moe/test_modular_kernel_combinations.py`:在 `test_aiter_moe_token_padding_garbage_rows` 中,当 quant 配置的 fp8 key 为 `kFp8StaticTensorSym` 时 `pytest.skip`,跳过信息注明根因与解除条件(`AITER_BRANCH` 包含 ROCm/aiter#6568 后移除):

```diff
+    if (
+        quant_config is not None
+        and config.fp8_quant_key_pair()[0] == kFp8StaticTensorSym
+    ):
+        pytest.skip(
+            "AITER fused_moe_2stages allocates a2 with torch.empty and stage1 "
+            "never writes the topk_ids=-1 slots, ..."
+        )
```

作者说明:tests/AGENTS.md 不鼓励加 skip,这是随同测试中既有 ROCm/aiter#6275 skip 并列的临时 ROCm-only 守卫,直到 AITER 版本升级。

## 影响与回归风险
- 影响窗口:正文未说明(CI 失败自 2026-10-08 起被观察到;验证用 AITER v0.1.26)。
- 回归面:纯测试守卫,不影响任何生产代码路径;仅影响 ROCm CI 上 fp8 per-tensor 配置的 garbage-rows 用例结果口径(由 fail 变 skip)。真正的修复在上游 ROCm/aiter#6568,合并进 `AITER_BRANCH` 后需移除本守卫。

## 关联
- AITER 修复 https://github.com/ROCm/aiter/pull/6568
- 同类 vLLM 侧尝试(未修复)https://github.com/vllm-project/vllm/pull/62927
- 既有 skip 对应 https://github.com/ROCm/aiter/issues/6275

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
