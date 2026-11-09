# PR #31523: [ROCm][Bugfix] Fix accuracy issue on fmoe when `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` enabled

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31523](https://github.com/vllm-project/vllm/pull/31523) |
| 作者 | ganyi1996ppo |
| 合并日期 | 2025-12-30 |
| labels | rocm, ready |
| 热度 | 👍 0 · 💬 3 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
AITER shared experts fusion 开启时 `grouped_topk` 自定义算子未被启用导致 MoE 模型精度严重回退;在 ROCm 平台检查中强制补上 `+grouped_topk`。

## 问题现象
在 ROCm 上启用 `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS`(shared experts fusion)后,所有 MoE 结构模型出现严重精度回退(正文:"all the model with MOE structure have serious accuracy regression when shared experts fusion enabled")。Test Plan 为在 gsm8k 上测 deepseek-r1:修复前 vllm main 分支 flexible-extract exact_match 仅 0.0099±0.0027、strict-match 0.0000±0.0000(几乎全错);修复后 flexible-extract 0.9477±0.0061、strict-match 0.9454±0.0063,恢复正常水平。

## 根因
正文披露:该精度问题由 PR #31221 引入;根因是 shared experts fusion 开启时 `grouped_topk` 的算子分发(dispatch)实际没有发生,正文原话是 "This PR add `+grouped_topk` into custom_ops to make the dispatch actually happens"——即此前 custom_ops 里没有启用 `grouped_topk`,即使开了 AITER fused MoE/shared experts fusion 路径,grouped topk 仍走不到对应自定义算子,导致 MoE 路由结果错误、精度崩塌。

## 修复方案
改动仅 `vllm/platforms/rocm.py` 的 `check_and_update_config`:
- 新增读取 `use_aiter_fused_moe = rocm_aiter_ops.is_fused_moe_enabled()` 与 `use_aiter_fused_se = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()`;
- 若 `use_aiter_fused_se` 且用户显式传了 `-grouped_topk`,打 warning 并强制移除该禁用项;
- 若 `use_aiter_fused_moe` 且 custom_ops 既无 `+grouped_topk` 也无 `-grouped_topk`,追加 `+grouped_topk`,确保 AITER 下 grouped_topk 默认启用。

```diff
+        if use_aiter_fused_se and "-grouped_topk" in compilation_config.custom_ops:
+            logger.warning_once(
+                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled, which "
+                "requires the 'grouped_topk' custom op. Overriding the "
+                "user-provided '-grouped_topk'."
+            )
+            compilation_config.custom_ops.remove("-grouped_topk")
```

## 影响与回归风险
- 影响窗口:正文指出精度回退由 PR #31221 引入并随 main 分支存在,直至本修复;具体版本号未提及。
- 回归面:ROCm 平台的 custom_ops 配置逻辑——AITER fused MoE 场景下 `grouped_topk` 会被默认启用;用户显式 `-grouped_topk` 的配置在 shared experts fusion 开启时会被覆盖(有 warning 提示),这是有意的行为改变;非 AITER 路径与其他平台不受影响。

## 关联
- 引入问题的 PR: https://github.com/vllm-project/vllm/pull/31221

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
