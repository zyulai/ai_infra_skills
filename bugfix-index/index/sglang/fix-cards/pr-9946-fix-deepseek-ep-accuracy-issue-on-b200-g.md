# PR #9946: [Fix] DeepSeek EP accuracy issue on B200 GPUs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9946](https://github.com/sgl-project/sglang/pull/9946) |
| 作者 | alhridoy |
| 合并日期 | 2025-09-03 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

B200 上 DeepSeek EP 精度劣化源于 Blackwell 专属 UE8M0 FP8 缩放优化,新增 `--disable-deepgemm-ue8m0` 开关可关闭。

## 问题现象

DeepSeek 模型开启专家并行(EP)后在 NVIDIA B200 GPU 上出现精度问题,GSM8K 等精度敏感负载结果劣化;相同配置在 H200(SM90)上正常。受影响配置:`--model deepseek-ai/DeepSeek-V3-0324 --tp-size 8 --ep-size 8 --trust-remote-code` 起服务后跑 `benchmark/gsm8k/bench_sglang.py`。症状特点:没有任何报错信息,仅表现为模型输出精度下降。

## 根因

正文披露:根因是 DeepEP dispatcher 中的 Blackwell 专属优化——B200 被自动识别为 Blackwell 架构(SM100)后,Blackwell 专属的 UE8M0 FP8 缩放优化被自动启用;这些优化以性能优先、牺牲数值精度,导致精度劣化。关键代码位置在 `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`:`ENABLE_JIT_DEEPGEMM` 时调用 `sglang_per_token_group_quant_fp8` 量化,`column_major_scales` / `scale_tma_aligned` / `scale_ue8m0` 三个参数均传入 `DEEPGEMM_SCALE_UE8M0`(正文注释:This causes issues on B200)。

## 修复方案

提供显式关闭 UE8M0 优化的入口:

1. `server_args.py`:新增 `disable_deepgemm_ue8m0: bool = False` 字段与 `--disable-deepgemm-ue8m0` CLI 参数(help 文案:可帮助 Blackwell GPU(B200)上 DeepSeek 等模型的精度问题);
2. `deep_gemm_wrapper/configurer.py`:`DEEPGEMM_SCALE_UE8M0` 改为 `DEEPGEMM_BLACKWELL and get_bool_env_var("SGL_ENABLE_DEEPGEMM_UE8M0", default="true")`,并新增 `update_deepgemm_scale_ue8m0()` 函数,传入 True 时强制置 False;
3. `deep_gemm_wrapper/compile_utils.py`:在 `update_deep_gemm_config` 中调用上述函数,把 server 参数接到 DeepGEMM 配置。

也可用环境变量替代:`export SGL_ENABLE_DEEPGEMM_UE8M0=false`。

关键 diff 摘录(configurer.py):

```diff
-DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
+DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL and get_bool_env_var(
+    "SGL_ENABLE_DEEPGEMM_UE8M0", default="true"
+)
+
+
+def update_deepgemm_scale_ue8m0(disable_ue8m0: bool):
+    """Update DEEPGEMM_SCALE_UE8M0 based on server arguments."""
+    global DEEPGEMM_SCALE_UE8M0
+    if disable_ue8m0:
+        DEEPGEMM_SCALE_UE8M0 = False
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:新增参数默认 False,未传参时行为不变(UE8M0 在 Blackwell 上仍默认开启),H200 等非 Blackwell 平台不受影响;改动作用于 DeepGEMM 配置初始化路径(`update_deep_gemm_config` 现在会改写模块级全局变量 `DEEPGEMM_SCALE_UE8M0`)。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
