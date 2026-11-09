# PR #53940: [ROCm][Kimi-K3] Enable a4w4 flydsl kernels for KimiK3

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ppalanga |
| 合并 | 2026-09-16 |
| 分类 | MoE（tier: major） |
| Label | rocm, ready, verified, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/53940 |

## 一句话结论

将 Kimi-K3 SiTUv2 MoE 从 a8w4 切到 AITER a4w4 FlyDSL 内核，GSM8K exact_match 0.9659 持平。

## 动机

Trace 对比（Kimi-K3 / MI355X）显示 vLLM dispatch 的是 `mfma_moe1_silu_mul_afp8_wfp4_fp8_*_gui_xcd4_*`（a8w4），而 ATOM 用的是 `mfma_moe1_silu_mul_afp4_wfp4_fp4_*_situv2_*`（a4w4）。AITER 自 ROCm/aiter#4463 起通过 `AITER_SITUV2_A4W4` 支持 a4w4，但 vLLM 之前把 Kimi-K3 recipe 开关接到了 a8w4 路径。AITER #4463 的孤立内核计时显示 tuned 行齐备时 a4w4 领先 a8w4，因此让既有 SiTUv2 recipe 默认走 a4w4。

## 优化点

在 `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1`（既有 recipe 开关）时，把 SiTU MXFP4 routed experts 从 a8w4（`afp8_wfp4_fp8`、`fp8q`、`_gui_` interleave）改走 AITER 的 a4w4 FlyDSL 内核（`afp4_wfp4_fp4`、`fp4q`）。vLLM 在 init 设置 `AITER_SITUV2_A4W4=1` 并清除遗留的 `AITER_SITUV2_A8W4` 覆盖（AITER 优先检查 A8W4，留着会仍选 afp8）；SiTUv2 gate/up 权重改用 separated layout 以匹配 ATOM 内核选择。不新增 env flag，保留原 recipe 变量名以兼容。该路径依赖 AITER ≥ v0.1.20（含 #4463 的 tuned `kimik3_a4w4_*_fmoe.csv`），在 v0.1.19 上会记录 heuristic FlyDSL fallback，并非 tuned a4w4。

## 关键改动

```text
vllm/_aiter_ops.py: 新增 _sync_aiter_situv2_moe_env()，把 SiTUv2 toggle 镜像为 AITER_SITUV2_A4W4=1 并 pop 掉 AITER_SITUV2_A8W4；rocm_aiter_ops 内 _MOE_SITUV2_A8W4 重命名为 _MOE_SITUV2，is_fused_moe_situv2_a8w4_enabled 改为 is_fused_moe_situv2_enabled，并在 refresh/注册时调用同步函数。
vllm/envs.py: 环境变量 VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4 改为 VLLM_ROCM_USE_AITER_MOE_SITUV2，旧名作为 deprecated alias 仍可启用 a4w4。
vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py: SiTU 分支 gate_mode 固定为 SEPARATED，去掉 a8w4 的 INTERLEAVE 判定。
vllm/model_executor/layers/fused_moe/oracle/mxfp4.py: SiTU 权重转换 guinterleave 设为 False（separated layout），并把 AITER_BF16_FP8_MOE_BOUND 设置移到仅 interleaved a16w4 路径。
tests/kernels/moe/test_rocm_aiter_moe.py: 新增三个测试，校验 AITER_SITUV2_A4W4=1、AITER_SITUV2_A8W4 被清除、旧 alias 仍启用 a4w4、以及 disabled 时清理 env。
```

```text
def _sync_aiter_situv2_moe_env() -> None:
    if envs.VLLM_ROCM_USE_AITER_MOE_SITUV2:
        os.environ["AITER_SITUV2_A4W4"] = "1"
        os.environ.pop("AITER_SITUV2_A8W4", None)
    else:
        os.environ.pop("AITER_SITUV2_A4W4", None)
```

## 性能声明

GSM8K（Kimi-K3 MXFP4，MI355X，TP=8，`VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1`，greedy 5-shot，full test split `n=1319`）：strict-match 与 flexible-extract 的 exact_match 均为 **0.9659 ± 0.005**（本 PR a4w4）对 prior a8w4 recipe 的 0.9666，正文结论为 "No accuracy regression (within stderr)"；decode dispatch 为 `flydsl_moe1_afp4_wfp4_bf16_*`（fp4 activations），无 `afp8` / `_gui_` 内核。`niah_single_2` at 32k（`n=20`）：a4w4 与 a8w4 均为 **1.0**。8k ISL / 1k OSL serving 在不含 #4463 tuned CSV 的 AITER 上（heuristic a4w4 vs tuned a8w4）"is a wash（~0.5% at conc 4/8）"。PR 正文未给出 serving 加速的定量数字。

## 验证与回归风险

Test plan 含 `pytest tests/kernels/moe/test_rocm_aiter_moe.py -k situv2`（gfx950）、[x] serve Kimi-K3 确认走 `afp4_wfp4_bf16` 而非 `afp8`/`fp8q`/`_gui_`、[x] GSM8K 与 niah_single_2 A/B，以及待办：在 AITER ≥ v0.1.20 确认 tuned a4w4 配置（`using 2stage`、无 `heuristic FlyDSL fallback`）、8k/1k serving A/B、全 decode 下 cudagraph 正确性验证（区别于 #44988 的通用 `AITER_MXFP4_MXFP4` cudagraph 问题）。风险：强依赖 AITER 版本，旧版退化为 heuristic 配置；launch 脚本不得再设置 `AITER_SITUV2_A8W4=1`（会覆盖 a4w4）。

## 关联

ROCm/aiter#4463（a4w4 dispatch 与 tuned CSV 依赖）、#50813（为 Quark checkpoint 显式 opt-in a8w4）、#44988（通用 AITER_MXFP4_MXFP4 cudagraph 问题）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
