# PR #9955: Revert "[Fix] DeepSeek EP accuracy issue on B200 GPUs (#9946)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9955](https://github.com/sgl-project/sglang/pull/9955) |
| 作者 | zhyncs |
| 合并日期 | 2025-09-03 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
整体 revert #9946:删除 `--disable-deepgemm-ue8m0` 开关及 UE8M0 缩放配置逻辑,`DEEPGEMM_SCALE_UE8M0` 恢复为恒等于 `DEEPGEMM_BLACKWELL`。

## 问题现象
正文未描述具体现象,只写明 "This reverts commit 6243c3670281d248eb7d13aa487b214f17a1c59b",Motivation / Modifications / Accuracy Tests / Benchmarking 各节均为空。即本卡是对 #9946(声称修复 B200 上 DeepSeek EP 精度问题的 PR)的回退;至于 #9946 上线后暴露了什么新问题,正文未说明。

## 根因
正文未披露根因。(由 diff 推断)本 PR 把 #9946 引入的整套 UE8M0 缩放可配置机制撤下:`update_deepgemm_scale_ue8m0()` 调用、`disable_deepgemm_ue8m0` server arg 及其 CLI 项、`configurer.py` 中基于 `SGL_ENABLE_DEEPGEMM_UE8M0` 环境变量与 server arg 的赋值逻辑全部删除,`DEEPGEMM_SCALE_UE8M0` 重新退化为编译期常量 `DEEPGEMM_BLACKWELL`。

## 修复方案
按 revert 逐文件恢复:

- `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py`:删除 `update_deep_gemm_config` 中对 `update_deepgemm_scale_ue8m0(server_args.disable_deepgemm_ue8m0)` 的调用与相应 import;
- `python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py`:删除 `DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL and get_bool_env_var("SGL_ENABLE_DEEPGEMM_UE8M0", default="true")`、`update_deepgemm_scale_ue8m0()` 整个函数,替换为一行 `DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL`;
- `python/sglang/srt/server_args.py`:删除 `disable_deepgemm_ue8m0: bool = False` 字段与 `--disable-deepgemm-ue8m0` CLI 参数(其 help 原文为 "Disable DeepGEMM UE8M0 scaling optimizations. This can help with accuracy issues on Blackwell GPUs (B200) for certain models like DeepSeek.")。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:DeepGEMM UE8M0 缩放行为回到 #9946 之前的固定逻辑(Blackwell 上恒开启);使用 `--disable-deepgemm-ue8m0` 或 `SGL_ENABLE_DEEPGEMM_UE8M0` 的用户升级后会因参数被删而报错/开关失效,需改回原行为;B200 上 DeepSeek EP 的精度问题(#9946 所针对的)随之不再有该 workaround。

## 关联
- #9946(被 revert 的 "[Fix] DeepSeek EP accuracy issue on B200 GPUs")

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
