# PR #40105: [MoE] Disable FlashInfer fused finalize by default for numerical accuracy

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zianglih |
| 合并 | 2026-09-18 |
| 分类 | MoE (tier: major) |
| Label | documentation |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 可用,实为 2 个文件的小改动) |
| 链接 | https://github.com/sgl-project/sglang/pull/40105 |

## 一句话结论

把 FlashInfer MoE fused atomic finalize 的默认值从开改为关(换数值精度),并给出在 B300 上关闭后 W4A4 延迟最高 +21.75% 的量化代价,性能需求者可用环境变量 opt-in。

## 动机

`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE` 由 #28354 以 `EnvBool(True)` 引入,并把 deterministic inference 强制为 0。更早之前,SGLang 的 CUTLASS 调用与 CuTe DSL v2 wrapper 不传 `use_fused_finalize`,而其 pin 的 FlashInfer 0.6.17 中两个 API 默认都是 `True`——即 #28354 只是把既有的 fused-finalize 默认暴露成配置项。fused atomic finalize 用原子累加合并 expert 输出,数值精度不如 unfused 路径,因此本 PR 把默认翻转为 `False` 以获得更好的数值精度;单独的 CuTe DSL v1 DeepEP masked 路径不在此 flag 作用域内。

## 优化点

严格说这是一次"性能换精度"的默认值回退而非优化:改动只有两处——`environ.py` 中 `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE` 从 `EnvBool(True)` 改为 `EnvBool(False)`,以及环境变量文档更新默认值并说明性能 opt-in。CUTLASS 与 CuTe DSL 调用方继续读同一个 flag,deterministic inference 继续强制关闭;无 kernel 改动、无新增测试。偏好激进性能的用户可设 `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=1` 恢复 fused 路径。

## 关键改动

```text
python/sglang/srt/environ.py: SGLANG_FLASHINFER_MOE_FUSED_FINALIZE 默认 EnvBool(True) -> EnvBool(False)
docs/docs/references/environment_variables.mdx: 更新默认值与说明(Directly disabled by default for better numerical accuracy)
```

关键片段:

```python
# Use FlashInfer's fused atomic CUTLASS/CuTe DSL MoE finalize.
SGLANG_FLASHINFER_MOE_FUSED_FINALIZE = EnvBool(False)
```

## 性能声明

这是测量"关闭 fused finalize 的代价"(Speedup = fused(True)/unfused(False),小于 1 表示关闭更慢)。硬件 1x NVIDIA B300 SXM6 AC(SM103),DeepSeek-V3 MoE(hidden 7168,intermediate 2048,256 experts,top-8),模拟 EP=8/TP=1,flashinfer v0.6.18.post1 `bench_moe_deepseek.py`,10 warmups/100 iters,CUDA graph + CUPTI 冷 L2,单轮 sweep 无重复:

- CuTe W4A4 per-tensor 在 1 token 时延迟增加 21.75%(speedup 0.8213x)
- 16384 tokens 时:CUTLASS +7.82%(0.9275x)、CuTe W4A4 per-tensor +12.55%(0.8885x)、CuTe W4A4 per-token +10.70%(0.9033x)、CuTe W4A16 -9.20%(1.1013x,关闭反而更快)
- PR 明确声明:这些单轮 sweep 不足以支撑"回归可忽略"的笼统结论;CUTLASS 的 `True` 只允许 eligible 的 fused tactic,autotune 仍可能选中 unfused tactic

准确性收益本身未量化:PR 未测量数值误差缩减,仅以精度动机驱动默认翻转。

## 验证与回归风险

静态验证:changed-file pre-commit 与 `git diff --check` 通过(macOS arm64 本地 CPU 检查,无 GPU);未运行模型准确性测试、未新增测试;Mintlify 构建检查未跑。回归风险:所有 FlashInfer CUTLASS/CuTe DSL MoE 用户默认走 unfused finalize,输出数值会与旧默认不同(精度更好但性能见表);依赖旧行为或追求性能需显式设 `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=1`;deterministic inference 行为不变;CI 状态 Base/Extra/AMD ROCm 均为失败标记。

## 关联

直接前序:#28354(引入该环境变量并强制 deterministic 关闭)。同区域为 FlashInfer MoE finalize 相关路径(moe_runner/flashinfer_cutlass.py 与 flashinfer_cutedsl.py 的调用方);pack 未列出同分类其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
