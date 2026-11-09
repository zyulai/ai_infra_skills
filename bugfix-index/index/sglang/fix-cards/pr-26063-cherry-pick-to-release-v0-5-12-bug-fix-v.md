# PR #26063: [Cherry-pick to release/v0.5.12] [Bug] Fix V4-Pro NaN on Blackwell by converting fp8_einsum input scale to ue8m0 (#25733)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26063](https://github.com/sgl-project/sglang/pull/26063) |
| 作者 | Kangyan-Zhou |
| 合并日期 | 2026-05-22 |
| labels | cherry-pick, deepseek |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论

V4-Pro 在 Blackwell 上 fp8_einsum 产生 NaN;对输入 scale 先做 ceil_to_ue8m0 转换后 cherry-pick 到 v0.5.12。

## 问题现象

DeepSeek V4-Pro 在 Blackwell 上出现 NaN(标题披露)。正文主体为 cherry-pick workflow 自动生成的说明,未另行描述现象、复现命令与影响面。

## 根因

标题披露根因方向:fp8_einsum 的输入 scale 未转换导致 NaN,修复方式是把输入 scale 转换为 ue8m0。由 diff 可见具体机制:在调用 `deep_gemm.fp8_einsum` 之前,对 scale 张量 `o_s` 执行 `deep_gemm.ceil_to_ue8m0(o_s)`,即 fp8_einsum 要求输入 scale 为 ue8m0 格式,此前未做该转换(由 diff 推断的机制细节)。

## 修复方案

改 `python/sglang/srt/models/deepseek_v4.py` 的 `forward`,在 fp8_einsum 调用前补一行:

```diff
                 o.reshape(T * G, D).contiguous(),
                 group_size=128,
             )
+            o_s = deep_gemm.ceil_to_ue8m0(o_s)
             output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
             deep_gemm.fp8_einsum(
```

本 PR 为源 PR #25733(源 commit `79ea30d1f134b741ddc55f75db45c0299fa2a642`)向 `release/v0.5.12` 分支的自动 cherry-pick,改动即该单行。

## 影响与回归风险

- 影响窗口:正文未说明,不写(cherry-pick 目标分支为 release/v0.5.12)。
- 回归面:仅 `deepseek_v4.py` 的 fp8_einsum 路径,对 scale 做向上取整到 ue8m0 会轻微改变 scale 数值(取整方向),影响面为 Blackwell 上 V4-Pro 的该计算路径;其他模型与量化路径未动。正文未附精度/性能验证数据。

## 关联

- 源 PR:https://github.com/sgl-project/sglang/pull/25733
- 源 commit:79ea30d1f134b741ddc55f75db45c0299fa2a642

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
