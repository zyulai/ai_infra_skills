# PR #23948: fix: restore FP4 deepgemm path for Blackwell broken by #23686

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23948](https://github.com/sgl-project/sglang/pull/23948) |
| 作者 | yhyang201 |
| 合并日期 | 2026-04-28 |
| labels | (none) |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
#23686 重构 FP4 权重处理后在 Blackwell 对 deepgemm/auto 后端抛 NotImplementedError,恢复原逻辑修复。

## 问题现象
正文描述:DeepSeek-V4 FP4 模型在 Blackwell GPU 上使用默认 deepgemm/auto MoE runner backend 启动即被 NotImplementedError 阻断。验证环境:8x NVIDIA B300 SXM6 AC (275GB) + DeepSeek-V4-Pro;修复后服务器可用默认 runner backend 正常启动,conc=32、ISL=8k、OSL=1k 的 benchmark 无崩溃通过。

## 根因
正文披露:PR #23686(Hopper w4a16 支持)重构 FP4 expert 权重处理时,意外用 NotImplementedError 阻断了 Blackwell GPU 上默认的 deepgemm/auto 后端路径。

## 修复方案
`python/sglang/srt/layers/quantization/fp8.py` 的 `process_weights_after_loading_block_quant`:恢复原逻辑——int8 view 对所有 FP4 后端无条件执行,marlin 保持特例直通,deepgemm/mega_moe 路径留在 else 分支不变;删除「非 marlin 且非 flashinfer_mxfp4 即 raise NotImplementedError」的分支,并以注释说明 FP4 experts 支持三种 MoE 后端(marlin/flashinfer_mxfp4 只需 int8 view;deepgemm/auto 为 int8 view + mega_moe 或 scale 转换)。mega_moe 构建与 UE8M0 scale 转换(`transform_sf_into_required_layout`)的条件逻辑原样保留。

```diff
-                elif not get_moe_runner_backend().is_flashinfer_mxfp4():
-                    raise NotImplementedError(
-                        "DeepSeekV4 FP4 experts now require a native FP4 MoE backend. "
+                layer.w13_weight.data = layer.w13_weight.data.view(torch.int8)
+                layer.w2_weight.data = layer.w2_weight.data.view(torch.int8)
```

## 影响与回归风险
- 影响窗口:#23686 合入后至本 PR 之前(正文明确 #23686 引入)。
- 回归面:fp8.py 中 FP4 expert 的权重初始化路径;marlin(Hopper w4a16)与 flashinfer_mxfp4 的既有行为应保持,deepgemm/mega_moe 与 scale 转换按原样恢复;主要影响 DeepSeek-V4 FP4 在 Blackwell 上的默认后端选择。

## 关联
- sgl-project/sglang#23686(引入回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
