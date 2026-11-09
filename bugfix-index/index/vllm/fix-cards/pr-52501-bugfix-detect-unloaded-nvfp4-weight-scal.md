# PR #52501: [Bugfix] Detect unloaded NVFP4 weight scales with a NaN sentinel

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#52501](https://github.com/vllm-project/vllm/pull/52501) |
| 作者 | pavelzak |
| 合并日期 | 2026-09-13 |
| labels | bug, ready, quantization, verified |
| 热度 | 👍 2 · 💬 24 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
ModelOpt NVFP4 线性层 weight_scale 未加载时静默产出 NaN 输出;改为 NaN 哨兵初始化并在 process() 检测,加载期即报错。

## 问题现象
混合精度 checkpoint 把某层存为 BF16、但该层未被排除出 NVFP4 量化时,加载不报错,推理时才产出 NaN 输出。验证:DeepSeek-V4-Flash-0731 于 2× DGX Spark (GB10)、TP=2 上服务,受影响 checkpoint 启动即失败并指认问题层,全 FP4 checkpoint 正常加载;NaN 哨兵检查在完全未加载与部分加载 scale 两种情况下均保持检测(FP32 cast 后仍可检出)。加载期验证用 `pre_commit run --all-files --hook-stage manual`(Ruff、clang-format、mypy 3.10-3.13)全部通过。

## 根因
正文披露:weight_scale 未被完整填充(skipped or partial scale loading)时,依赖 `torch.empty()` 的内存内容无法检测;需要显式哨兵才能在加载期发现,而不是等到推理产出 NaN。

## 修复方案
`vllm/model_executor/layers/quantization/modelopt.py`:
- `create_weights`:per-block(group_size)`weight_scale` 以 NaN 初始化(`init=float("nan")`),不再留 torch.empty 垃圾值,与 #45320 的哨兵模式一致。
- `KNvfp4Static.process()`:cast 到 FP32(backend 兼容)后检查 NaN,仍存在则抛错:

```python
if torch.isnan(layer.weight_scale.float()).any():
    raise RuntimeError(
        f"NVFP4 weight_scale for layer "
        f"{getattr(layer, 'name', repr(layer))!r} was never loaded "
        "(still NaN). The checkpoint likely stores this layer as "
        "BF16 (not FP4). Fix: pass quant_config=None when "
        "constructing this layer, or add it to the quantization "
        "ignore list."
    )
```

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围。
- 回归面:ModelOpt NVFP4 linear 权重加载/处理路径;真实未加载场景从「静默 NaN 输出」变为「启动期 RuntimeError」(有意的行为收紧)。rebase 后 GB10/model 检查未重跑(正文如实说明);rebase 含上游 #55630(无关的 CUDA 注释格式修复)。

## 关联
- #45320(MoE 路径缺失 per-expert scales 的检查;本 PR 覆盖 linear 路径 per-block scales)
- #42601(dead experts 的 activation-scale clamping,另一 NaN 来源)
- #55630(rebase 带入的注释格式修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
