# PR #16622: Fix FP8 MoE NaN with DeepGEMM on Blackwell

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#16622](https://github.com/sgl-project/sglang/pull/16622) |
| 作者 | luoyuyan |
| 合并日期 | 2026-01-08 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
MOE_RUNNER_BACKEND=AUTO 但实际走 DeepGEMM 时 UE8M0 重量化未触发导致 FP8 MoE 输出 NaN,修复 DeepGEMM 检测逻辑使权重重量化生效。

## 问题现象
Blackwell GPU 上,FP8 量化 MoE 模型(如 Qwen3-MoE)使用 DeepGEMM 运行时输出 NaN。触发场景:当 `MOE_RUNNER_BACKEND` 为 AUTO 但实际使用 DeepGEMM(经由 `--moe-a2a-backend deepep/mooncake`)。正文未贴文字日志,以截图展示修复前后对比(修复前 NaN,修复后正常,见原帖)。

## 根因
PR 正文披露:`process_weights_after_loading` 中的 UE8M0 scale 重量化(requantization)条件没有在「MOE_RUNNER_BACKEND=AUTO 但实际使用 DeepGEMM」时被正确触发——原条件用 `get_moe_runner_backend().is_deep_gemm()` 判定,而 AUTO 后端实际会经由 `create_moe_runner` 解析为 DEEP_GEMM,导致本应转成 UE8M0 格式的权重 scale 未做重量化,DeepGEMM 以错误 scale 格式计算产生 NaN。

## 修复方案
改动位于 `python/sglang/srt/layers/quantization/fp8.py`(`Fp8MoEMethod`):
- 新增静态方法 `is_deepgemm_moe_runner_backend_enabled()`:后端显式为 deep_gemm 返回 True;后端为 AUTO 时按 `ENABLE_JIT_DEEPGEMM 且 (a2a 后端为 deepep 或 mooncake)` 判定,即镜像 `create_moe_runner` 的实际解析逻辑;
- `process_weights_after_loading` 中 UE8M0 重量化条件由 `get_moe_runner_backend().is_deep_gemm()` 改为 `will_use_deepgemm`(上述新方法),并保留 `not layer.w13_weight_scale_inv.format_ue8m0` 检查;
- `create_moe_runner` 的 AUTO 分支同样改用该新方法,消除两处逻辑重复。

关键 diff 摘录:

```python
@staticmethod
def is_deepgemm_moe_runner_backend_enabled() -> bool:
    """Check if MoE will actually use DeepGEMM runner for FP8."""
    moe_runner_backend = get_moe_runner_backend()
    if moe_runner_backend.is_deep_gemm():
        return True
    if moe_runner_backend.is_auto():
        return deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and (
            get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
        )
    return False
```

```diff
-                    and get_moe_runner_backend().is_deep_gemm()
+                    and will_use_deepgemm
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,未写;受影响场景为 Blackwell 上 FP8 block-quant MoE + AUTO runner + deepep/mooncake a2a 后端。
- 回归面:FP8 MoE 的权重加载后处理与 runner 选择路径。修复仅影响「AUTO 且实际用 DeepGEMM」的组合,使其正确执行 UE8M0 重量化;显式指定 deep_gemm 后端的行为不变(条件真值相同,由 diff 推断)。`create_moe_runner` 改动为等价重构,AUTO 分支解析结果不变。正文未给出数值化精度/性能测试(截图对比),未提供基准数据。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
