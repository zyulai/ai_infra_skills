# PR #31585: [Bug] Revert torch warning fix

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31585](https://github.com/vllm-project/vllm/pull/31585) |
| 作者 | yewentao256 |
| 合并日期 | 2026-01-05 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
PyTorch 2.9.0 的 float32 matmul precision 警告在 2.9.1 已被移除,先前为规避警告的 fix 不再必要,回退为 `torch.set_float32_matmul_precision`("highest" 默认)。

## 问题现象
正文未描述运行时故障现象。背景:PyTorch 2.9.0 在 `aten/src/ATen/Context.cpp`(第 79 行)会发出一个警告,之前 vLLM 为此做过一个 fix(对应 issue #31579);作者发现该警告在 PyTorch 2.9.1 中似乎已被移除,因此该 fix 可以回退。本 PR 即执行回退。

## 根因
PR 正文披露:触发先前 fix 的根源是 PyTorch 2.9.0 在 `Context.cpp` 中发出的警告;2.9.1 已移除该警告,规避代码随之失去存在意义。正文未描述 issue #31579 中警告的具体表现。

## 修复方案
回退之前的 fix,涉及三个文件:

1. `vllm/envs.py`:`VLLM_FLOAT32_MATMUL_PRECISION` 的取值从 `Literal["ieee", "tf32"]`(默认 `"ieee"`)恢复为 `Literal["highest", "high", "medium"]`(默认 `"highest"`),合法选项重新镜像 `torch.set_float32_matmul_precision`:
```diff
-    VLLM_FLOAT32_MATMUL_PRECISION: Literal["ieee", "tf32"] = "ieee"
+    VLLM_FLOAT32_MATMUL_PRECISION: Literal["highest", "high", "medium"] = "highest"
```
2. `vllm/v1/worker/gpu_worker.py`:设置精度的调用从直接赋值 `torch.backends.cuda.matmul.fp32_precision = precision` 恢复为 `torch.set_float32_matmul_precision(precision)`。
3. `tests/v1/e2e/test_async_scheduling.py`:测试中的环境变量值由 `"ieee"` 恢复为 `"highest"`。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围(与 PyTorch 2.9.0/2.9.1 相关)。
- 回归面:影响所有通过 `VLLM_FLOAT32_MATMUL_PRECISION` 配置 float32 matmul 精度的用户——若此前显式设置过 `"ieee"`/`"tf32"`,升级后该取值不再合法(可选值为 highest/high/medium);默认精度从 ieee 变回 highest,数值行为随之变化。正文未附测试结果。

## 关联
- [issue #31579](https://github.com/vllm-project/vllm/issues/31579)(Fixes)
- [PyTorch Context.cpp L79](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/aten/src/ATen/Context.cpp#L79)(正文引用的警告来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
