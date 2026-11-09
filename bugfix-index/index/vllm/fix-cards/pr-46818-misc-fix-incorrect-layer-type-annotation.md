# PR #46818: [Misc] Fix incorrect layer type annotation in Fp8LinearMethod

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46818](https://github.com/vllm-project/vllm/pull/46818) |
| 作者 | skajre |
| 合并日期 | 2026-06-27 |
| labels | ready |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
`Fp8LinearMethod` 两个方法把 layer 参数误注解为 `RoutedExperts`(从 MoE 方法复制粘贴残留),实际只会收到 Linear 层;改为 `torch.nn.Module` 纯注解修正。

## 问题现象
`Fp8LinearMethod.create_weights` 与 `process_weights_after_loading` 的 `layer` 参数被注解为 `RoutedExperts` 类型,与实际传入类型不符。这是静态类型/可读性层面的错误:任何类型检查(如 mypy)或阅读代码的人都会被误导,认为这两个方法处理专家权重。正文未描述运行时故障现象——此 bug 只存在于类型注解,不影响运行时行为。

## 根因
正文披露:`Fp8Config.get_quant_method` 会把 `LinearBase` 层路由到 `Fp8LinearMethod`、把 `RoutedExperts` 层路由到 `Fp8MoEMethod`,因此 `RoutedExperts` 实例永远不会到达 `Fp8LinearMethod`,该注解是类型错误的。它是由 MoE 方法复制粘贴而来的残留(copy-paste leftover):方法体实际操作的是线性层属性(`logical_widths`、单个 `[out, in]` 形状的 `weight`),而非专家权重;且同类内部也不一致——`apply` 方法已把 `layer` 注解为 `torch.nn.Module`。

## 修复方案
改 `vllm/model_executor/layers/quantization/fp8.py` 中 `Fp8LinearMethod` 的两处参数注解:
```python
def create_weights(self, layer: torch.nn.Module, ...)
def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
```
即 `layer: RoutedExperts` → `layer: torch.nn.Module`,与基类 `LinearMethodBase.create_weights` 的签名以及同类 `apply` 方法保持一致。`Fp8MoEMethod` / `Fp8OnlineMoEMethod` 中的 `RoutedExperts` 注解是正确的,保持不变。diff 仅为这两行注解替换。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:纯类型注解改动,无运行时行为变化(Python 注解不在运行时强制执行,且两个类型都是 `nn.Module` 子类)。影响仅限静态类型检查、IDE 提示与代码可读性;`ruff check` 与 `ruff format --check` 在改动文件上通过。唯一可能的下游影响是依赖这两个签名做类型推导的工具输出会变化。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
