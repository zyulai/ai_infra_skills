# PR #46236: [Bugfix][Quant] Raise actionable error instead of bare assert for group-size/TP mismatch (#46230)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46236](https://github.com/vllm-project/vllm/pull/46236) |
| 作者 | ArsalanShakil |
| 合并日期 | 2026-06-30 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 9 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
group-size 与 TP 分片不整除时模型加载只剩裸 assert,改为统一 helper 抛出含层名与建议的 ValueError。

## 问题现象
加载 group 量化的 compressed-tensors checkpoint(W4A16 / W8A16,以及 W4A8-FP8 scheme)时,如果 tensor-parallel sharding 把层的输入维度切成的分片不是量化 group 的整数倍(`input_size_per_partition = input_size // tensor_parallel_size` 不能被 `group_size` 整除),模型加载阶段会直接中止,只留下一条无任何说明的断言:

```
AssertionError: assert input_size_per_partition % group_size == 0
```

用户面对这条裸断言无从判断该改什么配置。触发条件是 TP 并行度与 checkpoint 的 group size 组合不当,属于加载期必然失败,不涉及运行期数据错误。

## 根因
正文披露:三类 compressed-tensors scheme 里各自写死了相同的裸 `assert input_size_per_partition % group_size == 0`,当 TP 分片导致 `input_size_per_partition` 不是 `group_size` 整数倍时触发;断言本身语义正确(不整除的权重确实无法加载),问题在于报错信息为零,既没说层名、也没说两个值各是多少、更没给修复建议,等价于一次不可诊断的加载失败。

## 修复方案
把整除校验抽成单一 helper `verify_group_size_divides_partition`,放在 `vllm/distributed/utils.py`(与既有 `verify_marlin_supports_shape` 风格对齐),不整除时抛出带层名、具体数值和修复建议的 `ValueError`:

```python
+def verify_group_size_divides_partition(
+    input_size_per_partition: int,
+    group_size: int,
+    layer_name: str | None = None,
+    extra_suggestion: str = "",
+) -> None:
+    if input_size_per_partition % group_size == 0:
+        return
+    raise ValueError(
+        f"Weight {input_size_per_partition=}{location} is not divisible by "
+        f"{group_size=}. ... Consider reducing tensor_parallel_size{extra_suggestion}."
+    )
```

调用点按 diff 实际改动:正文点名 `compressed_tensors_wNa16.py`、`compressed_tensors_wNa8o8.py`、`compressed_tensors_w4a8_fp8.py` 三处裸 assert 换成 helper;diff 还显示 `compressed_tensors_w4a8_int.py` 原本带信息但格式冗长的 assert、以及 `marlin_utils.py` 中 `verify_marlin_supports_shape` 的内联 ValueError 也一并改用该 helper。修复后报错示例:`input_size_per_partition=320 is not divisible by group_size=128 ... Consider reducing tensor_parallel_size.`。新增 `tests/quantization/test_group_partition_divisibility.py` 覆盖整除/不整除两种情形(2 passed);作者注明完整 `create_weights` 路径因内核选择需 GPU 平台未在本地跑通,scheme 接线留给 CI/GPU review。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:按 diff,改动集中在加载期 `create_weights` 的参数校验与 marlin 形状校验,不影响运行期数值路径。行为变化:同样场景下异常类型从 `AssertionError`(部分场景原本已 ValueError)统一变为 `ValueError`,若有外部脚本捕获 `AssertionError` 需相应调整;`verify_marlin_supports_shape` 的判断由 `group_size < input_size and ... % group_size != 0` 改为 `group_size < input_size` 后委托 helper,逻辑等价但报错文案有细微变化。

## 关联
- [#46230](https://github.com/vllm-project/vllm/issues/46230)(本 PR 修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
