# PR #3869: [Bugfix] Fix incorrect output on OLMo models in Tensor Parallelism

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#3869](https://github.com/vllm-project/vllm/pull/3869) |
| 作者 | Isotr0py |
| 合并日期 | 2024-04-05 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Parallelism / Communication |
| bug 类型 | correctness |

## 一句话结论
OLMo 的 `ff_proj` 未实现张量并行,TP>1 时输出错误;改用 `MergedColumnParallelLinear` + `SiluAndMul` 并在加载时逆序拼接权重修复。

## 问题现象
OLMo 模型在 `tensor_parallel_size>1` 时输出不正确。触发条件:以 TP>1 部署/推理 OLMo 模型(修复前 `ff_proj` 完全没有张量并行实现)。正文未描述更细的日志或复现输出,仅说明该错误输出问题对应 issue #3775。

## 根因
正文披露:作者自述此前「忘了为 OLMo 模型的 `ff_proj` 实现 Tensor Parallelism」,即 FFN 输入投影层没有接 vLLM 的并行线性层机制,导致 TP>1 时输出错误。
(由 diff 推断)具体机制为:`ff_proj` 原先用普通 `ColumnParallelLinear`(输出维度为整个 `hidden_size`,且 OLMo 的 SwiGLU 是 gate/up 合并在一个投影里、各占一半),配合手写 `SwiGLU`(按最后一维 chunk 成两半做 `silu(gate) * x`);这种单列并行的切分方式无法与 OLMo「gate+up 合一投影」的结构在 TP 下正确对齐,故改为 `MergedColumnParallelLinear(config.d_model, [self.hidden_size // 2] * 2)`(gate/up 各成 shard)+ 标准 `SiluAndMul`(`output_multiplier = 0.5`);同时 HF checkpoint 的 `ff_proj` 权重两半顺序与 `MergedColumnParallelLinear` 期望的 shard 顺序相反,`load_weights` 中需 `torch.concat(loaded_weight.chunk(2)[::-1])` 逆序后再加载(diff 注释「Reverse the weight for the MergeColumnParallelLinear」)。

## 修复方案
改 `vllm/model_executor/models/olmo.py`:
- 删除手写 `SwiGLU` 类,`self.act` 改用 vLLM 的 `SiluAndMul` 并设 `self.act.output_multiplier = 0.5`。
- `ff_proj` 由 `ColumnParallelLinear` 改为 `MergedColumnParallelLinear`:
  ```python
  self.ff_proj = MergedColumnParallelLinear(
      config.d_model,
      [self.hidden_size // 2] * 2,
      bias=config.include_bias,
      linear_method=linear_method,
  )
  ```
- `load_weights` 中把 `.ff_proj`/`.ff_out` 的重命名拆开处理,并对 `ff_proj` 权重做 shard 逆序:
  ```python
  if ".ff_proj" in name:
      name = name.replace(".ff_proj", ".mlp.ff_proj")
      # Reverse the weight for the MergeColumnParallelLinear
      loaded_weight = torch.concat(loaded_weight.chunk(2)[::-1])
  ```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:仅 OLMo 模型的 FFN 构建(`OlmoMLP`)与权重加载:TP=1 下 `MergedColumnParallelLinear` 两 shard 合并计算与原单一投影等价(结构不变,仅加载顺序调整);TP>1 下 `ff_proj` 现在按 gate/up 两 shard 正确切分;权重加载对含 `.ff_proj` 的 checkpoint 名做逆序拼接,对其他名字不变;删除的自定义 `SwiGLU` 只被 OLMo 使用。

## 关联
- [FIX #3775](https://github.com/vllm-project/vllm/issues/3775)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
