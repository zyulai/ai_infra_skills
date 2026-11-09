# PR #38178: [CI] Fix conch kernel crash on 3D input by reshaping to 2D before GEMM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38178](https://github.com/vllm-project/vllm/pull/38178) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-03-26 |
| labels | rocm, ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
transformers backend 传 3D 输入使 conch 量化 GEMM 解包崩溃;先 reshape 成 2D 计算再还原输出形状。

## 问题现象
量化模型(如 AWQ)经 transformers backend 运行时 conch 内核崩溃:
```
ValueError: too many values to unpack (expected 2)
```
位于 `conch/ops/quantization/gemm.py:86` 的 `m_dim, k_dim = x.shape`。

## 根因
正文披露:transformers backend 把 3D 张量(batch, seq_len, hidden_dim)传入 linear 层,而 `mixed_precision_gemm` 期望 2D 输入 (M, K)。

## 修复方案
`vllm/model_executor/kernels/linear/mixed_precision/conch.py` `apply_weights`:调用前把 `x` reshape 为 2D,输出再 reshape 回原 batch 维度,与 machete 内核的既有模式一致:
```python
x_2d = x.reshape(-1, x.shape[-1])
out_shape = x.shape[:-1] + (self.config.partition_weight_shape[1],)

output = mixed_precision_gemm(
    x=x_2d,
    w_q_packed=w_q.data,
    ...
)
...
return output.reshape(out_shape)
```
测试:`pytest tests/models/test_transformers.py::test_quantization -v` 通过。

## 影响与回归风险
- 回归面:conch mixed-precision 量化 GEMM 的 apply 路径(transformers backend 下的量化模型);纯形状适配,数学计算与输出值不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
