# PR #13736: [VLM] Replace torch.repeat_interleave with faster np.repeat for Qwen-VL series

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-11-22 |
| 分类 | Model-specific Fast Path(major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/13736 |

## 一句话结论

Qwen-VL 系列用 np.repeat 替换 torch.repeat_interleave 计算 cu_seqlens,E2E TTFT 降低 1.5%。

## 动机

Qwen2-VL / Qwen3-VL / Qwen3-VL-MoE 在 vision transformer 前向中用 `torch.repeat_interleave` 从 `grid_thw` 计算 `cu_seqlens`。这一小段 CPU 侧张量操作是 TTFT 路径上的可观测开销,作者观察到换成 NumPy 实现更快,值得为 Qwen-VL 系列建立专门的快路径。

## 优化点

新增 `compute_cu_seqlens_from_grid_numpy` 工具函数:要求输入 `grid_thw` 是 CPU tensor,先转 `numpy()`,用 `np.repeat(arr[:, 1] * arr[:, 2], arr[:, 0]).cumsum(dtype=np.int32)` 计算 cu_seqlens,再拼上首位的 0 并通过 `torch.from_numpy` 转回 int32 tensor。相比 torch 版本省去了 GPU/dtype 转换的多步张量操作。Qwen2-VL 与 Qwen3-VL 的 forward 中原 `torch.repeat_interleave + cumsum + cat` 的多行实现被替换为对该函数的单行调用。

## 关键改动

```text
python/sglang/srt/models/utils.py: 新增 compute_cu_seqlens_from_grid_numpy,CPU 上用
  numpy 完成 repeat+cumsum,返回 int32 torch tensor
python/sglang/srt/models/qwen2_vl.py: forward 中 cu_seqlens 计算改走新函数
python/sglang/srt/models/qwen3_vl.py: 同上
test/srt/ops/test_repeat_interleave.py: 新增正确性测试(CPU 随机用例 + 零 repeat 边界)
test/srt/run_suite.py: 将 ops/test_repeat_interleave.py 加入测试套件
```

关键摘录(新函数核心逻辑):

```python
arr = grid_thw.numpy()
cu_seqlens = np.repeat(arr[:, 1] * arr[:, 2], arr[:, 0]).cumsum(
    axis=0, dtype=np.int32
)
cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
cu_seqlens = torch.from_numpy(cu_seqlens)
```

## 性能声明

- E2E TTFT 减少 1.5%(PR 正文未注明测试条件:模型规模、硬件、workload 均未说明)。
- 函数级 CPU benchmark(`test_repeat_interleave.py`,迭代 2000 次):T=128 时 torch 54.57 ms vs numpy 18.58 ms;T=512 时 61.48 vs 26.38 ms;T=2048 时 92.67 vs 65.20 ms;T=8192 时 299.10 vs 279.42 ms。PR 未注明该测试的 CPU 具体型号。

## 验证与回归风险

新增 `test/srt/ops/test_repeat_interleave.py`:与纯 torch 参考实现对比输出 shape/dtype/值(4 种 T 配置 x 10 组随机用例),并覆盖 repeat=0 的边界情况,已加入 `run_suite.py` 测试套件。PR 正文提到 "gsm8k matched",但未给出更系统的精度对比。回归风险:新函数通过 assert 强制要求 `grid_thw` 在 CPU 上,若上游调用方式改变(传入 GPU tensor)会直接报错;影响面限于 Qwen2-VL/Qwen3-VL/Qwen3-VL-MoE 的 cu_seqlens 计算路径。

## 关联

无(pack 内未提及同区域其他 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
