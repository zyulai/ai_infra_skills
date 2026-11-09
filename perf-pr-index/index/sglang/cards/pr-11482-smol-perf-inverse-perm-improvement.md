# PR #11482: [smol] [perf] Inverse perm improvement

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @vincentzed |
| 合并 | 2025-10-21 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat（diff 内容可用：2 个文件） |
| 链接 | https://github.com/sgl-project/sglang/pull/11482 |

## 一句话结论

Qwen2.5-VL 的 window 重排还原用 scatter 式 `permute_inv` 替代 `torch.argsort`，该模型 e2e 快约 2%。

## 动机

Qwen2.5-VL 的 ViT 路径按 `window_index` 重排 patch 序列后，需要求逆置换把序列还原。旧实现用 `torch.argsort(window_index)` 求逆，比直接 scatter 赋值慢；这是 ViT 前向里每步都跑的固定开销，值得换成 O(n) 的索引写入实现。

## 优化点

在 `models/utils.py` 新增 `permute_inv(perm)`：`inv_perm[perm] = arange(numel)`，一步 scatter 完成逆置换，替代 `argsort`（排序 + 间接寻址）。`qwen2_5_vl.py` 的 `forward` 把 `reverse_indices = torch.argsort(window_index)` 移到重排之前直接用 `permute_inv(window_index)` 计算，在 merger 输出后直接索引使用。计算结果语义相同，不改变模型输出。

## 关键改动

```text
python/sglang/srt/models/utils.py: 新增 permute_inv 工具函数（scatter 求逆置换）
python/sglang/srt/models/qwen2_5_vl.py: ViT forward 用 permute_inv 替代 torch.argsort(window_index)
```

关键代码（摘录）：
```python
def permute_inv(perm: torch.Tensor) -> torch.Tensor:
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv_perm
```

## 性能声明

PR 正文仅一句：在 Qwen2.5-VL-3b 上比 plain 实现 e2e 快 2%（"2% faster e2e than plain implementation in my test on Qwen2.5-VL-3b"）。无硬件、精度、workload 详情——PR 未注明测试条件；正文其余 Benchmarking/Profiling 章节留空。

## 验证与回归风险

PR 未提供 accuracy 测试与新增单测（Checklist 均未勾选）。`permute_inv` 语义与 `argsort` 等价（当 perm 为合法置换时），风险主要是该工具函数若被复用于非置换输入，结果与 argsort 不同；仅影响 Qwen2.5-VL 路径。

## 关联

PR 正文未提及同区域其他优化；无直接关联 PR（写"无"亦可，此处保留 CodeRabbit 摘要中指出的范围：仅 Qwen2.5-VL 路径）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
