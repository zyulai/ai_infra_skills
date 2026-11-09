# PR #20390: [Fix #20389] Illegal memory access in triton attention for large token counts

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20390](https://github.com/sgl-project/sglang/pull/20390) |
| 作者 | akhilg-nv |
| 合并日期 | 2026-03-17 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
大 token 数下 triton attention 非法内存访问,把 qo_indptr 的 dtype 从 int32 升为 int64。

## 问题现象
使用 triton attention 后端时,大 token 数(large token counts)场景下出现 illegal memory access(CUDA 非法内存访问崩溃)。正文本身未展开症状细节,仅注明「解释与复现见 issue #20389」(https://github.com/sgl-project/sglang/issues/20389)。

## 根因
正文未披露根因,将解释与复现指向 issue #20389,PR 正文 Motivation/Modifications 留空。

从 diff 看,修复把 `triton_backend.py` 中非 skip_prefill 时预分配的 `self.qo_indptr` 从 `torch.int32` 改为 `torch.int64`。可推断机制是:`qo_indptr` 作为指向 KV/token 位置的索引指针数组,在大 token 数下累计值超出 int32 表示范围,导致 triton kernel 用溢出索引访存而触发 illegal memory access;升为 int64 消除溢出(由 diff 推断)。

## 修复方案
改动 `python/sglang/srt/layers/attention/triton_backend.py` 的 `__init__`,共一行:

```python
if not self.skip_prefill:
    self.qo_indptr = torch.zeros(
        (max_bs + 1,), dtype=torch.int64, device=model_runner.device
    )
```

即把预填充路径使用的 `qo_indptr` 缓冲 dtype 由 `torch.int32` 提升为 `torch.int64`;相邻的 `mask_indptr` 等其他缓冲未改动。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅影响 triton attention 后端、非 skip_prefill 的 `qo_indptr` 缓冲分配;int64 缓冲占用显存翻倍(每 bs 8 字节,量级极小),使用该 indptr 的 kernel 需兼容 int64 索引输入。skip_prefill 路径与 decode 路径不受影响。

## 关联
- [#20389](https://github.com/sgl-project/sglang/issues/20389)(问题解释与复现所在 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
