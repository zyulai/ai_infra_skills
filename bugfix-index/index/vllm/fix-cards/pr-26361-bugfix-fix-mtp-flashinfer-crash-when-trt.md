# PR #26361: [Bugfix] Fix MTP+FlashInfer crash when trtllm kernels are available but disabled

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#26361](https://github.com/vllm-project/vllm/pull/26361) |
| 作者 | benchislett |
| 合并日期 | 2025-10-07 |
| labels | bug, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
强制禁用 trtllm attention 时 batch 重排阈值判断失配导致 MTP+FlashInfer 崩溃,在 can_use_trtllm_attention 中提前返回 False 修复。

## 问题现象
MTP+FlashInfer 场景下崩溃。触发条件:在 trtllm kernels 可用但被强制禁用(force disable trtllm attention 标志)时,`can_use_trtllm_attention` 返回 True 而 `use_trtllm_attention` 返回 False,二者不一致导致 padding 不正确,最终崩溃(即 #26312 观察到的崩溃)。

## 根因
正文披露:`reorder_batch_threshold` 的设置基于 `can_use_trtllm_attention`,默认假设「spec 启用且可用时 use_trtllm_attention 会返回 True」。但存在一个遗漏的边界情况:当 force disable trtllm attention 标志生效时,`can_use_trtllm_attention` 判定为 True,而 `use_trtllm_attention` 判定为 False,两者失配造成错误的 padding,进而引发 #26312 中的崩溃。

## 修复方案
修改 `vllm/utils/flashinfer.py` 的 `can_use_trtllm_attention`:在检查 kernels 可用性之前,先检查强制禁用标志,若 `force_use_trtllm_attention() is False` 则直接返回 False,使两个判定函数在禁用场景下保持一致。

关键 diff 摘录:
```python
def can_use_trtllm_attention(num_qo_heads: int, num_kv_heads: int) -> bool:
    """Check if the current configuration supports TRTLLM attention."""
    if force_use_trtllm_attention() is False:
        return False
    has_trtllm = supports_trtllm_attention()
    return has_trtllm and (num_qo_heads % num_kv_heads == 0)
```

## 影响与回归风险
- 回归面:仅改动 `can_use_trtllm_attention` 的判定逻辑,影响所有依赖该函数设置 `reorder_batch_threshold` 的路径(FlashInfer 后端、spec decode 相关 batch padding);显式强制禁用 trtllm 的配置现在会正确走非 trtllm 分支,其他场景行为不变。

## 关联
- [#26312](https://github.com/vllm-project/vllm/issues/26312)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
