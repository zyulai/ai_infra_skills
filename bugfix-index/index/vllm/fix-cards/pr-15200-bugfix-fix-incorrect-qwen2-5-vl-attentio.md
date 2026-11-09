# PR #15200: [Bugfix] Fix incorrect qwen2.5-vl attention mask pre-computation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#15200](https://github.com/vllm-project/vllm/pull/15200) |
| 作者 | Isotr0py |
| 合并日期 | 2025-03-21 |
| labels | ready |
| 热度 | 👍 3 · 💬 5 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
qwen2.5-vl 窗口注意力的 cu_seqlens 元数据在层循环外按全注意力预计算,窗口层用错 seq 长度;修复为逐层按 cu_seqlens_now 重新计算。

## 问题现象
正文未描述具体现象,仅写明修复 #14377 引入的 qwen2.5-vl 窗口注意力 attention mask 错误构造。关联的 issue #15122 与 #15197 被此 PR 修复;测试中针对 #15122 使用了一幅特定图片(hato.jpg)配 `size_factor=[0.5]` 的输入在 Qwen2.5-VL-3B-Instruct(max_model_len=4096,max_num_seqs=2)上做回归验证,可推断该问题在特定图片尺寸/窗口划分下触发(由 diff 推断)。

## 根因
正文未披露根因,只说修复 #14377 的错误构造。从 diff 看,qwen2_5_vl.py 的 `forward` 原先在层循环之前一次性计算 `max_seqlen`(FLASH_ATTN)或 `seqlens`(XFORMERS),且数据源固定为全注意力的 `cu_seqlens`;但窗口注意力层应使用 `cu_window_seqlens`。两种层交错出现(fullatt_block_indexes 控制),窗口层拿到的 seq 长度元数据与实际窗口划分不符,导致 attention mask/元数据错误(由 diff 推断)。

## 修复方案
改 `vllm/model_executor/models/qwen2_5_vl.py` 的 `forward`:把 `max_seqlen`/`seqlens` 的预计算移入层循环内,按每层实际使用的 `cu_seqlens_now`(全注意力层用 `cu_seqlens`,窗口层用 `cu_window_seqlens`)逐层重新计算:

```python
# pre-compute cu_seqlens for window attn
if self.attn_backend == _Backend.FLASH_ATTN:
    max_seqlen = (cu_seqlens_now[1:] -
                  cu_seqlens_now[:-1]).max().item()
elif self.attn_backend == _Backend.XFORMERS:
    seqlens = (cu_seqlens_now[1:] - cu_seqlens_now[:-1]).tolist()
```

覆盖 FLASH_ATTN 与 XFORMERS 两个后端。另在 `tests/models/decoder_only/vision_language/test_models.py` 增加 `qwen2_5_vl-windows-attention` 回归用例及配套 custom input。

## 影响与回归风险
- 影响窗口:#14377 引入窗口注意力预计算之后的版本(正文未给出具体版本号)。
- 回归面:仅 qwen2.5-vl 视觉编码器的 attention 元数据构造;每层循环内多一次 `.max().item()`/`.tolist()`(FLASH_ATTN 路径有一次 GPU→CPU 同步,略增开销);全注意力层的取值与修复前相同,其他模型不受影响。

## 关联
Fixes #15122, #15197;修复 #14377 引入的回归

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
