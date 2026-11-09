# PR #48440: [Perf] [AMD] moe: Run the small-batch sort + stage-1 quant up to 1024 pairs on Kimi-K3

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @RolaoDenthu |
| 合并 | 2026-11-04 |
| 分类 | MoE(major) |
| Label | jit-kernel |
| Diff | pack 未提供 diffstat(diff 正文可用,单文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/48440 |

## 一句话结论

把 Kimi-K3 MoE 小批量 sort + stage-1 量化的 patched sort 上限从 256 对提到 1024 对,MI355X 上该区间比 aiter 路径快 31-46%。

## 动机

Kimi-K3 的 MoE(E=896,top-16,block_m 32)在小 batch 时走一个同时把 stage-1 输入量化到 MXFP8 的 patched sort kernel。此前该路径最多覆盖 256 个 (token, expert) 对,即 top-16 下只有 16 个 token;更大的 batch 回退到 aiter 的 opus sort + quant。而在 MI355X 上,patched sort 在 960 对以内比 aiter 的 sort + quant 快 31-46%,直到 1152 对才打平(其 bitonic 宽度超过 1024 后翻倍)。因此 256 的上限让 17-64 token 的 decode/verify batch 一直留在慢路径上。

## 优化点

改动很小但机制清晰:按 shape 区分上限。`moe_sorting_small.py` 把 `_EMIT_MAX_PAIRS` 重命名为 `_SMALL_SORT_MAX_PAIRS`(默认仍为 256),新增 `_KIMI_K3_MOE_SHAPE = (896, 16)` 与 `_KIMI_K3_SMALL_SORT_MAX_PAIRS = 1024`;`_small_sort_supported` 按 `(num_experts, topk)` 是否命中 Kimi-K3 的 shape 选择上限,命中时放宽到 1024 对(top-16 下 64 token),其他 shape 保持 256(作者注明其他 shape 在该区间未测,不盲目放宽)。同时统一了原先分开硬编码的 `256` 检查:选 patched sort 与其 stage-1 量化用同一个常量,`fused_moe_wrapper` 的 emit 门控放宽到 1024 并由 `_small_sort_supported` 的 per-shape 限制兜底。超过 1024 对的 batch 仍走 aiter 的 sort + quant。

## 关键改动

```text
python/sglang/kernels/ops/moe/moe_sorting_small.py:
  _small_sort_supported: 按 (num_experts, topk)=(896,16) 放宽 p <= 256 → p <= 1024
  fused_moe_wrapper: emit 门控从 _EMIT_MAX_PAIRS(256)提到 1024
```

## 性能声明

Kernel 级:MI355X 上 patched sort 相对 aiter sort + quant,至 960 对快 31-46%,1152 对起打平。端到端:Kimi-K3.1,InferenceX fixed-seq 8k/1k,每 GPU 吞吐——concurrency 4(TP8)+2.3%,concurrency 8(DCP8)+1.1%。PR 未注明端到端测量的其他条件(如重复次数)。

## 验证与回归风险

正确性:patched sort 在 384-1024 对区间的输出(sorted ids、weights、expert ids 与量化后的 stage-1 输入)与 aiter 逐字节相同,模型输出不变。PR 未新增单元测试(checklist 中 unit tests / format 未勾选,三项 CI 状态为失败)。风险面:非 Kimi-K3 shape 严格保持原 256 上限,不受影响;命中 Kimi-K3 shape 但 expert_mask / num_local_tokens 非空的分布式路径仍被 `_small_sort_supported` 的其余条件排除;emit 门控放宽后,若 `_small_sort_supported` 与 wrapper 门控条件不同步,可能出现 patched sort 不支持但已标记 emit 的组合(diff 中由 `_small_sort_supported` 统一裁定的注释说明此点)。

## 关联

同区域:AMD MoE 路径相关 #4053(aiter fused moe)、#39273(AITER MoE runner);与 aiter sort + quant 的回退关系互为前序/后续。pack 未列出其他具体 PR 编号。
