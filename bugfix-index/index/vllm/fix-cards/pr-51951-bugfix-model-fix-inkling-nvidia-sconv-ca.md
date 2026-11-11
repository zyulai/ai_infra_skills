# PR #51951: [Bugfix][Model] Fix Inkling NVIDIA sconv cache block-size mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51951](https://github.com/vllm-project/vllm/pull/51951) |
| 作者 | fjosw |
| 合并日期 | 2026-11-09 |
| labels | bug, inkling, nvidia, ready |
| 热度 | 👍 3 · 💬 11 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Inkling NVIDIA 路径用静态窗口 W 而非被 planner 扩大的 conv 块尺寸索引缓存,错页读写致 greedy 生成乱码;移植 AMD 的 `cache_block_size` 修复。

## 问题现象
作者在 Blackwell 上 benchmark Inkling-Small、调 KV cache block size 时发现:配 `--block-size 64` 时,`unify_kv_cache_spec_page_size` 会把 sconv conv group 的 block size 放大以匹配 attention page(此处 4 -> 8),但所有 NVIDIA call site 仍按静态 kernel 窗口 W=4 索引缓存。slot mapping 用放大后的尺寸构建,conv 的读/写因此落进错误页:greedy 生成在几个 token 内退化为垃圾输出。E2E 复现:Inkling-Small-NVFP4、TP=4、`--block-size 64`,greedy 补全 "The capital of France is Paris. The capital of Germany is",未修复输出坍缩成 "they they they..." 循环。

## 根因
正文披露:NVIDIA 侧各 call site 用静态 kernel 窗口尺寸(W)索引 conv 缓存,而混合 KV-cache planner 可能把逻辑 conv 块放大以对齐 attention page,slot mapping 按放大后的尺寸构建,两侧尺寸不匹配导致读写错页(测试注释还指出大缓存下甚至会越界产生 illegal memory access)。AMD 路径已通过 `InklingConvState.cache_block_size`(#48841)正确处理,本 PR 将其移植到 NVIDIA。

## 修复方案
在 `vllm/models/inkling/nvidia/sconv_swa_attn.py` 的 `InklingConvState` 上新增 `cache_block_size` property,并把 call sites 全部切换过去(`short_conv.py` 的 forward、`attention.py`、`model.py` 的 `_sconv_add_norm`),同时修正注释并新增测试 `test_short_conv_indexes_planner_enlarged_conv_blocks`:

```python
@property
def cache_block_size(self) -> int:
    """Return the block size used by cache metadata and physical indexing."""
    if self.kv_cache.numel() > 0:
        return self.kv_cache.shape[2]
    return self.block_size
```

即以绑定的物理缓存的运行时 token 维为准,而非 kernel 窗口尺寸。

## 影响与回归风险
- 影响窗口:正文未说明受影响版本/提交范围,不写。
- 回归面:按 diff,只改 NVIDIA Inkling 的 sconv 缓存索引:三处 call site(`short_conv.forward`、`attention.forward`、`model._sconv_add_norm`)从 `block_size` 换成 `cache_block_size`。在 planner 未放大块时 property 返回原 `block_size`,行为不变;仅当 conv 块被统一放大时索引才随之改变。AMD 路径与 kernel 本身不动。新增单元测试在放大场景下对齐参考实现输出与缓存内容。

## 关联
- [#48841](https://github.com/vllm-project/vllm/pull/48841)
- [#48799](https://github.com/vllm-project/vllm/pull/48799)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
