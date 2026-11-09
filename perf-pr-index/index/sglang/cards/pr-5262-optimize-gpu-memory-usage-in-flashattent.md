# PR #5262: Optimize GPU memory usage in FlashAttentionBackend's strided indexing

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @CatherineSue |
| 合并 | 2025-04-11 |
| 分类 | Attention / Kernel(tier: major) |
| Label | high priority |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 全量保留(单文件单处改动) |
| 链接 | https://github.com/sgl-project/sglang/pull/5262 |

## 一句话结论

修掉 FlashAttention CUDA graph replay 元数据里 `[total_requests, max_seq_pages]` 的中间张量分配,改为只为当前 batch 元素做两步索引,降低 GPU 内存占用。

## 动机

`init_forward_metadata_replay_cuda_graph` 中原有的切片策略:先用 `self.req_to_token[:, strided_indices[:max_seq_pages]]` 对全表做广播式索引,生成一个 `[total_requests, max_seq_pages]` 的中间张量,之后再用 `page_indices[req_pool_indices]` 挑出当前 batch——这意味着每层 replay 都要为一个覆盖所有请求、最大页数的完整张量分配显存,而实际只有 batch 内的请求被使用。

## 优化点

把"全表广播切片 + 二次高级索引"改为两步直接索引:行方向用 `req_pool_indices[:, None]`、列方向用 `strided_indices[:max_seq_pages][None, :]` 交叉索引,使结果直接就是 `[batch_size, max_seq_pages]`,只为本 batch 需要的元素分配内存;除法改用 in-place 的 `page_indices //= self.page_size`,又省掉一次除法结果张量。随后的 `metadata.page_table[:, :max_seq_pages].copy_(page_indices)` 与尾部 fill(0) 逻辑保持不变。核心收益是消除中间张量的峰值显存与分配开销,不改任何语义。

## 关键改动

```text
python/sglang/srt/layers/attention/flashattention_backend.py:
  page_indices = self.req_to_token[req_pool_indices[:, None],
                  strided_indices[:max_seq_pages][None, :]]
  page_indices //= self.page_size   # 原为全表切片后 page_indices[req_pool_indices] // page_size
```

## 性能声明

PR 未给出 benchmark 数字,正文仅说明动机是旧实现会创建 `[total_requests, max_seq_pages]` 的中间张量、新实现只为所需 batch 元素分配内存;未注明测试条件。

## 验证与回归风险

依赖原仓库既有单测与 CI(checklist 勾选了 unit tests/documentation/benchmark 项,但正文未附具体数字)。风险面很小:改动仅在 CUDA graph replay 的 page table 元数据构建路径,语义等价(同一批 req_pool_indices 行 + 同样的 strided 列);需要确认的是高级索引下 `req_pool_indices` 与 `strided_indices` 的形状/设备约定不变,以及 in-place `//=` 不影响 `page_indices` 其他引用(此处为新建张量,无别名问题)。

## 关联

同区域:同分类 Attention / Kernel 其他 PR(FlashAttention backend 相关);后续 FlashMLA/FlashInfer backend 的 metadata 优化(如 PR #31927 的 counter buffer 常驻化)属同类"避免 replay 路径临时分配"思路。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
