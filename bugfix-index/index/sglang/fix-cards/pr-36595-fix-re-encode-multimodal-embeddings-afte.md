# PR #36595: [Fix] Re-encode multimodal embeddings after cache mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#36595](https://github.com/sgl-project/sglang/pull/36595) |
| 作者 | mickqian |
| 合并日期 | 2026-08-27 |
| labels | run-ci, bypass-maintenance |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

紧凑多模态特征哈希碰撞导致缓存 embedding 长度与占位符不匹配、prefill 中止;校验 token 数并驱逐重编码修复。

## 问题现象

紧凑的 multimodal feature-hash 发生碰撞时,缓存返回的 embedding 的 token 数与请求中占位符跨度不一致。该不匹配会一直传到 `_adjust_embedding_length`,使 prefill 以 `Insufficient multimodal embedding length` 报错中止。触发条件:使用 multimodal embedding 缓存且存在哈希碰撞(不同 token 数的输入命中同一缓存键);影响 chunked-prefill 的缓存命中路径。

## 根因

按 PR 正文:compact feature-hash 对不同 token 数的输入可能碰撞,缓存命中路径未校验缓存 embedding 的 token 数是否与条目偏移量一致,就把过期条目交给调度器,长度不匹配最终在 `_adjust_embedding_length` 处表现为不可恢复的 prefill 崩溃。此外批内按 hash 去重时,不同占位符长度的碰撞条目会被错误地合并为同一项。

## 修复方案

改 `python/sglang/srt/managers/mm_schedule.py`,并扩充 `test/registered/chunked_prefill/test_mm_chunked_embedding_unit.py`:

- 新增 `_embedding_token_count` 与 `_discard_mismatched_cached_embedding`,后者记录 warning 并调用 `embedding_cache.free` 驱逐坏条目。
- 在 `_get_chunked_embedding_full` 等全部 chunked-prefill 缓存命中路径上,校验缓存 embedding 的 token 数与 `items_offset` 推出的期望值;不一致则驱逐条目并重新编码,而不是传给调度器:

  ```diff
  +        expected_token_count = sum(end - start + 1 for start, end in items_offset)
  +        cached_token_count = _embedding_token_count(embedding_per_req.embedding)
  +        if cached_token_count != expected_token_count:
  +            _discard_mismatched_cached_embedding(...)
  +            embedding_per_req = None
  ```

- 批内临时映射 `_batch_encode_per_image_misses` 的键从 `hash` 改为 `(hash, token_count)`,同一批内 token 数不同的碰撞条目不再被去重;`_assemble_per_image_chunk` 按新键取 embedding。
- 新增 full、per-item、batched 三条路径的回归测试,包括模拟哈希碰撞(`items[1].hash = items[0].hash`)的去重用例。

## 影响与回归风险

- 影响窗口:正文未给出版本范围;基于 #32404 的多模态 chunked-prefill 缓存路径。
- 回归面:多模态 chunked-prefill 的全部缓存命中路径(full / per-item / batched)。碰撞时坏条目会被驱逐并重编码,带来一次额外的 ViT 编码开销(仅碰撞场景);缓存键变更只作用于批内临时映射,持久缓存仍按 hash 存取。验证仅完成 pre-commit 与 compileall(本地无 PyTorch/pytest),作者注明仍需 NVIDIA CI。

## 关联

- #32404(本 PR 基于该 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
