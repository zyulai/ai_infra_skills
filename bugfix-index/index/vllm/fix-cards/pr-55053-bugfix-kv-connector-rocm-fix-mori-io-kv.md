# PR #55053: [Bugfix][KV Connector][ROCm] Fix MoRI-IO KV block-offset for speculative decoding (MTP)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#55053](https://github.com/vllm-project/vllm/pull/55053) |
| 作者 | MohitAMD |
| 合并日期 | 2026-10-07 |
| labels | bug, kv-connector, ready, rocm |
| 热度 | 👍 0 · 💬 25 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
MTP 下 MoRI-IO PD 分离的生产者预留 lookahead 块导致 KV 传输块数不匹配报 ValueError,WRITE 保存路径按 prompt 长度裁剪修复。

## 问题现象
在投机解码(如 MTP)叠加 MoRI-IO PD-disaggregation 时,prefill(producer)侧 KV-cache 管理器会为 `num_speculative_tokens` 预留 lookahead 块,而 decode(consumer)侧从不分配这些块,生产者本地块列表比消费者远端列表更长,第一次 prefill→decode KV 写入即崩溃:抛出 `ValueError: local_block_ids longer than remote_block_ids`(如 n=3 时 8003 vs 8000),MTP 在 MoRI-IO 上完全不可用。chunked prefill 场景还存在最终 chunk 不分配新块(`new_block_ids is None`)时保存被跳过、decode 侧挂起的问题(由 diff 测试用例描述佐证)。

## 根因
正文披露:生产者按投机 lookahead 预留的块是尾部草稿暂存块,不是 prompt KV,本不该进入传输清单;但 WRITE 模式保存路径没有裁剪它们,导致本地块列表带上了多余的尾部块。旧的最终 chunk 判定按「已积累块数 × block_size ≥ num_prompt_tokens」计数,lookahead 块会把块数撑大,并且最终 chunk 若不分配新块(`new_block_ids is None`)就根本走不到保存分支。

## 修复方案
改 `vllm/distributed/kv_transfer/kv_connector/v1/moriio/` 下 `moriio_connector.py`(与 `moriio_layout.py` 注释精简):
1. 新增 `_clamp_to_prompt_blocks`:WRITE 保存前把每个全注意力 KV 组裁剪到 `ceil(num_prompt_tokens / block_size)` 个 prompt 块,按组各自 block_size 生效,滑动窗口组不动。
2. 新增 `_is_final_prefill_chunk`:以 token 进度(`num_computed_tokens + num_scheduled >= num_prompt_tokens`)判定最终 chunk,替代块数计数,对 lookahead 膨胀免疫,并覆盖最终 chunk 不分配新块的情况。
3. `build_connector_meta` 相应重排:先累积/跳过,再在最终 chunk 统一 clamp 后保存;非 disagg 请求在 `new_block_ids` 为 None 时也安全跳过。
4. 保持 `compute_block_transfer_offsets` 对「本地比远端长」的响亮报错(READ 模式下这是真 bug),不静默钳制。
5. 失败关闭(fail closed):WRITE 模式 + 滑动窗口注意力(无论有无 spec decode)直接抛 `NotImplementedError`,因为按位置保存 prompt 前缀会发出窗外空块;READ + SWA 不受影响。

关键 diff 摘录:
```diff
+            num_prompt_blocks = math.ceil(
+                req.num_prompt_tokens / self._group_block_sizes[g]
+            )
+            if len(group_blocks) > num_prompt_blocks:
+                clamped[g] = group_blocks[:num_prompt_blocks]
...
-            if req.num_prompt_tokens > tokens_covered:
-                # not last chunk prefill
+            if not self._is_final_prefill_chunk(req, scheduler_output):
...
+            save_block_ids = self._clamp_to_prompt_blocks(req, block_ids)
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本,验证基于 vLLM v0.27.1 + AITER v0.1.19。
- 回归面:仅 MoRI-IO connector 的 WRITE 保存路径与构造期校验;READ 模式的部分传输行为与「本地更长」报错语义保留(有专门测试);新增的 NotImplementedError 会让「WRITE + SWA + spec」组合在启动时失败(原先也是不正确行为);chunked prefill 非 disagg 请求的跳过逻辑改为先查 `_reqs_need_pending_save`,涉及该分支的所有调用路径。

## 关联
正文未列出具体 issue/PR 链接(文中提及"上游 hybrid/multi-KV-group MoRI-IO refactor"但未给编号)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
