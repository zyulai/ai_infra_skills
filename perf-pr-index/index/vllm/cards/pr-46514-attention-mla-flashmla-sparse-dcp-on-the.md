# PR #46514: [Attention][MLA] FlashMLA sparse: DCP on the fp8_ds_mla mixed-batch path + MTP

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @drakosha |
| 合并 | 2026-08-19 |
| 分类 | Speculative Decoding（tier: major） |
| Label | documentation, ready, v1, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 未标注 dropped 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/46514 |

## 一句话结论

让 `fp8_ds_mla` 稀疏路径支持 DCP+MTP，4×H200 上 batch32 聚合 854 tok/s。

## 动机

正文说明 `fp8_ds_mla` 是 backend selector 在 Hopper 上为每 rank head 数较少的 DSA 模型（GLM-5.2 / DeepSeek-V3.2，TP4/DCP4）所选的 KV 路径，而 `FLASHMLA_SPARSE` 后端此前直接拒绝 DCP，导致 MTP spec decode 无法在该路径上与 DCP 共存。本 PR 为该后端补上 DCP 支持，并使其在 full cudagraphs 下可与 MTP 一起工作。早期版本堆叠在 #46076 上并携带 sparse-indexer 的 DCP 机制（global top-k merge、`--dcp-sparse-indexer-mode`），这些已通过 #46076 及后续 PR 落入 main，故当前 PR 只剩 backend 侧改动（正文自述 +234/−34，涉及 `flashmla_sparse.py`、`flashmla.py` 与一个测试文件）。

## 优化点

`flashmla_sparse.py` 让后端接入共享的 DCP decode merge（`can_return_lse_for_decode`）：fp8 mixed-batch 内核现在返回 decode LSE，与 head padding 一起切片并 reshape 成 reducer 消费的 `(tokens, heads)` 布局。DCP 下 indexer 产出 global token ids，本 rank 保留自己的 shard 并用 `triton_filter_and_convert_dcp_index`（`compact_valid_to_front=False`）转换成本地 `fp8_ds_mla` slot，散落的 `-1` 原地保留由 fp8 内核原生 mask，省去每次 front-pack；本 rank 未持有任何被选 token 的行返回 `(0, -inf)`，即跨 rank LSE merge 的单位元。允许 DCP 下 varlen decode（`supports_dcp_with_varlen` 当 `cp_kv_cache_interleave_size == 1`），因果性由 indexer top-k 索引而非 kernel metadata 决定，这正是 MTP+DCP 在 full cudagraphs 下可用的关键。另加 fail-closed 守卫：仅 `ag_rs` comm backend、仅 fp8 mixed-batch 路径（split prefill/decode 路径与 bf16 sparse 路径在 DCP 下被拒）、以及 head-envelope 守卫拒绝 local 与 DCP-gathered head 数 padding 不一致的配置（如 128 q-heads 的 TP8/DCP8）。`flashmla.py` 把 dense FlashMLA decode 的 `[batch, heads, seq_len]` LSE 展平为 `[tokens, heads]`，在 spec decode（`seq_len > 1`）下尤为必要。

## 关键改动

```text
vllm/v1/attention/backends/mla/flashmla_sparse.py: 接入 DCP decode LSE merge、DCP index 本地化、varlen 支持与 fail-closed 守卫
vllm/v1/attention/backends/mla/flashmla.py: 把内核 LSE 由 [batch, heads, seq_len] 展平为 [tokens, heads] 供 DCP reducer 使用
tests/v1/attention/test_sparse_mla_backends.py: 新增 test_fp8_dcp_head_envelope_guard 与 test_fp8_mixed_batch_dcp_neutralizes_empty_rows
```

```python
        if self.need_to_return_lse_for_decode:
            # FlashMLA returns LSE as [batch, heads, seq_len]; the DCP reducer
            # consumes [tokens, heads]. Flattening matters under spec-decode,
            # where seq_len > 1. Only DCP consumes lse, so skip the copy
            # otherwise.
            num_decodes, q_num_heads, seq_len = lse.shape
            lse = (
                lse.permute(0, 2, 1)
                .reshape(num_decodes * seq_len, q_num_heads)
                .contiguous()
            )
```

## 性能声明

PR 正文 Testing 段给出当前 rebase 的实测（测试条件：4×H200、GLM-5.2-NVFP4、TP4/DCP4/EP + MTP、`fp8_ds_mla`、SM90）：

- Warm decode 78 tok/s at batch 1，854 aggregate at batch 32（786k context window）
- `test_sparse_mla_backends.py` + `test_indexer_dcp_localize.py`：140 passed / 0 failed (SM90)
- Full cudagraph capture clean；198k context 下所有 rank 精确 needle recovery
- dcp=1/2/4 在长（>2048-tok）上下文 byte-identical（同 merge 路径的早期版本）

正文另注明该分支自 7 月中旬起在生产的 4×H200 GLM-5.2-NVFP4（TP4/DCP4/EP + MTP）部署上持续 soak。正文 `<details>` 的 History 段还列出前 rebase 版本的数字（exact merge 相对独立 indexer #45426 有约 30% batch-1 decode 回退：52.0 → 36.4 tok/s，union 恢复到 43.5、localizer 优化到 50.7），但正文明确标注这些数字 "no longer describe the current diff"，属已废弃历史，不作为本卡当前性能结论。

## 验证与回归风险

测试：正文点名 `test_fp8_dcp_head_envelope_guard`（在 `test_sparse_mla_backends.py` 中，按 local head 数 × DCP world size 参数化，含接受与拒绝两类用例），diff 中另有 `test_fp8_mixed_batch_dcp_neutralizes_empty_rows` 验证空行返回 `(0, -inf)` 以免 NaN 污染 merge；整体 140 passed / 0 failed (SM90)。可能受影响的路径：`fp8_ds_mla` 稀疏 decode、DCP 下的 cross-rank LSE merge、MTP 与 full cudagraphs 组合；正文以 fail-closed 守卫明确排除 `ag_rs` 之外的 comm backend、非 fp8 mixed-batch 路径、bf16 sparse 路径及 head-envelope 不一致配置（如 TP8/DCP8 的 128 q-heads），这些是被拒绝而非回归的场景。正文亦列出周边 DCP infra 的关联修复另行拆分处理。

## 关联

正文明确引用：#46076（sparse-indexer DCP 机制已落 main，本 PR 复用其 upstream merge machinery）、#45426（更早的独立 indexer 版本，被本 PR supersede）、#48379（fp8_ds_mla boot on main，merged）、#50823（DCP 下 block-table width，merged）、#50883（DCP 下 offload block-size scaling，open）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
