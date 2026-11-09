# PR #39616: [ROCm][Feature] Enable AITER MLA attention backend to work with Eagle3 speculative decoding on ROCm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @larryli2-amd |
| 合并 | 2026-04-20 |
| 分类 | Speculative Decoding（tier: major） |
| Label | documentation, rocm, ready, v1 |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失）；主体改动集中在单文件 `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` |
| 链接 | https://github.com/vllm-project/vllm/pull/39616 |

## 一句话结论

解除 AITER MLA 与 Eagle3 的 kernel_block_size 冲突：AMD 最快 MLA 后端叠加 Eagle3 spec7 后 MI355X 吞吐 +73–77%，baseline 不回退。

## 动机

AITER MLA（经 aiter 库调用 AMD ASM kernel）是 MI300X/MI355X 上最快的 MLA attention 后端，但无法与 Eagle3 等投机解码同用：AITER MLA 声明 `get_supported_kernel_block_sizes()=[1]`，Eagle3 draft 模型（flash_attn）要求 block_size 为 16 的倍数，两者共享同一 KV cache group，`select_common_block_size()` 无解。用户被迫二选一：AITER MLA（decode 快、无投机）或更慢后端（可投机但 baseline 低）。

## 优化点

关键洞察：aiter MLA decode kernel 内部永远以 page_size=1 运行——vLLM wrapper 在调用 ASM kernel 前把 `kv_buffer` `.view(-1, 1, 1, head_size)` 展平，kernel 从不感知多 token page；因此可支持任意 kernel_block_size，只需在 metadata builder 中把块级索引展开为逐 token 的扁平索引。四点改动（单文件）：

1. `get_supported_kernel_block_sizes()`：`[1]` → `[MultipleOf(1)]`，允许任意块大小。
2. metadata builder 记录 `self.kernel_block_size = kv_cache_spec.block_size`；flat view 下每 token 即一页，`max_num_pages_per_req = max_model_len` 与 block size 无关。
3. `_copy_page_indices_kernel` 重写为 `_expand_page_indices_kernel`：块表条目 b（覆盖 K 个 token）展开为扁平索引 `b*K .. b*K+(K-1)`；`K=1` 时数学上退化为直接拷贝，与原 kernel 完全一致（无回归）。
4. `get_mla_metadata_v1` 仅在 `max_qo_len == 1`（单 token decode）时调用，结果经 metadata dataclass 上的 `has_persistent_metadata` 传递；Eagle3 verification 时 `qseqlen = num_speculative_tokens + 1 > 1`，`get_mla_metadata_v1` 内部的 `get_heuristic_kernel_mla` 会在 qseqlen>1 时崩溃，跳过后 kernel 自行计算 metadata（等同 v0.18.0 的回退路径）。

## 关键改动

```text
vllm/v1/attention/backends/mla/rocm_aiter_mla.py: MultipleOf(1) 声明 + kernel_block_size 记录 + 索引展开 kernel + 条件 metadata
docs/design/attention_backends.md: ROCM_AITER_MLA 的 block size 由 1 改为 %1
```

```python
# When KERNEL_BLOCK_SIZE=K: block table entry b (covering K tokens)
# is expanded to flat indices b*K, b*K+1, ..., b*K+(K-1).
block_idx = token_offsets // KERNEL_BLOCK_SIZE
offset_in_block = token_offsets % KERNEL_BLOCK_SIZE
block_ids = tl.load(row_ptr + block_idx, mask=mask)
flat_indices = block_ids * KERNEL_BLOCK_SIZE + offset_in_block
```

## 性能声明

（AMD MI355X，TP=4，amd/Kimi-K2.5-MXFP4（MLA，64 heads）+ lightseekorg/kimi-k2.5-eagle3 draft，ISL=1024/OSL=1024，random 数据集，每配置 10 seeds。）

- MML=2248, C=16：baseline 206.1 ± 3.7 → Eagle3 spec7 363.7 ± 20.2 tok/s/gpu（**+76.5%**）。
- MML=4096, C=16：204.2 ± 10.9 → 353.4 ± 27.2 tok/s/gpu（**+73.0%**）。
- Baseline（无投机）性能不变——无回归；输出质量经 10 个多样化 prompt（temperature=0，chat API）与 baseline 一致，spec-dec acceptance 35–55% 正常区间。

## 验证与回归风险

- 验证：baseline 吞吐不变、Eagle3 spec7 正确运行、输出质量 10 prompt 对照、acceptance 率正常、+73–77% 吞吐五项 checklist 全勾；docs 表格更新。
- 已知限制（PR 明示，非本 PR 引入）：aiter ASM kernel 对非 2 的幂 `max_qo_len` 有上游 bug（如 `num_speculative_tokens=5` → `qseqlen=6` 产生错误 attention 输出），仅 spec+1 为 2 的幂（1/3/7/15）可用；已上报 ROCm/aiter issue #2720。
- 回归风险：K=1 路径与原 kernel 逐位等价（直接拷贝），Triton kernel 改名但语义兼容；qseqlen>1 时 persistent metadata 被跳过、kernel 内部自算 metadata，性能特性回到 v0.18.0 行为；`forward_mqa` 的 metadata kwarg 构造改为条件拼接。

## 关联

- 同区域：本批 PR #57057（UltraQuant 4-bit KV，同样面向 ROCm AITER attention 体系，示例命令即用 `ROCM_AITER_UNIFIED_ATTN` 后端）；本批 Speculative Decoding PR #65874（fused GDN MTP）、PR #60695（Mooncake EAGLE proof）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
