# PR #38359: [Bugfix] Revert "Zero-init MLA attention output buffers to prevent NaN from CUDA graph padding"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38359](https://github.com/vllm-project/vllm/pull/38359) |
| 作者 | elvircrn |
| 合并日期 | 2026-04-01 |
| labels | bug, nvidia, ready, v1 |
| 热度 | 👍 1 · 💬 0 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
回滚 MLA decode 输出缓冲的零初始化改动,恢复 `torch.empty` 分配,消除预分配缓冲开销与 FlashInfer `out=` workaround。

## 问题现象
正文未描述具体现象。PR 本质是对此前「Zero-init MLA attention output buffers to prevent NaN from CUDA graph padding」改动的整体回滚:正文 Summary 说明该改动带来了预分配零初始化缓冲的开销,并引入了 FlashInfer MLA 的 `out=` workaround,本 PR 将两者一并移除。测试计划要求在「GB200 DeepSeek-R1 NVFP4 decode with CUDA graph padding」场景验证无 NaN、且移除预分配缓冲后无性能回归,可侧面说明该改动运行在 GB200 DeepSeek-R1 NVFP4 的 CUDA graph padding 场景中。

## 根因
正文未披露根因,未解释为何回滚而不是修补。正文 Summary 仅披露回滚动机:恢复原始的 `torch.empty` 分配,消除预分配零初始化缓冲的开销以及 FlashInfer MLA 中的 `out=` workaround。
(由 diff 推断)被回滚的改动在 `cutlass_mla.py` 与 `flashinfer_mla.py` 的 decode 路径各自维护一个 `self._decode_out` 预分配输出缓冲:首次调用时按 `(B, MAX_HEADS, kv_lora_rank)` 形状 `new_zeros`/`torch.zeros` 一次性零初始化,之后每次 CUDA graph replay 复用该缓冲以避免重复 memset;在 FlashInfer 路径还因「FlashInfer `out=` 校验只接受 3D 形状、kernel 却写 4D」的已知 bug,只在 `q_len == 1` 时传 `out=`,`q_len > 1` 时改为 kernel 返回后将 `o[num_real:]` 手动清零。回滚即删除上述缓冲复用、形状判断与 padding 清零逻辑。

## 修复方案
整体 revert 两处 MLA 后端的零初始化缓冲逻辑:
- `vllm/v1/attention/backends/mla/cutlass_mla.py`:`_sm100_cutlass_mla_decode` 删除 `self._decode_out` 懒初始化缓冲复用,恢复每次直接分配:
  ```python
  out = q_nope.new_empty((B_q, MAX_HEADS, D_latent), dtype=dtype)
  ```
- `vllm/v1/attention/backends/mla/flashinfer_mla.py`:`forward_mqa` 删除 `out_kwargs`/`out=` 传递、`q_len > 1` 时 `o[num_real:] = 0` 的 padding 清零,以及 `is_quantized_kv_cache` 导入,恢复直接调用 `trtllm_batch_decode_with_kv_cache_mla` 不带 `out=`。

## 影响与回归风险
- 回归面:SM100 Cutlass MLA 与 FlashInfer(trtllm batch decode)MLA 的 decode 输出缓冲分配路径,以及依赖 `self._decode_out` 复用的 CUDA graph replay 场景;被回滚 PR 针对的「CUDA graph padding 槽位 NaN」问题是否回归,正文通过测试计划(GB200 DeepSeek-R1 NVFP4 decode + CUDA graph padding 验证无 NaN)覆盖,该数值正确性是主要关注点;另外 FlashInfer 路径不再绕过其 `out=` 形状校验 bug。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
