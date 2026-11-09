# PR #42755: [Model][Hardware][AMD][Kernel] Part 2/2 -> Enable e2e QK Norm + RoPE + KV Cache runtime fusion for Qwen3-30B-A3B on ROCM_ATTN

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jhu960213 |
| 合并 | 2026-10-16 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready, torch.compile, v1, qwen |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「15 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/42755 |

## 一句话结论

ROCM_ATTN 上把 QK Norm+RoPE+KV Cache 融进 aiter HIP 内核，正文未给定量性能数字。

## 动机

正文说明本 PR 是 stacked PR 系列的 Part 2/2，依赖 #42749（Part 1）；因 Part 1 尚未合并，当前 diff 也含 Part 1 内容，待 #42749 合入后会 rebase 到 main 并只保留本 PR 的增量提交。该系列把原始 `jhu96/optimize-qwen30b` 分支（PR #39527）拆成两个可评审的 PR。目标是让 Qwen3-30B-A3B 在 ROCM_ATTN 上端到端启用 QK Norm + RoPE + KV Cache 的运行时融合，把多步算子收敛到 aiter 的高性能 HIP 内核。

## 优化点

在 compile 阶段把融合 pattern 换成 aiter 的 `fused_qk_norm_rope_cache_pts_quant_shuffle` HIP 内核，对满足预设 token compile range（由 `rope_kvcache_fusion_max_token_num` 守护）的所有 decoder layer 生效。难点在于 `shuffle=True` 时融合内核以交错格式写 K/V：K 形状为 `(blocks, heads, dim/x, block_size, x)`，V 形状为 `(blocks, heads, block_size/x, dim, x)`；decode 路径原本硬编码按 `(blocks, heads, dims, block_size)` 读 V cache。因此在 ROCM_ATTN 的 prefill Triton、面向非 2 的幂 block size 的 decode Triton 2D、以及 decode paged attention ll4 GFX9 变体上都要改为能读交错 V cache；K cache 读逻辑不动，因为融合内核已按 prefill/decode 期望格式写 K。功能通过 `--compilation-config` 的 `pass_config` 中 `fuse_qk_norm_rope_kvcache` 与 `rope_kvcache_fusion_max_token_num` 开关控制。

## 关键改动

```text
csrc/cache_kernels.cu: reshape_and_cache_kernel 增加 interleaved_v 参数，按交错格式计算 V cache 写入偏移与 stride
csrc/rocm/attention.cu: paged_attention_ll4mi_QKV_mfma16_kernel 增加 USE_INTERLEAVED_V_CACHE 模板参数与交错 V 读取
csrc/ops.h、csrc/torch_bindings.cpp、csrc/rocm/ops.h、csrc/rocm/torch_bindings.cpp: reshape_and_cache 签名增加 interleaved_v 绑定
tests/compile/passes/test_rocm_aiter_qk_norm_rope_kvcache_fusion.py: 融合 pass 单测
tests/kernels/attention/test_attention.py、test_prefix_prefill.py: attention 相关测试改动
```

```cpp
  if (interleaved_v) {
    tgt_value_start = v_head_base + (block_offset / x) * (head_size * x) +
                      (h_block * x) * x + (block_offset % x);
    v_dim_stride = x;
  } else {
    tgt_value_start = v_head_base + h_block * x * block_size + block_offset;
    v_dim_stride = block_size;
  }
```

## 性能声明

PR 正文未给出定量性能数据：Test Result 与 Recent Sweep 的结果均以截图形式呈现，无可读取的吞吐/延迟数值。测试条件方面，Recent Sweep 注明使用 ISL=1024、OSL=1024 的 balanced config，覆盖 3 个模型：Hy3（TP=8）、GLM4.7（TP=8）、Qwen3-30B-A3B（TP=1），并附 Accuracy 截图；硬件型号与精度 PR 未注明测试条件（标签与标题指向 AMD ROCm / ROCM_ATTN）。正文未列出具体 benchmark 数字，本卡无法给出性能结论。

## 验证与回归风险

Test Plan 给出启动方式：`VLLM_ROCM_USE_AITER=1`、`vllm serve /path-to-your/Qwen3-30B-A3B --attention-backend ROCM_ATTN --no-enable-prefix-caching -O1`，并在 `--compilation-config` 的 `pass_config` 中设 `fuse_qk_norm_rope_kvcache: true`、`rope_kvcache_fusion_max_token_num: 256`；单测命令为 `pytest tests/compile/passes/test_qk_norm_rope_kvcache_fusion.py -v`（diff 中对应文件名为 `test_rocm_aiter_qk_norm_rope_kvcache_fusion.py`）。可能影响的路径：ROCM_ATTN 的 prefill/decode 内核、`reshape_and_cache` 写 KV 的布局、paged attention GFX9 变体，以及开启该融合的所有 decoder layer；交错 V cache 读取是主要风险面，K cache 读逻辑未改。正文声明依赖 #42749，合并顺序需留意；pack 末尾注明 15 file(s) dropped from diff。

## 关联

正文明确依赖 #42749（Part 1），并说明源自把 PR #39527（`jhu96/optimize-qwen30b` 分支）拆分的 stacked 系列；未提及其他前序/后续 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
