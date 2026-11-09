# PR #45589: [Bugfix] Fix MoE model load OOM in FlashInfer_TRTLLM  backend with sleep mode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @andakai |
| 合并 | 2026-06-17 |
| 分类 | MoE \| tier: major |
| Label | bug, ready, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/45589 |

## 一句话结论

修复 FlashInfer TRTLLM BF16 MoE + sleep mode 加载 OOM；dummy load 转换快约 4x（10.72s→2.39s）。

## 动机

#43951 报告用 `--enable-sleep-mode` 加载 `Qwen/Qwen3-235B-A22B-Instruct-2507` 时在权重加载阶段（CUDA graph profiling 之前）OOM，traceback 指向 `convert_moe_weights_to_flashinfer_trtllm_block_layout`，失败分配可小至 20 MiB。根因是 #41268 引入 `max_split_size_mb=20` 分配策略保护大块缓存以避免加载期碎片，而 FlashInfer TRTLLM BF16 MoE 转换会为每个 expert 反复创建 20–50 MiB 级临时张量（clone、permute 索引、reorder、contiguous、block-layout 输出、stack），在 128 experts × 94 层并开启 sleep mode 时经 CuMem weights pool 分配且不中途回收。实测 `max_split_size_mb=20` OOM、25/32/64/128 成功。正文给出的形状参数：num_hidden_layers=94、num_experts=128、hidden_size=4096、moe_intermediate_size=1536、TP=4、BF16，故 w13=[128,768,4096]（每 expert 6 MiB）、w2=[128,4096,384]（每 expert 3 MiB）；w13 的 uint8 视图约 6,291,456 元素、对应 int64 permute 索引约 48 MiB，w2 的约 3,145,728 元素对应约 24 MiB，正是超过 20 MiB 阈值的那一项。

## 优化点

不绕过也不调高全局 `max_split_size_mb`（作者认为会削弱 #41268 的保护，且阈值依赖模型形状与 TP，属 case-specific 规避）。改为减少转换本身的瞬时显存：预分配最终 `w13`/`w2` block-layout 输出张量，把每个 expert 视为 uint8，内联 `convert_to_block_layout`（reshape 成 block 后 permute 维度），用 `torch.index_select(..., out=...)` 把置换后的 expert 直接写入预分配输出，避免原先 clone / list / stack 中间件；并把 `swap_w13_to_w31` 折进 W13 的 permute 索引（`(permute_indices + rows // 2) % rows`），省去完整物化的 swapped W13。block-layout 变换仍执行，只是与行置换融合并直写目标张量。作者还说明 FlashInfer CUTLASS 与 Triton 后端不复现该问题，因为它们的 MoE 后处理不产生同样的重复瞬时分配；且去掉 #41268 的分配策略后会在 warmup 阶段再次 OOM，说明该策略仍然必要，因此选择减少转换本身的瞬时显存，而非改动全局阈值。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/flashinfer_utils.py: 重写 convert_moe_weights_to_flashinfer_trtllm_block_layout，预分配输出并 index_select 直写。
vllm/model_executor/layers/fused_moe/oracle/unquantized.py: 移除 FLASHINFER_TRTLLM 分支里额外的 swap_w13_to_w31 调用。
```

```python
def _copy_permuted_expert_to_block_layout(out, expert_uint8, source_indices):
    expert_blocks = expert_uint8.view(
        expert_uint8.shape[0], out.shape[0], block_k
    ).permute(1, 0, 2)
    torch.index_select(expert_blocks, 1,
                       source_indices.to(expert_uint8.device), out=out)

rows = w13_expert_uint8.shape[0]
permute_indices = (permute_indices + rows // 2) % rows
_copy_permuted_expert_to_block_layout(
    w13_weights_shuffled_tensor[i], w13_expert_uint8, permute_indices)
```

## 性能声明

Qwen3-235B TP=4、dummy load、强制 H20 通过 FlashInfer TRTLLM 能力门（SM100 gate）：旧转换 + sleep mode OOM；新转换 + sleep mode 加载完成，显存 109.61 GiB、load time 3.001113s（后续 H20 报 `No supported CUDA architectures found for major versions [10, 12]`）。无 sleep mode 对比：旧转换 10.723114s / 10.214007s，新转换 2.385791s，正文称该 dummy-load、转换密集场景下新路径约 4x faster。正确性：随机 BF16 张量、按 uint16 位比较，覆盖 Qwen-like TP4 全形状（w13=[128,768,4096]、w2=[128,4096,384]）、DeepSeek-like TP4 与 TP2 形状，结果 `ALL_BITWISE_EQUAL`。

## 验证与回归风险

复现命令：`CUDA_VISIBLE_DEVICES=0,1,2,3 ... serve Qwen/Qwen3-235B-A22B-Instruct-2507 --tensor-parallel-size 4 --enable-sleep-mode --load-format dummy --dtype bfloat16 --moe-backend flashinfer_trtllm --gpu-memory-utilization 0.93 --max-num-seqs 1024 --skip-tokenizer-init`；仅 `flashinfer_trtllm` 后端复现，Triton 与 `flashinfer_cutlass` 因后处理简单不受影响。作者说明 H20 通过强制 capability gate 只能验证加载/转换路径，FlashInfer TRTLLM BF16 MoE kernel 为 SM100/B 系列专用，最终运行时 kernel 正确性仍需在 B 系列硬件验证。

## 关联

#43951（所修 issue）；#41268（`max_split_size_mb=20` 分配策略）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
