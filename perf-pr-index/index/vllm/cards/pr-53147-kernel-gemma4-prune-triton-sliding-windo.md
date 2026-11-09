# PR #53147: [Kernel][Gemma4] Prune Triton sliding-window tiles for multimodal prefixes

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mobicham |
| 合并 | 2026-08-31 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 含 3 个文件(helper、kernel、测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/53147 |

## 一句话结论

修复 Gemma4 多模态 prompt 下 Triton attention 丢失 sliding-window tile 剪枝的回归,RTX PRO 6000 上 E2E 快 3.23x、server prefill 快 4.15x。

## 动机

Triton attention 通常会剪掉 sliding window 之外的 KV tile,但当 multimodal prefix ranges(图像 token 的双向 attention 范围)存在时,compute_tile_loop_bounds 直接放弃剪枝扫描整个序列——代码中留有 `TODO(Isotr0py): sliding window pruning with image bidirectional mask`。对 Gemma4 这类多图长 prompt(均值约 16.5k token),每个 query block 都全量扫描 KV,prefill 严重退化。

## 优化点

在 compute_tile_loop_bounds 引入 MM_PREFIX_CLAMP_SW(Gemma4 的 window-clamped multimodal prefix mask 语义):prefix range 本来会 override sliding window 要求全序列扫描,但 Gemma4 把 prefix attention 限制在滑动窗口左缘,因此 tile 范围可取"常规 SWA 窗口"与"与本 query block 相交的各 image range 端点"的并集——kernel 内逐 range 读 mm_prefix_range_ptr(按 seq_idx * MAX_MM_RANGES * 2 寻址),相交时 first_allowed_key 取 max(range_start, query_abs_lo - SLIDING_WINDOW + 1) 的最小值、last_allowed_key 取 max(base 窗口末, range_end),最后 clamp 到 seq_len-1。同时新增 USE_R_SWA constexpr,把 restricted sliding window 从 USE_MM_PREFIX 的全扫描条件中拆出(`can_prune_sliding = (not USE_R_SWA) and ((not USE_MM_PREFIX) or MM_PREFIX_CLAMP_SW)`);非 causal 分支的 last_allowed_key 不再提前 clamp,统一移到 MM 循环之后。

## 关键改动

```text
vllm/v1/attention/ops/triton_attention_helpers.py: compute_tile_loop_bounds 新增 MM_PREFIX_CLAMP_SW/USE_R_SWA/MAX_MM_RANGES 参数与 per-range tile bound 计算
vllm/v1/attention/ops/triton_unified_attention.py: kernel 向 helper 传入 mm_prefix_range_ptr/seq_idx 等新参数
tests/kernels/attention/test_triton_unified_attention.py: tile-bounds 回归测试 + dense reference 对拍
```

## 性能声明

E2E:unsloth/gemma-4-26B-A4B-it-NVFP4,NVIDIA RTX PRO 6000 Blackwell Server Edition,60 images/request、concurrency 12、mean prompt ~16550 tokens,Triton language 与 multimodal-encoder attention,FP8 target KV + FlashInfer CUTLASS NVFP4 MoE + DFlash,1 次 warmup 后 3 次测量取均值:Nightly vs 本 PR = Mean E2E 31.24s -> 9.67s(3.23x faster),Mean TTFT 21.99s -> 5.67s,Server prefill 19.90s -> 4.80s(4.15x faster),decode throughput 58.58 -> 97.24 tok/s,system output throughput 124.93 -> 410.27 tok/s。作者概括为 up to 3-4x E2E 提升。

## 验证与回归风险

新增 GPU 回归测试:4096 token 序列、1024 滑窗、宽于窗口的 multimodal range,断言选中保守 [3,72) tile 区间而非 [0,128);test_clamped_mm_prefix_preserves_noncausal_right_window 验证 non-causal 右窗口与 image range 并集([3,68));ref_paged_clamped_mm_attn dense reference 对拍(atol/rtol 2e-2,并断言与纯 chunk reference 不相等)。revert 实现保留测试会编译失败,证明覆盖新路径。回归面:所有走 compute_tile_loop_bounds 的 Triton attention(带 mm_prefix_range 的 Gemma4 系模型、restricted sliding window 路径 USE_R_SWA 的行为被显式拆分),bound 推导错误会漏 KV 或多扫,靠 tile-bounds 断言与 dense 对拍兜底。

## 关联

同区域:Triton unified attention 的 SWA/3D scoping 剪枝;Gemma4 相关内核优化 PR #48666(FA4 FP8 Kernel,本索引库同批次)。正文未给出其他关联 PR 编号。
