# PR #57420: [Kernel][MiniMax-M3] Add Triton query-tiled sparse prefill tuned on Hopper

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @NVShreyas |
| 合并 | 2026-10-27 |
| 分类 | Attention / Kernel |
| Label | performance, ready, ci/build, minimax |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),diff 覆盖 4 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/57420 |

## 一句话结论

为 MiniMax-M3 稀疏 prefill 新增 Hopper query-tiled Triton kernel:相邻 query 合并取 KV block 并集、每块只加载一次,高重叠场景 microbenchmark 提速最高 5.4x,E2E prefill 吞吐 +4.5-5.1%。

## 动机

MiniMax-M3 稀疏 prefill 原先每个 query token 启动一个 Triton program,而相邻 query 的 top-k 稀疏 KV block 常常大量重叠,同一 K/V block 被反复加载。decode kernel 还缺少对持久化 top-k block ID 的校验,直接解引用 block table 在混合 chunked-prefill/prefix-reuse batch 下不安全。作者声明与 #52963(ROCm/gfx950)、#55372(query-tiled verify-decode)不重复,本 PR 专注 Hopper 上的 prefill。

## 优化点

新增 `_gqa_sparse_fwd_tiled_kernel`:一个 program 处理 `BLOCK_SIZE_Q=16` 个相邻 query(按 `grid=(cdiv(max_query_len, tile_q), num_kv_heads, batch)` 组织),先把 tile 内所有 query 的 top-k block ID 读成 `(BLOCK_Q, BLOCK_T)` 矩阵,无效/越界 ID 填 0x7FFFFFFF 哨兵;循环内用 `blk = min(remaining)` 逐个枚举并集 block,每个 distinct block 只从 paged KV cache 加载一次 K/V,再用 `member[:, None] & causality & pos_mask` 的三维 mask 控制哪些 query 行真正参与该块的 `tl.dot` softmax 在线更新(`exp2` 形式,含 -inf 行的 safe_m 处理避免 NaN)。FP8 KV 支持标量与 per-token-head 两种 scale 模式(KV_SCALE_MODE 1/2)。非对齐 stride 用 `do_not_specialize_on_alignment` 避免 Triton 重编译;BLOCK_SIZE_K=128 即一个 sparse block 对应一个 page。Hopper(SM90)在 `sparse_attention.py` 中自动选择该 prefill kernel,其他 CUDA 架构沿用 generic scalar kernel;decode kernel 增加 top-k block ID 有效性校验。tile 固定 16(在 H200 上调优)。

## 关键改动

```text
vllm/models/minimax_m3/nvidia/ops/sparse_prefill.py: 新增 query-tiled 稀疏 prefill Triton kernel(Hopper 自动选择)
vllm/models/minimax_m3/common/sparse_attention.py: SM90 平台分派到 nvidia.ops 的 tiled prefill
tests/kernels/attention/test_minimax_m3.py: 参数化 scalar/query_tiled 两种 kernel 做 CUDA parity
.buildkite/test_areas/kernels.yaml: 新增 H100 MiniMax-M3 sparse prefill CI 分片
```

## 性能声明

H200 kernel microbenchmark(6 种 prefill 几何,tile 1-32,BF16/FP8 KV,16/8/0 shared of top-k 16),相对原 single-tile kernel 中位加速:BF16/16of16 4.49x(tile 8)、FP8/16of16 5.37x(tile 16)、BF16/8of16 1.16x、FP8/8of16 1.20x;代表点 batch 32 q256 BF16:1.0259 ms → 0.2252 ms(tile 8,4.56x)。E2E SPEED-Bench(TP8 H200,`speed_bench_throughput_8k_low_entropy`,512 conversations,OSL 256,中位 ISL 8223):并发 8 输入吞吐 14458→15108 tok/s(+4.50%),并发 16 17365→18248(+5.09%),request latency -4.3~-4.9%、ITL -5.8~-6.4%。E2E 固定 ISL 8102/OSL 1024(MiniMax-M3-MXFP8, TP8, 8xH200):tile16 在并发 64-256 提升 output 吞吐 6.6-6.8%、TTFT 6.6-8.2%、ITL 5.9-6.2%(并发 1 时仅 +0.9%)。精度:H200 benchmark 矩阵最大绝对误差 0.0009765625。

## 验证与回归风险

- 测试:`tests/kernels/attention/test_minimax_m3.py` 的 prefill correctness 参数化到 scalar 与 query_tiled 两个 kernel(覆盖 partial query tiles、混合请求长度与 prefix、block membership、非恒等 paging、NHD/HND layout);`test_minimax_m3_sparse_attn_fp8_scale.py` 同样参数化(FP8 KV scale 模式);新增 Buildkite H100 分片跑这两个测试;pre-commit 通过。
- 风险:仅 Hopper(SM90)自动启用,opt-in 性质;union 遍历的 `tl.min` 循环在 top-k 完全不重叠时退化(0 shared 时仍有 1.16-1.20x 增益,但依赖 BLOCK_T padding);decode 路径的 block ID 校验改变混合 batch 行为,需 chunked-prefill 场景回归。

## 关联

- 同区域:#52963(ROCM/gfx950 稀疏 attention)、#55372(query-tiled verify-decode attention)、MiniMax-M3 系列模型支持 PR;与 #20034/#34732 等 MLA prefill 优化同属 attention prefill 提速线。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
