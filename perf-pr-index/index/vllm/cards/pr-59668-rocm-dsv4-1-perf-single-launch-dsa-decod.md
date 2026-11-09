# PR #59668: [ROCm][DSV4.1][Perf] Single-launch DSA decode candidate mask

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @cpersson-amd |
| 合并 | 2026-10-27 |
| 分类 | Speculative Decoding (tier: major) |
| Label | rocm, ready, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾无 "file(s) dropped from diff" 说明,diff 完整保留 2 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/59668 |

## 一句话结论

ROCm 把 DSA decode candidate mask 从 4 个 kernel 合成 1 次 launch,MI355X kernel 快 1.06×~2.50×,e2e 吞吐 +0.44%。

## 动机

ROCm 上 DeepSeek V4.1 的每个 consumer indexer layer(decode step 共 4 个)此前用 4 个 kernel 给 decode logits 打 candidate mask:`direct_copy`(`visible.to(torch.int64)`,4.4 µs)、`FillFunctor<long>`(`torch.zeros_like(visible)`,5.2 µs)、`_candidate_flags_kernel`(8.7 µs,清整行 `nblocks+1` flags 再 scatter candidates)、`_mask_candidates_strided_kernel`(4.1 µs,按 `[0, end)` 掩码)。前两个 elementwise kernel 只为构造全零 starts tensor 与 int64 的 `seq_lens`,Triton kernel 并不需要;flags kernel 还会清整行、不管 live context 多短。

## 优化点

把 flags kernel 折进 #56628 引入的 `_mask_candidates_strided_kernel`,并去掉前面的 int64 cast 与 `zeros_like`。新 kernel 自己建 flags:每行 tile 按 round-robin 分给各 program,每个 program 只清自己 live tiles 的 flags、scatter 落在其中的 candidates 并掩码,不触碰 live context 之外的部分。AMD 上 `tl.debug_barrier()` 只等 LDS 流量,故每个 barrier 前加 store wait、flags 以 `.cv` 读取;wait 指令按架构区分(gfx9 `s_waitcnt vmcnt(0)`、gfx11 `s_waitcnt_vscnt`、gfx12 `s_wait_storecnt`,因 gfx10+ 单独计 store)。tile 在 64 行以下为 2048 列、64 行起 4096 列,仅依赖行数以便在 FULL CUDA graphs 下 grid 固定;store 到 `end` 即停,不再补到下一个 1024 列边界。decode 调用点把 int64 cast、`zeros_like`、`repeat_interleave` 换成 `None` starts、int32 `seq_lens` 与 `row_repeat`(与 CUDA 路径一致),mask 与 select 两条路径都改。结果每个 consumer indexer layer 由 4 个 kernel 变 1 个。

## 关键改动

```text
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py: 核心。_mask_candidates_strided_kernel 自建 flags 并融合 scatter/掩码(round-robin tiles、store wait + .cv 读、按行数选 tile);调用点用 None starts / int32 seq_lens / row_repeat 替换 int64 cast 与 zeros_like,mask 与 select 路径同改
tests/kernels/attention/test_rocm_aiter_candidate_mask.py: 新增/扩展用例,校验融合 kernel 与旧链、共享 apply_candidate_mask 在 [0, end) 逐位一致,覆盖 last-column clamp 与 "end 及之后不写"
```

## 性能声明

正文 Claims:decode candidate mask 在 MI355X 上 1.06× 到 2.50× faster(不计去掉的两个 elementwise kernel,约每次调用再省 9.6 µs);e2e mean TPOT −0.48%、throughput +0.44%(MI355X,12 个 sweep 点)。Kernel 条件:MI355X、CUDA graph(50 calls per replay,5 次取优)、width 262144、2048 个 8 列 candidate blocks、ends 设为 live length;"old" = `_candidate_flags_kernel` + `_mask_candidates_strided_kernel`,每点输出与旧链及共享 `apply_candidate_mask` 在 `[0, end)` 完全一致。old → new µs(speedup):rows/live cols 4/2048 5.71 → 3.11(1.84×)、16/2048 6.03 → 3.20(1.88×)、64/2048 8.04 → 3.54(2.27×)、256/2048 17.11 → 6.84(2.50×)、4/15000 6.22 → 3.51(1.77×)、256/32000 25.94 → 16.98(1.53×)、256/131072 53.21 → 50.28(1.06×)等 12 点。End-to-end 条件:vLLM on MI355X、TP4 + EP、FP8、fused 对 default,ISL/OSL 8192/1024、60000/600、128000/1024,并发 4-256;均值 −0.48% TPOT / +0.44% throughput,单点变动均不超过 1%,12 点中 11 点吞吐改善、10 点 TPOT 改善,中位 TPOT −0.53%。Accuracy GSM8K 5-shot(`lm_eval`):flexible-extract 0.9227±0.0074 → 0.9363±0.0067、strict-match 0.9219±0.0074 → 0.9371±0.0067,正文称差异不显著(1.4σ / 1.5σ,合并标准误约 0.010)。

## 验证与回归风险

`pytest tests/kernels/attention/test_rocm_aiter_candidate_mask.py tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py`:在 MI355X 均通过;新增 `test_matches_shared_kernel_model_shape`(2048 candidates、crowded/spread、8/64 行覆盖两种 tile)、`test_matches_shared_kernel_candidates_past_width`(末列 clamp),`test_leaves_columns_past_end_untouched` 改断言 end 及之后不写;mxfp4 indexer 测试覆盖 `_apply_candidate_mask_strided` 另一调用方,CI 在 MI300 attention-kernels job 运行。调用点旧/新序列对 mask 与 select 路径在 `next_n` 1/2、1-D/2-D `seq_lens` 下 logits 与 candidates 逐位一致,profiler 显示旧 3 kernel / 新 1 kernel。风险面(正文 Limitations):tile size 仅在 gfx950 调优,gfx942 只在 CI 跑未 benchmark;gfx11/gfx12 的 store wait 只在编译 ISA 上核对(无 RDNA 硬件);与 #57859 存在代码/合并顺序交互。

## 关联

PR 正文明确:#56628(已合并)引入本 PR 所基于的 strided decode mask 且保留独立 flags kernel;#57859(open)为 gfx950 decode 加 candidate-only paged MQA scorer、编辑同一代码,其 fast path 跳过 mask 故本 PR 融合 kernel 不运行,其 dense fallback 仍含本 PR 所改部分;正文给出两种合入顺序下的 rebase / 冲突说明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
