# PR #41951: [perf] [multi-lora] remove empty CTAs from fused LoRA expand and add split-K to shrink

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kailash109 |
| 合并 | 2026-10-13 |
| 分类 | Other(tier: major) |
| Label | documentation, lora, run-ci, jit-kernel |
| Diff | diffstat 缺失(素材未统计行数);diff 覆盖 expand/shrink kernel、tune 脚本、chunked backend、environ、测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/41951 |

## 一句话结论

多 LoRA CSGMV expand 剔除空 CTA、shrink 增加可选 split-K,贡献者 serving 结果吞吐提升约 9.5%-11.6%。

## 动机

CSGMV(chunked SGMV)fused LoRA expand 对较小的 packed QKV slice 也按 max slice size 启动矩形网格,产生大量空 tile 直接 early-return;rank-32 shrink 则没有用尽并行度(K 维只有一个 block 序列)。PR 将 expand 网格用 CPU 侧 slice offset 压紧,并为 shrink 增加可选 split-K(FP32 部分和 + 单独 reduction),释放 K 维并行。

## 优化点

expand 侧:`chunked_sgmv_lora_expand_forward` 新增可选 `slice_offsets_cpu`,当拿到 CPU offset(模型元数据本就在 CPU,不额外拷贝 device 张量)时,把网格从 `(cdiv(max_slice_size,BLOCK_N), num_slices, segments)` 压成按真实 slice 宽度求和的扁平 `(total_tiles, segments)` 二维网格;kernel 内以 `SLICE_OFFSETS` constexpr 在 `static_range` 中定位 flat_tile 属于哪个 slice,消除空 CTA。无 CPU offset 时保留原矩形路径与 early-return。shrink 侧:新增 `_chunked_lora_shrink_split_k_kernel` 写 FP32 部分和 `[SPLIT_K, TOTAL_ROWS, N]`,再由 `_reduce_shrink_split_k` 只归约活跃行/列;`SGLANG_ENABLE_LORA_CSGMV_SPLIT_K=1` 启用已有 BF16 rank-32 自动策略(K=12288 -> split_k=16,K=4096 -> 8),默认关闭;确定性推理保持原累加顺序(split-K 改变舍入);partial buffer 行步长作 runtime 参数,使不同 token 数复用已编译 kernel;`split_k=None` 自动、显式正值保留、非法值先于 deterministic fallback 报错。

## 关键改动

```text
python/sglang/kernels/ops/lora/dense/chunked_sgmv_expand.py: SLICE_OFFSETS constexpr 扁平化 ragged expand 网格,消除空 CTA
python/sglang/kernels/ops/lora/dense/chunked_sgmv_shrink.py: 新增 split-K kernel + reduce kernel 与自动派发策略
python/sglang/srt/lora/backend/chunked_backend.py: 三个入口透传 output_offset_cpu
python/sglang/srt/environ.py: 新增 SGLANG_ENABLE_LORA_CSGMV_SPLIT_K(默认 False)
benchmark/kernels/lora_csgmv/bench_csgmv_split_k.py: early_return / compact / compact_split_k 三态对比
```

```python
if offsets:  # CPU slice offsets 可用时压缩网格
    grid = (sum(triton.cdiv(hi - lo, BLOCK_N)
                for lo, hi in zip(offsets, offsets[1:])),
            segment_grid)
```

## 性能声明

贡献者的 corrected serving 结果(PR 内评论链接)报告约 9.5%-11.6% 吞吐提升;正文明示这些测量早于 cleanup commits,该 head 未声称新的 serving/性能测量。测试机 H200(commit `a7deef7008...`)上的验证为正确性回归(见下节),不含端到端性能表。基准脚本以 `do_bench_cudagraph` 覆盖 shrink+expand(rank 32,BF16,adapters=8,tokens 32/128/8192,qkv/gate_up/down 三投影),但素材未包含其数值输出。

## 验证与回归风险

`test/registered/kernels/ops/lora/test_chunked_sgmv_cuda_graph.py -f` 8 个测试通过,覆盖 FP32 partials、ragged/空 slice、output stride、mixed ranks、CUDA-graph metadata replay、显式 split 因子与 deterministic fallback;新增断言覆盖:split 边界与 rank 尾部须匹配 FP32 oracle、compact 与矩形路径逐位相等、graph replay 跟随 adapter 激活状态、确定性模式抑制自动与显式 split-K。风险:split-K 改变舍入顺序,确定性推理下强制回退;自动策略仅对 H200 上测得的 BF16 rank-32、K in (4096,12288)、BLOCK(16,16,256) 生效,其他形状保持原路径;expand 网格压缩依赖调用方传入正确的 `output_offset_cpu`。CI:Base 通过,Extra 与 AMD ROCm 10 失败(素材所示)。

## 关联

同区域:#28371(该测试脚本原本回归的 dynamic chunked-SGMV LoRA segments)。其他同分类 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
