# PR #10543: [sgl-kernel] Optimize concat_mla_k kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-09-28 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/10543 |

## 一句话结论

重写 sgl-kernel 的 concat_mla_k CUDA 内核(流式流水线 + no-allocate 访问),DeepSeek-R1 TP8 prefill E2E input throughput 提升 2%。

## 动机

受 #10473 启发,现有 `concat_mla_k`(MLA 把 k_nope 与 k_rope 拼进连续 K 的 elementwise 内核)采用"先逐 head 缓存到寄存器数组、再写回"的两段式实现,访存未被优化;作者为其补基准并重写以提速。

## 优化点

内核改为 per-warp 软件流水:每个线程持有当前 int2(nope,8B/线程)与 int(rope,4B/线程)向量,循环内先对下一个 head 的源地址发 `prefetch.global.L2`,再加载下一拍数据、写回当前拍,把 load/store 沿 head 维展开成_prefetch-载-存重叠的流。读写统一走 `ld.global.nc.L1::no_allocate` / `st.global.L1::no_allocate`( adapted from DeepEP 的 `utils.cuh`,新置于 `sgl-kernel/csrc/elementwise/utils.cuh`),避免流式数据污染 L1;指针全部加 `__restrict__`;`static_assert` 保证 nope/rope 向量宽度与 head_dim 精确匹配(256B/行 nope、128B/行 rope)。同时删除了旧的 `get_lane_id`/`ceil_div` 本地定义(移入 utils.cuh)。新增 `benchmark/kernels/elementwise/benchmark_concat_mla.py`,对比 torch / torch.compile / triton / hack_non_strided / cuda 五种实现并做输出一致断言。

## 关键改动

```text
sgl-kernel/csrc/elementwise/concat_mla.cu: concat_mla_k_kernel 重写为流水线式向量化访存
sgl-kernel/csrc/elementwise/utils.cuh: 新增,DeepEP 风格的 no-allocate load/store 与 prefetch PTX helper
benchmark/kernels/elementwise/benchmark_concat_mla.py: 新增内核微基准(含正确性断言)
```

## 性能声明

微基准(`triton.testing.do_bench`,num_tokens 2048–32768,bf16,128 heads,nope 128 + rope 64):cuda 内核 2048 tokens 0.030880 ms、32768 tokens 0.425984 ms,对比 torch 0.114656 / 1.610928 ms(约 3.7 倍),也快于 triton 0.266016 / 1.651552 ms 与 torch_compiled。E2E:DeepSeek-R1,TP8,fp8,flashinfer,chunked-prefill 16384,bench_serving random(input 4000/output 1,max-concurrency 16,100 prompts):input token throughput 9165.76→9327.91 tok/s(约 +2%),duration 43.64→42.88 s,Mean TTFT 6566.73→6445.19 ms。PR 未注明 E2E 测试的 GPU 型号。

## 验证与回归风险

benchmark 脚本内置 torch 参考输出与 CUDA 输出的 `torch.all` 相等断言(32768 tokens 上验证)。PR 的 Accuracy Tests 与 Benchmarking 章节为空模板,未提供模型级精度结果。风险面:内核硬编码 `NUM_LOCAL_HEADS=128`、`QK_NOPE_HEAD_DIM=128`、`QK_ROPE_HEAD_DIM=64` 与向量宽 static_assert,只适用于该 MLA shape;no-allocate/nc 访问假定流式一次性数据,若复用到其他 elementwise 内核需注意缓存语义。

## 关联

#10473(动机来源)。`utils.cuh` 改编自 DeepEP 的 `csrc/kernels/utils.cuh`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
