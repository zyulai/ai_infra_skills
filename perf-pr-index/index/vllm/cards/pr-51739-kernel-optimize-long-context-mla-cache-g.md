# PR #51739: [Kernel] Optimize long-context MLA cache gathers

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-08-11 |
| 分类 | Attention / Kernel（tier: major） |
| Label | performance, ready, ci/build |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失）；pack 中 diff 截断，省略 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/51739 |

## 一句话结论

重构 MLA cache gather 为按逻辑 page 调度 + 合并向量化访存，长上下文 kernel 延迟最高降 331 倍、有效带宽达 6.6 TB/s。

## 动机

原 `cp_gather_cache` 以 1024 线程 CTA 的小 batch/split grid 启动、对长请求串行遍历；转换类 kernel 则对每个 token 重复做 request/block 查找，内存事务填充不满（underfilled）。在 chunked prefill 中当各请求 cached-context 长度差异很大时，这些开销占主导，长上下文场景下 gather 成为明显瓶颈。

## 优化点

- 调度单位从"每个输出 token 一个任务"改为"每个逻辑 cache page 一个任务"：新增 `GatherPageTask`（req_id、logical_block、page_token_begin/end、output_token_begin），`map_gather_page_task` 对每个逻辑 page 只映射一次 block table；partial 首尾 page 携带显式 valid token range，走同一条路径，不再需要 per-token fallback，且支持非零 `seq_starts`。
- Page 内拷贝/转换展平为 coalesced vector load/store（int4）：按指针对齐与 stride 对齐情况在 vectorized/scalar、contiguous/non-contiguous 四个模板实例间静态分发。
- FP8→BF16 upconvert 同样页级调度：原来每 token 一次 warp + 二分查找 request，改为 CTA 内 warp 迭代 page 内 token，scale 常量化进寄存器/共享内存；转换 kernel 的 FP8-to-BF16 store 也做了合并。
- CTA 几何针对长且不均匀的 prefill 调优：grid 大小按 `ceil(total_tokens/block_size) + 2*num_reqs` 估计，线程数按总 token 数取 128/256；`gather_and_maybe_dequant_cache` 不再依赖 `token_to_seq` 反查表。

## 关键改动

```text
csrc/libtorch_stable/cache_kernels.cu: 三个 gather kernel 重写为 *_page 页级调度版本
benchmarks/kernels/benchmark_cp_gather.py: 新增可复用 kernel benchmark（60K-300K 场景）
tests/kernels/attention/test_cache.py + tests/kernels/test_cp_gather_fp8.py: 长上下文正确性用例
```

页任务映射的核心结构：

```cpp
struct GatherPageTask {
  int32_t req_id;
  int32_t logical_block;
  int32_t page_token_begin;
  int32_t page_token_end;
  int32_t output_token_begin;
};
```

## 性能声明

（单卡 NVIDIA GB300，block size 64、entry size 576，`benchmarks/kernels/benchmark_cp_gather.py`，median cold-cache latency。）

- `cp_gather_cache`：60K 4231.14→19.17 us（220.7x，PR 3.61 TB/s）；300K 19919.41→60.13 us（331.3x，5.75 TB/s）；60K-300K batch 8 19907.10→201.73 us（98.7x，6.62 TB/s）。
- FP8 upconvert：60K 0.99x、300K 1.04x（88.61 us，6.12 TB/s）、batch 8 344.77→313.66 us（1.10x，6.69 TB/s）。
- maybe-dequant：60K 1.39x、300K 160.10→95.26 us（1.68x）、batch 8 593.57→335.65 us（1.77x，5.97 TB/s）。
- Nsight Compute（batch-8 shape）：raw copy 达峰值 DRAM 吞吐的 83.54%，FP8 upconvert 83.92%，maybe-dequant 74.67%。

## 验证与回归风险

- `tests/kernels/attention/test_cache.py`（-k gather_and_maybe_dequant_cache_mla or cp_gather_cache_mla）17 passed；`tests/kernels/test_cp_gather_fp8.py` 17 passed；pre-commit 通过。
- 新增长上下文用例覆盖 auto/FP8、seq_starts、shuffled blocks、partial pages、 uneven request lengths、strided raw views；新增 benchmark 供后续复测。
- 回归风险：改动为字节拷贝/转换调度，不改模型数学，PR 未跑模型级评估（作者自述理由）；raw copy 保留 vectorized/scalar 两条路径处理非对齐 stride，对齐探测逻辑（`alignof(int4)`、stride 检查）决定是否走向量路径；kernel 面向 MLA（entry size 576/656 等形状）与 chunked prefill 路径。

## 关联

- 相关 PR（PR 正文 duplicate check 提及，均不改 gather 调度）：#45537（block-table 越界正确性修复）、#51252（sparse-indexer prefill buffer sizing）。
- 同分类：本批 Attention/Kernel PR #57057（UltraQuant 4-bit KV backend）、PR #39616（ROCm AITER MLA + Eagle3）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
