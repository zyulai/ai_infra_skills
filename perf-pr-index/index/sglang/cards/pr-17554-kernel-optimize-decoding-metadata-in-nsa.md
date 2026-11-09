# PR #17554: Kernel: optimize decoding metadata in NSA multi-spec backend with fused kernels

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Johnsonms |
| 合并 | 2026-02-14 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel, blackwell, npu, run-ci, deterministic, piecewise-cuda-graph, diffusion, model-gateway, mthreads |
| Diff | diffstat 缺失(素材未提供 +N/-N);pack diff 片段含 2 个文件,另有 6 个文件因体积未收录 |
| 链接 | https://github.com/sgl-project/sglang/pull/17554 |

## 一句话结论

NSA 多步投机解码的 metadata 拷贝融合为单次 kernel launch,端到端吞吐 +18.5%。

## 动机

NSA backend 在 CUDA graph replay 的投机解码路径中,需要把 cache_seqlens、cu_seqlens_k、page_table、nsa_cache_seqlens 及可选的 real_page_table / FlashMLA metadata 等 24 次小拷贝逐一发起。每次只搬少量 int32 数据,瓶颈在 host 侧 launch 而非带宽或算力。作者先做了 Triton 原型:Triton 减少了 device kernel 数,但端到端延迟没有实质改善(host-issue-bound),因此最终改用手写 CUDA kernel + 薄 C++/Python 绑定,把 metadata 拷贝打包进一次 fused launch。

## 优化点

新增 `fused_metadata_copy_cuda`(单后端,统一支持 DECODE / TARGET_VERIFY / DRAFT_EXTEND 三种 forward mode):源/目标指针打包进 SourcePointers/DestinationPointers 结构体,经 `__grid_constant__` 常量内存传入,`HAS_REAL_PAGE_TABLE`/`HAS_FLASHMLA` 以模板参数在编译期裁剪;新增 `fused_metadata_copy_multi_cuda`,一次 launch 同时拷贝到 3 个后端。运行时(`nsa_backend.py`)按 `speculative_num_steps >= 3` 对前 3 个后端用 fused kernel、其余逐个拷贝;`< 3` 时开销不划算,维持逐个拷贝。实现走 `python/sglang/jit_kernel` 的 JIT 编译体系(TVM FFI,按 forward_mode/可选张量组合缓存编译产物)。

## 关键改动

```text
python/sglang/jit_kernel/csrc/elementwise/fused_metadata_copy.cuh: 融合 metadata 拷贝 CUDA kernel(JIT)
python/sglang/jit_kernel/fused_metadata_copy.py: JIT 编译封装与 Python API
python/sglang/srt/layers/attention/nsa_backend.py: 按 speculative_num_steps 分派 fused kernel
```

(PR 正文另列 sgl-kernel/csrc/elementwise/fused_metadata_copy.cu、sgl-kernel/python/sgl_kernel/elementwise.py、sgl-kernel/tests/test_fused_metadata_copy.py;pack 注明 6 个文件的 diff 未收录)

## 性能声明

metadata 拷贝路径:CPU 侧处理 193µs → 12µs(16× faster),GPU kernel 执行 177µs → 4.7µs(37.6× faster),measured path 的 kernel launch 从 24 次减到 1 次;端到端吞吐 151 → 179 tokens/sec(+18.5%)。精度:gsm8k、gpqa-diamond、aime 2025(DeepSeek-V3.2-Exp,TP8/DP8,EAGLE,speculative-num-steps 3);40 个测试全部通过。硬件型号 PR 未注明。

## 验证与回归风险

`sgl-kernel/tests/test_fused_metadata_copy.py` 覆盖单后端各 forward mode、可选张量存在/缺失、多后端 3 目标同时拷贝与性能基准。运行时门控 `speculative_num_steps >= 3`,低步数保持原路径;JIT 编译失败会 raise(首次运行有编译开销);可选张量(real_page_table、FlashMLA metadata)以 nullptr 区分,kernel 内按 compile-time 分支处理。

## 关联

前序 PR #14781(同为 NSA 多步投机解码 metadata 路径,做 precompute-once-copy-many 优化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
