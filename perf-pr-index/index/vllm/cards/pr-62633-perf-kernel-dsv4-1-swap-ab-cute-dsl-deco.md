# PR #62633: [Perf][Kernel][DSv4.1] Swap-AB CuTe DSL decode attention for small TP2/TP4 steps

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @roborluo |
| 合并 | 2026-11-03 |
| 分类 | Attention / Kernel \| tier: major |
| Label | performance, ready, ci/build, deepseek, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 1 file(s) dropped from diff for size (see PR for full diff). |
| 链接 | https://github.com/vllm-project/vllm/pull/62633 |

## 一句话结论

新增 swap-AB decode kernel，DSv4.1 TP2/TP4 小步 decode attention 提速 1.4-2.15x。

## 动机

Mega attention 是 SM10x 上 DeepSeek-V4.1 的默认 decode 路径，其 FP8 decode kernel 接收 64 或 128 个 query heads，TP2/TP4 下有一半或四分之三的 MMA 计算耗费在 padding heads 上；`flash_mla_mega_attn.py` 里 mega vs split-KV 的说明已记录这一 padded-head 成本主导小步 decode。低并发 decode 与 spec-decode 的 verify / draft 步骤正是这类小步场景。

## 优化点

新增 swap-AB CuTe DSL decode attention kernel：把 KV slots 放到 tcgen05 MMA 的 M 侧（128 行）、16 或 32 个 live heads 放到 N 侧，不再把 heads pad 到 64。FlashMLA decode 把 query heads 映射到至少为 64 的 MMA M 维，无法靠更小 tile 解决，故改用独立 kernel；它在 JIT warmup 从 Python 编译、无需改构建，改编自 SGLang 的 swap-AB DSv4 decode attention。每 (token, KV tile) 一个 CTA、每 token 一个 cluster，一次 launch、PDL、无 workspace 与 host sync。dispatch 仅在 SM10x、compressed cache、16/32 live heads、CuTe DSL 构建于 CUDA 13.2+、`VLLM_BATCH_INVARIANT` off、步骤 ≤16 tokens 等条件全满足时切换，否则仍走 mega kernel，且不新增任何 flag、环境变量或配置项。

## 关键改动

```text
vllm/models/deepseek_v41/nvidia/ops/swapab_decode_attn.py: 新增 SwapABDecodeKernel，KV slots 置于 MMA M 侧、live heads 置于 N 侧
vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py: 在 _forward_decode_mega 中加入 dispatch 规则与 JIT warmup key 注册
benchmarks/kernels/benchmark_dsv41_mega_attn.py: 增加 swap-AB 对比 arm，以 CUDA graph replay 计时
tests/models/deepseek_v41/nvidia/test_swapab_decode.py: 新增精度、padding、CUDA graph 确定性与 dispatch 单测
.buildkite/test_areas/models_basic.yaml: 把新测试放入 optional 4xB200 nightly step
```

## 性能声明

正文声明 "1.4-2.15x faster decode attention for 1-16 tokens"：16 live heads（TP4）1.8-2.15x、32 live heads（TP2）1.4-1.8x，覆盖 NVFP4（288 B）与 fp8（528 B）compressed records，低端值来自 B300。逐档表（speedup = mega / swap-AB step time，两次运行中位数、每格差异 ≤9%，CUDA graph replay、warm L2）中 16 heads/TP4/NVFP4 在 1/2/4/8/16 tokens 为 2.15x/2.00x/2.06x/1.94x/2.04x，32 heads/TP2/fp8 为 1.76x/1.66x/1.71x/1.62x/1.71x。端到端 TP2 并发 1-16 下 median TPOT 降低 4.5-7.6%。精度：64 个测试案例中相对 fp64 参考的误差 ≤ mega kernel 的 1.1x，输出与 mega kernel 相差在 2% 内。硬件为 SM10x（含 B300），记录格式为 NVFP4 / fp8 compressed records。

## 验证与回归风险

新增 test_swapab_decode.py：test_matches_reference 64 passed、test_untouched_padding 2 passed、test_cuda_graph_and_determinism 12 passed、test_dispatch_rule 9 passed（CPU）、test_warmup_key_matches_launch 4 passed（CPU），复用 test_v41_cache_writers_match_flashmla_decode 3 passed；测试需 SM10x 与 CUDA 13.2+ 的 CuTe DSL，否则 skip，CI 置于 optional nightly "(4xB200) DeepSeek-V4.1 NVIDIA Kernels"。限制：仅 `DeepseekV4MegaAttnAttention` 改变，FlashMLA sparse、FlashInfer sparse MLA、SM90/SM120/ROCm 与 prefill 不变，TP1/TP8 保留 mega kernel；输出非 bit-identical，batch-invariant 全程用 mega kernel；成本随 token 数增长故设 16-token 上限；工具链需 CUDA 13.2+，V4.1-Flash warmup 增加两次 JIT 编译。

## 关联

正文提及 #56935（mega-attention layer）、#57448（所属工作）、#61371（interleaved V4.1 KV records），并声明不重复 #59418、#61615、#57711、#61868。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
