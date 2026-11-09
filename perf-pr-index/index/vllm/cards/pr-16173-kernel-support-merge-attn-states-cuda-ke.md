# PR #16173: [Kernel] support merge_attn_states CUDA kernel, 3x speedup

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @DefTruth |
| 合并 | 2025-04-11 |
| 分类 | Attention / Kernel |
| Label | ready, ci/build, v1 |
| Diff | diffstat 缺失(pack 标注 +None/-None);可见 diff 涉及 8 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/16173 |

## 一句话结论

用 CUDA kernel 替换 Triton 版 merge_attn_states 合并 split-KV 部分注意力结果,微基准最高 4.43x,消除 CPU 启动开销。

## 动机

cascade attention 与 split-KV 场景需要把 prefix/suffix 两份部分注意力输出按 log-sum-exp 合并(实现 arXiv 2501.01005 §2.2),V1 的 FlashAttention 与 MLA 公共后端此前调用 `triton_merge_attn_states.py`。Triton kernel 在小 shape 下 CPU 启动开销占比高,成为该路径的主要开销;作者以手写 CUDA kernel 降 CPU overhead。

## 优化点

`csrc/attention/merge_attn_states.cu` 新增 `merge_attn_states_kernel`:每线程处理一个 128-bit pack(float32 pack 4 元素、half/bf16 pack 8 元素),以 `uint4` 做 128b load/store;LSE 合并按 `max_lse` 归一后用 float FMA 完成 `p_out*p_scale + s_out*s_scale`,FMA 全程保持 float 精度;`output_lse` 由每个 head 的 pack_idx==0 线程写一次;inf 值先归一为 -inf 再参与合并。模板按 scalar_t 分发(float/uint16_t/__nv_bfloat16),不支持 FP8 dtype、head_size 不是 pack_size 整数倍或 ROCm(`#ifndef USE_ROCM`)时回退原 Triton kernel。Python 侧新增 `vllm/attention/ops/merge_attn_states.py` 统一分发入口,V1 flash_attn 与 MLA common 后端切换到该入口。

## 关键改动

```text
csrc/attention/merge_attn_states.cu: 新增 CUDA kernel + DISPATCH_BY_SCALAR_DTYPE 分发
vllm/attention/ops/merge_attn_states.py: 新增统一入口,dtype/headdim 检查失败回退 Triton
vllm/v1/attention/backends/flash_attn.py: 切换 import 到新入口
vllm/v1/attention/backends/mla/common.py: 切换 import 到新入口(不再依赖 HAS_TRITON)
vllm/_custom_ops.py: 暴露 torch.ops._C.merge_attn_states 包装
tests/kernels/test_merge_attn_states.py: 新增正确性+性能对比测试
```

```cpp
const float max_lse = fmaxf(p_lse, s_lse);
p_lse = p_lse - max_lse;
s_lse = s_lse - max_lse;
const float p_se = expf(p_lse);
const float s_se = expf(s_lse);
const float out_se = p_se + s_se;
const float p_scale = p_se / out_se;
const float s_scale = s_se / out_se;
```

## 性能声明

微基准(MLA TP=8 场景,每 rank 16 query heads,headsize 128,NVIDIA L20,CUDA vs Triton):float32 256 tokens 0.05647→0.01638 ms(3.45x),1536 tokens 3.58x;float16 4096 tokens 0.08372→0.01955 ms(4.28x);bfloat16 4096 tokens 0.08524→0.01925 ms(4.43x);512~1024 tokens 各 dtype 约 3.3~3.5x。TLDR:相对 Triton kernel 最高 speedup 超过 3x。

E2E:DeepSeek-R1 671B,L20x3,PP=3 + TP=8,`--no-enable-prefix-caching --enable-chunked-prefill`:4K IN:1K OUT Mean TTFT 5687.80→5654.02 ms;8K IN:64 OUT Mean TTFT 8861.07→8767.16 ms;TPOT/ITL 基本持平,作者结论"inference 性能不劣化"。CEval(evalscope,DeepSeek-R1)AverageAccuracy 0.90198。

## 验证与回归风险

新增 `tests/kernels/test_merge_attn_states.py`,参数化 NUM_TOKENS(256~4096)x heads(4~64)x head_size(32~256)x 3 dtype,与 Triton 输出对比(atol 1e-3,bf16 rtol 1e-2),含 inf LSE 输入构造;LSE 对比 CUDA vs Torch diff 为 0。cascade_flash_attn(使用 merge_attn_states)测试通过。风险面:CUDA 路径仅覆盖 CUDA 平台且 `head_size % pack_size == 0`,FP8 dtype 与不满足对齐的 shape 走 Triton 回退;ROCm 编译排除该 kernel;MLA common 后端不再依赖 `HAS_TRITON` 分支提供 merge 函数。

## 关联

同区域(attention 状态合并/级联):索引内 PR #26132(DS-MLA decode 优化)、PR #20034(FlashInfer MLA ragged prefill)、PR #13747(FlashMLA 集成);消费方为 V1 FlashAttention 与 MLA common cascade 路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
