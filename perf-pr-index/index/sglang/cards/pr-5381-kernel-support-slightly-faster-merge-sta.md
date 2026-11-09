# PR #5381: kernel: support slightly faster merge_state_v2 cuda kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DefTruth |
| 合并 | 2025-04-15 |
| 分类 | Attention / Kernel (major) |
| Label | 无 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 含 6 个文件:新 CUDA kernel、CMake、扩展注册、Python 封装与测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/5381 |

## 一句话结论

为 sgl-kernel 新增 merge_state_v2 CUDA kernel,替换 Triton merge_attn_states,速度约 2 倍,并比 v1 kernel 快 3%-16%。

## 动机

speculative decoding 与 cascade attention 需要把 prefix/suffix 两段 attention 输出按 log-sum-exp 合并,此前依赖 Triton 实现的 merge_attn_states。已有 v1 CUDA kernel(merge_state)只支持 headsize 64/128/256,而 Triton 版本较慢;另外上游 Triton 实现存在 inf 值处理问题会产生 nan(vllm 同款代码中可见)。需要一个更快、覆盖更多 headsize 且正确处理 inf 的合并 kernel。

## 优化点

新增 `merge_attn_states.cu`,实现论文 section 2.2(arXiv 2501.01005)的 attention state 合并:对每个 (token, head) 读取两个 LSE,把 inf 归一为 -inf,计算 max_lse 后求出两路 softmax scale,再对输出做 FMA 加权合并。关键机制是 128-bit(`uint4`)向量化的 pack load/store:float 每 thread 处理 4 个元素、half/bfloat16 处理 8 个,`threads_per_head = head_size / pack_size` 精确铺满 token×head×pack 的线程网格。FMA 一律在 float 域进行以保精度,输出写回时才转回 scalar_t。与 v1 相比放宽了 headsize 限制(只要是 pack_size 的倍数:float 4 的倍数、half/bf16 8 的倍数),并按 `scalar_t` 模板 dispatch 支持 float32/float16/bfloat16。Python 封装对齐 v1 API,且允许传入预分配的输出张量避免重复分配。

## 关键改动

```text
sgl-kernel/csrc/attention/merge_attn_states.cu: 新增 merge_attn_states_kernel 与 merge_state_v2 入口,128bit 向量化 + inf 处理
sgl-kernel/csrc/common_extension.cc: 注册 sgl_kernel::merge_state_v2 torch op
sgl-kernel/python/sgl_kernel/attention.py: merge_state_v2 Python 封装,支持可选预分配输出
sgl-kernel/tests/test_merge_state_v2.py: 150 个正确性+性能用例(含 torch/triton/v1/v2 对比)
sgl-kernel/CMakeLists.txt: 加入新源文件编译
```

向量化合并核心:

```cuda
using pack_128b_t = uint4;
const uint pack_size = 16 / sizeof(scalar_t);
const uint threads_per_head = head_size / pack_size;
...
const float o_out_f = p_out_f * p_scale + (s_out_f * s_scale);
// Pack 128b storage
reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_offset / pack_size] = o_out_pack;
```

## 性能声明

PR 正文给出 L20 GPU 上的微基准表(tokens 256-1536,heads 8/16/32,headsize 32/64/128/256,float16/bfloat16,对照 torch、Triton、CUDA v1、CUDA v2):

- 对 Triton 实现的加速约 1.94-2.59×(如 256tok/8head/32hs float16:0.0391→0.0204 ms,1.9198×;1536tok/32head/256hs bfloat16:0.0583→0.0235 ms,2.4844×)。
- 对 v1 kernel 的加速约 1.02-1.16×(如 1024tok/32head/256hs bfloat16:0.0220→0.0197 ms,1.1174×;1536tok/32head/256hs:0.0273→0.0235 ms,1.1637×)。headsize 32 时 v1 不支持(表中记 0.0000)。
- 单元测试实测示例(1536/32/256 bfloat16):Torch 0.507ms、Triton 0.058ms、v1 0.027ms、v2 0.023ms。

测试条件仅注明 device 为 NVIDIA L20,未注明精度设置之外的更多 workload 信息。

## 验证与回归风险

新增 `test_merge_state_v2.py` 共 150 个参数化用例通过(4.15s),覆盖 NUM_TOKENS×heads×head_size×dtype 组合,构造含 inf 的 prefix/suffix LSE 验证 inf→-inf 处理,以 Triton 输出为参考做 `torch.testing.assert_close`(bf16 rtol=1e-2),同时比较 output 与 LSE 的最大绝对差(CUDA v2 vs Triton 输出差为 0,LSE 差 4.77e-07)。回归风险:v2 kernel 不支持 FP8 与非 CUDA 设备(Python 封装留有 TODO 注明必要时需回退 Triton);headsize 必须是 pack_size 整数倍,不满足时由 TORCH_CHECK 报错;merge_state v1 的 Python 封装同步改为可选预分配输出,签名向后兼容。

## 关联

同区域:v1 `merge_state` kernel(本 PR 之前已有,本卡 v2 与其 API 对齐);引用 vllm 的 triton_merge_attn_states 实现作为 inf 问题的来源。speculative decoding 的多分支合并与 cascade attention 均为潜在受益路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
