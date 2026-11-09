# PR #58737: [Perf] Fuse the mamba align block-table gather into one Triton kernel, 7 kernels -> 1, 6x speedup on all shapes for this kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fululi12 |
| 合并 | 2026-10-15 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含 2 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/58737 |

## 一句话结论

把 mamba align block-table gather 的 7 个 PyTorch kernel 融合为 1 个 Triton kernel,该 kernel 本体在 MI355X 上快约 6 倍。

## 动机

`mamba_get_block_table_tensor` 的 "align" 分支(取每个请求最后 1+num_speculative_blocks 个 block-table 表项)原先是一段五语句 PyTorch 表达式:sub、div、clamp、arange、add、cast、gather,展开为 **7 次 kernel launch**,作用在 `(num_reqs,)` 张量上——单请求 decode 时只有 1 个元素,开销几乎全是 launch overhead 而非实际计算。在 Kimi-K3 concurrency-1 decode trace 中按 Python 调用点归因 GPU kernel,该点是全模型所有非 model-math 调用点中 kernels-per-invocation 最高的(7.0 kernels、19.6 us/次),其余调用点均已接近 1 kernel/次。这是共享工具函数,`gdn_attn`、`mamba_attn`、`short_conv_attn`、`linear_attn` 都调用它,改动惠及所有 mamba 类 hybrid 模型。

## 优化点

新增 `_mamba_align_block_table_kernel`:每请求一个 program,单次 launch 完成 seq_len→start_index(含 clamp 0 保护 CUDA graph padding 行)、按行读取 n_out 个表项并写出。正确性细节有三处关键设计:(1) `block_table.stride(0)/(1)`、`seq_lens.stride(0)` 显式传入——被替换的 `torch.gather` 支持任意布局,共享工具不能假设列 stride 为 1(初版正是因此被 review 抓出静默读错 block 的 bug);行 stride 标注 `tl.int64` 避免 i32 乘法回绕(4096x625000 表在 MI355X 从 3436 行起 fault);(2) 三个指针加 `do_not_specialize_on_alignment`,否则 MRV1 DBO 路径传入的 ubatch 视图对齐不同会触发 mid-step JIT 重编译(冷 25-28 ms、热 1.5 ms);(3) 保留 CPU 张量的 torch 表达式路径(GDN/linear-attention metadata builder 单测合法传 CPU),并新增 seq_lens 行数/设备一致性断言;`num_reqs == 0` 不 launch 空 grid。收益随 batch 大小平坦,因为代价是 per-launch 而非 per-element。

## 关键改动

```text
vllm/v1/attention/backends/utils.py: 新增 _mamba_align_block_table_kernel 并替换 align 分支实现(CPU 路径保留)
tests/v1/attention/test_mamba_align_block_table.py: 新增 55 个等价测试(48 形状组合 + 非连续表 + 边界情形)
```

## 性能声明

MI355X,同一进程内对照新旧实现(kernel 级微基准,PR 明确声明**非端到端吞吐结论**:相对毫秒级 decode step 该改动处于运行间波动量级,不声称也不应推断出 serving 吞吐提升):num_reqs=1/spec blocks=7 为 10.81 us → 1.78 us(6.06x);1/3 为 10.91→1.79 us(6.11x);8/7 为 10.73→1.82 us(5.90x);64/7 为 12.02→1.95 us(6.16x)。选点依据:Kimi-K3 conc=1 decode trace 中该点 7.0 kernels、19.6 us/次。

## 验证与回归风险

新增 in-tree `tests/v1/attention/test_mamba_align_block_table.py`,55 tests 全过:48 种形状组合(num_reqs 1/2/8/64 x block_size 16/64/256 x nspec 0-7)与原表达式 bit-identical(atol=0, rtol=0);非连续 block table(stride(1)==2)用例在旧版首版 kernel 上失败、修复后通过(真实回归覆盖);seq_len==0 clamp 到第 0 行;num_reqs==0;`"none"` passthrough 返回原张量;CPU 张量等价测试。风险面:kernel 保持与原 `torch.gather` 相同的 in-range 假设(start+nspec 需在行内);新增断言(seq_lens 行数/设备匹配)改变了错误行为;DBO ubatch 视图路径依赖 `do_not_specialize_on_alignment` 注解不丢失。

## 关联

同区域:mamba/gdn attention 后端元数据构建(`gdn_attn`、`mamba_attn`、`short_conv_attn`、`linear_attn` 共用 `mamba_get_block_table_tensor`);风格参照 `kimi_k3/nvidia/kda_metadata.py` 与 `mamba_utils.py` 的 stride 处理惯例。pack 素材未列出明确的前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
