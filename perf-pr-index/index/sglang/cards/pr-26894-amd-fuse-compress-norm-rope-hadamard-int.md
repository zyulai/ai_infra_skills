# PR #26894: [AMD] Fuse compress norm+rope+hadamard into single Triton kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yichiche |
| 合并 | 2026-06-03 |
| 分类 | Attention / Kernel (major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 含 2 个文件:DSV4 compressor 调度与 Triton kernel |
| 链接 | https://github.com/sgl-project/sglang/pull/26894 |

## 一句话结论

把 DSV4 压缩器 decode 路径的 RMSNorm+RoPE 与 Hadamard 旋转融合为一个 Triton kernel,C 型层 kernel 时间 16.6→8.1 us。

## 动机

DeepSeek-V4 compressor 在 MI355 上的 decode 路径中,压缩后的 `kv_compressed` 张量([num_tokens, 512])要背靠背跑两个 kernel:`_compress_norm_rope_kernel`(RMSNorm+RoPE,约 4.0 us)与 `fast_hadamard_transform_kernel`(Hadamard 旋转,约 4.1 us)。两者操作同一份数据,且完整 head_dim 512 可放进寄存器,分开执行意味着一次多余的 kernel launch 与一次全局内存往返。C4 压缩器每次调用 rotate=True 都会触发(C 型层每层 2 次调用),C128 压缩器 rotate=False 不受影响。

## 优化点

新增 `_compress_norm_rope_hadamard_kernel` 单个 Triton kernel,按序完成 RMSNorm(rsqrt(mean(x²))+eps 缩放)、尾部 RoPE 维的复数旋转、以及 Walsh-Hadamard butterfly 变换(dim=512 共 9 个 stage)。RoPE 结果先写回 kv 缓冲,butterfly 各 stage 通过 L1 cache 的 store-reload 模式读写同一行(每行 2KB),stage 间用 `tl.debug_barrier()` 保证多 warp 共享一行时的读写顺序,最后一级乘 hadamard_scale(head_dim^-0.5)。Python 侧新增 `hip_compress_fused_norm_rope_hadamard_inplace()` 封装,要求 head_dim 为 2 的幂;compressor 调度上 rotate=True(C4)走新融合 kernel,rotate=False(C128)保持原 `hip_compress_fused_norm_rope_inplace()`,省去原先的 `rotate_activation()` 调用。kernel 同时支持 decode(按 seq_lens 行)与 prefill(按 compress_plan 行)两种寻址。

## 关键改动

```text
python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py: 新增 _compress_norm_rope_hadamard_kernel 与 hip_compress_fused_norm_rope_hadamard_inplace 封装
python/sglang/srt/layers/attention/dsv4/compressor.py: forward_compress 中 rotate=True 分支改派发到融合 kernel
```

butterfly 核心(store-reload 过 L1,屏障同步):

```python
for stage in tl.static_range(LOG2_HEAD_DIM):
    stride = 1 << stage
    is_even = ((offs >> stage) & 1) == 0
    partner = tl.where(is_even, offs + stride, offs - stride)
    tl.debug_barrier()
    x_self = tl.load(kv_ptr + base + offs, mask=mask)
    x_partner = tl.load(kv_ptr + base + partner, mask=mask)
    result = tl.where(is_even, x_self + x_partner, x_partner - x_self)
    if stage == LOG2_HEAD_DIM - 1:
        result = result * hadamard_scale
    tl.debug_barrier()
    tl.store(kv_ptr + base + offs, result, mask=mask)
```

## 性能声明

测试条件:DeepSeek-V4-Pro,TP8,AMD MI355,attention backend compressed,page size 256,chunked prefill 8192。

- Kernel 替换(C 型层,decode):before 为 `_compress_norm_rope_kernel`×2(8.3 us)+ `fast_hadamard_transform_kernel`×2(8.3 us),共 16.6 us;after 为融合 kernel×1(4.1 us)+ C128 rotate=False 的 norm_rope×1(4.0 us),共 8.1 us。
- Kernel 到 E2E 推演:30 个 C 型层 × 6.3 us/层 = 每迭代节省 189 us;期望 ITL 改善 189 us / 20,570 us = **0.92%**,PR 自述这在 1-2% 的 run-to-run 噪声内,即 kernel 级提升真实但 E2E 精度下不可分辨;TTFT 影响可忽略。
- E2E benchmark(il8k_ol1k,concurrency 4,32 prompts):Total throughput 1560.0→1583.5 tok/s(+1.5%),Output throughput 173.3→175.9 tok/s(+1.5%),Median TPOT 21.51→21.10 ms(-1.9%),E2E/TTFT/ITL 中位数约 -0.0%。

## 验证与回归风险

准确率:GSM8K(2000 题、5-shot、parallel 1000)得分 0.950,阈值 0.94,PASS。CI 状态显示 PR Test (Base) no_entry_sign、PR Test (Extra) 失败,pack 未见新增单测(checklist 未勾选)。风险面:融合 kernel 要求 head_dim 为 2 的幂(assert),仅接在 DSV4 compressor 的 rotate=True 路径;barrier 同步的 store-reload butterfly 依赖 L1 缓存行为,语义上改变了原两 kernel 间的一次往返精度路径(Hadamard 在 FP32 中间量上做);C128 与非 rotate 路径保持原实现。

## 关联

同区域:DSV4 compressor 系列优化、`_compress_norm_rope_kernel`(本 PR 复用其 norm+rope 逻辑并追加 butterfly);pack 未列出其他具体编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
