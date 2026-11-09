# PR #60084: [ROCm][Perf] Restore 32-bit buffer addressing for TRITON_ATTN KV cache views

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @umarkovi-amd |
| 合并 | 2026-10-25 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | performance, rocm, ready |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 `triton_attn.py` 与 `attention/backends/utils.py` 两个文件,纯 Python 改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/60084 |

## 一句话结论

为 TRITON_ATTN 的 K/V cache view 显式设置 Triton ptr_range,恢复 RDNA3 上 32 位寻址,修掉 #51718 引入的约 16% 吞吐回退(实测 +18%)。

## 动机

#51718 把各层 KV cache 合并进一个 backing allocation 后,单层 cache 变成指向约 36.6 GiB 大分配的 view。Triton AMD backend 只在指针参数的 `ptr_range()` 或整个 storage 不超过 2 GiB 时才用 `tt.pointer_range = 32` 的 32 位 buffer 寻址;storage 检查因此失败,`kernel_unified_attention` 退回 64 位寻址,VGPR 从 205 涨到 256,gfx1100 上 occupancy 从 7 降到 5 waves/SIMD。作者二分确认:#51718 为 5.40 req/s,其 parent 为 6.40 req/s,约 16% 回退。

## 优化点

新增 `set_triton_ptr_range(region, *views)`:按 shape/stride 计算从 view 的 `data_ptr()` 到 region 末端的字节数,写进每个 view 的 `ptr_range()`,让 Triton 不再看整个 storage 的大小。`TritonAttentionImpl` 仅在 ROCm 平台于 `forward` 与 `do_kv_cache_update` 两处对 K/V view 调用它。安全边界:绑定的是 view 的真实范围,Triton 每次 launch 仍检查 2 GiB,超过 2 GiB 的层或 block-outermost 布局继续 64 位寻址;kernel 本来就只通过 block-table 索引/slot mapping 访问 view 内部(#51718 之前的既有假设);view 每次 forward 重建,只读 host 侧元数据,无 GPU 同步。

## 关键改动

```text
vllm/v1/attention/backends/utils.py: 新增 set_triton_ptr_range 工具函数
vllm/v1/attention/backends/triton_attn.py: ROCm 下 forward 与
  do_kv_cache_update 调用 set_triton_ptr_range
```

关键片段:

```python
last = sum((n - 1) * stride for n, stride in zip(region.shape, region.stride()))
region_end = region.data_ptr() + (last + 1) * region.element_size()
for view in views:
    view.ptr_range = functools.partial(int, region_end - view.data_ptr())
```

## 性能声明

W7900 (gfx1100),base 为本 PR 的 parent、同构建、同一 GPU 背靠背各跑 3 次:Whisper-large-v3(fp16, eager, LibriSpeech, 100 prompts × 200 tokens)5.34 → 6.31 req/s(+18.2%,回到 #51718 之前的 6.40 水平);Qwen2.5-7B-Instruct(bf16, TRITON_ATTN, CUDA graphs + compile, random 1024/256, 500 prompts)2.698 → 3.191 req/s(+18.4%)。rocprofv3:Whisper cross-attention 的 `kernel_unified_attention` 中位耗时 1,843 µs → 1,379 µs,约占 GPU 时间 63%。精度无变化:Whisper 2,703 条 LibriSpeech validation-clean 输出逐 token 一致(WER 4.410% 持平);GSM8K 5-shot 79.98/83.32 → 79.76/83.85,在 stderr ±1.1 噪声内。

## 验证与回归风险

编译产物验证:空 `TRITON_CACHE_DIR` 下 `key_cache_ptr`/`value_cache_ptr` 携带 `tt.pointer_range = 32`,VGPR 256 → 205。无新增测试——改动只影响代码生成,CI 无 RDNA runner(作者注明)。风险:安全前提是 kernel 不访问 view 之外的字节;若未来 kernel 越界访问 region 尾部会静默出错。不影响 KV cache 分配、layout、KV connectors/offloading、非 ROCm 平台,无新配置项。

## 关联

回归来源:#51718(单分配 KV cache 重构)。同属 KV Cache / Memory 分类的其他 PR 在索引中可按分类检索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
