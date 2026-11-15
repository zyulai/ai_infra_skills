# PR #47013: [Fix] kvcache: Skip the reserved padding slot in the zero-RoPE MLA KV writer

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JackChuang |
| 合并 | 2026-11-14 |
| 分类 | Attention / Kernel(tier: major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/47013 |

## 一句话结论

zero-RoPE MLA KV writer(`set_mla_kv_buffer_kernel_norope`)像 RoPE writer 一样跳过 slot 0 保留槽,修复 GLM-5.3-Flash 在 FP8 KV cache + TileLang DSA 下并发请求输出乱码的问题:GSM8K 50.0%→98.0%,且端到端无性能回退(TTFT/TPOT/饱和 prefill 吞吐与 main 差距 1.5% 以内)。

## 动机

自 #36003 起 MLA KV writers 会跳过为 padding rows 保留的 slot 0,但 zero-RoPE writer(每个 GLM-5.3-Flash KV write 都走它,`qk_rope_head_dim = 0`)仍会写入。Padding rows(CUDA graph capture 与 padded replays)以未定义输入写 `out_cache_loc = 0`;自 #42689 起 per-token FP8 量化保留 BF16 NaN 而不再 clamp,zero-RoPE writer 可把 NaN 行写进 slot 0。FP8 TileLang sparse decode 把未使用的 top-k 条目映射到 row 0,而 masking scores 挡不住 NaN(`-inf + NaN`、`0 * NaN`),短于 `index_topk` 的 prompt 解码出 NaN logits。reader 侧替代方案 #45988 在 FP8 decode kernel 里 mask 每 个 KV load,代价是该 kernel 在 GLM-5.3-Flash TP4 shape(16 heads per GPU,top-k 2048)下单 token 1.3x、16–256 token 2.7–3.3x 变慢;本 PR 改为在写侧拦住,decode kernel 不动。

## 优化点

- `set_mla_kv_buffer_kernel_norope` 增加 `reserved_skip_index` 参数,与 `set_mla_kv_buffer_kernel` 完全一致地 mask store;`_set_mla_kv_buffer_impl` 负责传入。torch-native fallback `set_mla_kv_buffer_naive` 对 zero RoPE 也跳过该槽,两条路径保持 bit-identical。
- 只丢弃目标为保留槽的写入,真实 token 行照常写入;覆盖所有 `qk_rope_head_dim = 0` 的 MLA 模型(GLM-5.3-Flash,BF16 或 FP8 KV)。KV allocator 从不发 slot 0 且 pool 对其零初始化,槽位从此保持干净。
- attention kernels 保持原样,decode 速度不变;正文注明 #46225(把 KDA/GDN recurrent kernel 的 padded 输出清零)与本 PR 相互独立、落地顺序无所谓,本 PR 不依赖每个 kernel 都清零自己的 padded 输出。

## 关键改动

```text
python/sglang/kernels/ops/kvcache/mla_buffer.py: norope Triton kernel 增加 reserved_skip_index 并 mask store;naive fallback 同步跳过;_set_mla_kv_buffer_impl 传参
test/registered/kernels/ops/kvcache/test_set_mla_kv_buffer.py: 两个 Triton reserved-slot 测试参数化加 rope_dim=0,skip case 在 main 失败、本 PR 通过
```

```python
loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
is_valid = loc != reserved_skip_index
safe_loc = tl.where(is_valid, loc, 0)
dst_ptr = (
    kv_buffer_ptr
    + paged_row_offset(safe_loc, PAGE_STRIDE, buffer_stride, PAGE_SIZE)
    + offs
)
...
tl.store(dst_ptr, src, mask=mask & is_valid)
```

## 性能声明

- 修复效果(GLM-5.3-Flash,4x H200,H200 CI 测试的 server 参数,FP8 KV cache,GSM8K 200 题):main 50.0% → 本 PR 98.0%;8192-token 截断答案 51.0% → 0.5%;MTP 5/1/6 下 78.0% → 98.0%、截断 21.5% → 0.5%。
- 写侧代价:H200 上以 CUDA graph replay 计时的 zero-RoPE writer kernel,无 padding 行时 1–8,192 行任意规模至多比 main 慢 0.07 µs(1,024 行 2.00→1.99 µs,8,192 行 10.74→10.74 µs);有 1/4 padding 行时 8,192 行 24.20→10.72 µs(更快,因 main 把它们全写进同一槽)。
- 端到端(BF16 KV cache,其余同 accuracy 设置,中位数):TTFT 1k/4k/16k prompt 150/187/609 → 148/184/611 ms;TPOT 1 请求 7.38→7.38 ms,16/64 并发 22.60/47.46 → 22.60/47.44 ms;饱和 prefill 吞吐 28.7k→28.7k tok/s。
- reader 侧对照(#45988):FP8 KV 下端到端 TTFT 16k prompt 656→1,114 ms(约 1.7x),8/32 并发 1.74/4.44 → 2.98/7.68 s,饱和 prefill 吞吐 26.6k→15.3k tok/s,TPOT 64 并发 36.5→39.4 ms / 1,608→1,425 tok/s(decode 至多慢 8%);kernel 级 `tilelang_sparse_fwd` 1 token 19.9→26.5 µs,16/64/256 token 24.7→66.1 / 84.2→254.5 / 305.5→1016.2 µs。

## 验证与回归风险

`test_set_mla_kv_buffer.py` 通过(含新增 zero-RoPE cases),改动文件 pre-commit 通过。per-PR NVIDIA suites(base-a/b/c,1/2/4/8 GPU)与五个 CPU suites 中,本 PR 失败的测试在 parent commit(`535aeae2a7`)同样失败。H200 GLM-5.3-Flash 测试(BF16 KV cache,TileLang DSA,KV writes 全走改动 kernel)在有无本 PR 下都通过:GSM8K 97.3%(本 PR)vs 97.2%(main,with EAGLE)、96.9% vs 97.4%(without)。两个修复相互独立,同时应用时 GSM8K 98.5%。未运行:PD disaggregation、Mooncake/NIXL、`hicache/test_pp_with_hicache.py`、Blackwell GPU、ROCm;PR Test (Extra) 与 AMD ROCm 10 CI 显示失败。风险:行为变化仅是保留槽写入被丢弃,DCP 处理不变(#36886、#40433)。

## 关联

#36003(RoPE writers 已有的 skip 行为)、#42689(FP8 量化保留 NaN)、#45988(reader 侧 masking 的替代修复)、#46225(KDA/GDN padded 输出清零)、#39117 / #40433(DCP rework;本 PR 只落地非 DCP 部分,#40433 可在其上加 DCP 参数 rebase)、#36886。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
