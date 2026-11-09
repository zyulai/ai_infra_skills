# PR #36505: [ROCm][Perf] aiter: page-level KV view for gfx950 fp8 page-64 asm prefill

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @omirosh |
| 合并 | 2026-09-26 |
| 分类 | Quantization(tier: major) |
| Label | documentation, amd, run-ci |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及 4 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/36505 |

## 一句话结论

gfx950 fp8 page-64 prefill 改用页级视图就地读 KV pool,免去每层 gather:60k 以上吞吐 +1.1–1.9%,单次调用 1.04–1.22x。

## 动机

gfx950 fp8 hd256 chunked prefill 在 `--page-size 64` 下,main(#36758)每层都要把 prefix KV gather 进连续缓冲:没有页级表时 aiter 把 page-64 调用当作 page_size=1 而落到 CK kernel。gather 有两笔开销——拷贝本身,以及按 forward 总 KV(而非当前 chunk)计的 scratch 缓冲(最大实测 451 MB:8 个 prefill 共享 111k prefix)。aiter 的 paged-varlen ASM kernel(ROCm/aiter#4971,已包含在 main pin 的 aiter `acf8fdf` 中)能按页级视图就地读 pool,两者都省掉。

## 优化点

`aiter_backend.py` 新增 `_build_paged_kv_view()`:把 token 级 KV 表转换为 `(page_indptr, page_ids, last_page_len)`(页 id = slot // page_size;索引算术在 `seq_lens_cpu` 上进行,无 device sync),每 batch 构建一次存入 `ForwardMetadata.paged_kv_view`。构建条件:gfx95、`page_size == 64`、普通 NHD pool(非 sliding-window、非 vectorized 5D)。`forward_extend` 每层计算 `paged_asm_available`(view 存在且 GQA 比为 2 的幂,#36758 gather 分支要求其为 False、paged 分支要求其为 True,一个 flag 读两次保证互斥);paged 分支换入 4D K/V 视图、页级 indptr/table 与 `kv_last_page_lens`,guard 镜像 aiter 自身的 asm guard(该形状无 CK arm,拒绝即 raise 而非回落)。`SGLANG_AITER_PAGED_PREFILL_ASM`(默认开)控制是否构建视图;其他平台或 page size 下 prefill 与 main 逐路径一致。page 64 是 aiter kernel 唯一编译的 page size,与 #37465 的 page-16 MTP ASM verify kernel 无法同服务器同时生效。

## 关键改动

```text
python/sglang/srt/layers/attention/aiter_backend.py: 页级 KV 视图构建与 paged asm prefill 分支
python/sglang/srt/environ.py: 新增 SGLANG_AITER_PAGED_PREFILL_ASM(默认开)
docs/docs/references/environment_variables.mdx: 记录 flag 语义(仅 page 64 生效)
test/manual/test_aiter_paged_kv_view.py: 索引算术与 GQA guard 的 CPU 单测
```

```python
pages = (seq_lens + page_size - 1) // page_size
page_indptr[1:] = torch.cumsum(pages, dim=0)
...
gather = torch.repeat_interleave(tok_base[:-1], pages) + within * page_size
page_ids = (kv_indices[gather.to(device)] // page_size).to(torch.int32)
last_page_len = ((seq_lens - 1) % page_size + 1).to(torch.int32)
```

(摘录有省略;完整实现见 `python/sglang/srt/layers/attention/aiter_backend.py` 的 `_build_paged_kv_view`。)

## 性能声明

amd/Qwen3.8-2.4T-A95B-Quark-MXFP4,8x MI355X,TP8,fp8 KV,chunked-prefill 16384。① 对 released 镜像 v0.5.19-rocm10-mi35x(默认 page 1)双节点:60k/600 输出吞吐 +1.1%–1.9%、128k/1k +0.9%–1.6%,TTFT 好 2%–7%;8k/1k 持平(-0.1%–+0.9%)。② 对 main(同 page 64)单节点:60k geomean 101.3%、128k geomean 101.8%(如 128k/64:131.2→134.0 tok/s;128k TTFT 239.8→234.9 s@conc64);1k/8k 单元在噪声内(98.4%–99.9%)。③ op 级单调用(page 64,hd256,GQA 8:1):paged view + paged asm 比 gather + contiguous asm 快 1.04x–1.22x(比 CK 快 2.8x–3.7x),gather scratch 峰值 451 MB 而本路径为 0;60k trace 确认 kernel 已换为 `aiter::fmha_fwd_hd256_fp8_causal_paged_varlen_gfx950`。accuracy:gsm8k 200 题、page 64:stock 0.985 / main 0.990 / 本 PR 0.995。

## 验证与回归风险

新增 CPU-only 单测:用打乱页分配 + ragged 最后一页验证索引算术(注释指出 identity 表会漏过错误);GQA guard 用例覆盖非 2 幂、除不尽与 0/负 head 数。门控严格:非 gfx95 根本不构建视图;flag 关闭恢复 main 行为;文档注明非长上下文 workload 可设 false(中等 ISL 下持平到略负)。作者如实记录:宿主机存在周期性 whole-server stall(占 5–9% 墙钟,所有 arm 均出现),亚几个百分点的 per-cell delta 单次运行不可分辨;128k conc 64 为容量受限单元。pack 中 CI:Base 通过,Extra 失败标记,AMD ROCm 10 进行中。

## 关联

前序:#36758(gather 路径)、#37465(page-16 MTP ASM verify kernel,与本路径 page size 互斥);上游依赖 ROCm/aiter#4971。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
