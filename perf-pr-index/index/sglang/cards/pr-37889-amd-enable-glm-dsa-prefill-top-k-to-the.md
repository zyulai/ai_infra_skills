# PR #37889: [AMD] Enable GLM DSA prefill top-k to the v2 kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @EricKing626 |
| 合并 | 2026-09-21 |
| 分类 | Attention / Kernel |
| Label | amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/37889 |

## 一句话结论

GLM DSA prefill top-k 走 v2 fused kernel,单次 launch 快约 66%,ISL 70K 下 token 吞吐 geomean +4.9%。

## 动机

#36684/#36851 已把 v2 fused top-k 带到 GLM-5.x on ROCm,但只有 decode 走到了;prefill 仍用 legacy fused transform,需要 gather 宽 `page_size=1` 表。阻塞在寻址而非数学:paged v2 入口假设每行从自己 score row 的第 0 列开始、独占一行 page table;DSA extend 中所有请求共享一个 packed score buffer(行从中间列开始),且一个请求的多行共用同一 page-table 行。用两个查找表(row_starts、row_to_batch)解决,不拷贝 score buffer、不建 per-row page table——后者在 327K context 下每次 forward 要数百 MB。

## 优化点

在 `topk_v2.cuh` 新增 `TopKPackedParams` 与 `topk_packed_kernel<kPDL>` 及 host 入口 `TopKKernel::transform_packed`(整体在 `USE_ROCM` 下,CUDA 从 ragged 路径获得同样 fusion):行窗口为 `scores[i, row_starts[i] : row_starts[i]+seq_lens[i]]`,page-table 行经 `row_to_batch[i]` 映射(缺省按行号 identity);选中索引保持 row-local,输出路径不变。为对齐 16 字节向量化读,kernel 把 `row_starts[i]` 向下取整到 4-float 边界并原地 mask 掉拉入的 ≤3 列,score buffer 原地写、行不得重叠;实现级别(register-2/register-4/streaming)按行在运行时从加宽后的长度选择,因此无需 `topk_v2_plan`、无 cluster path、无编译期 level。`dsa_topk_backend.py` 把 packed PAGED extend 路由到 `_topk_transform_v2_packed`(ks 作 row_starts):因 #36960 把 MQA-logits 预算压到 aiter 2 GiB `buffer_store` 上限,70K prefill 除请求首 chunk 外每个 forward 都会拆行,只路由 whole-forward 形状会让 147 次 launch 中 126 次留在 legacy kernel(由 a6d769e 修复,两种形状都路由)。两种形状回退 legacy:row stride 非 4 倍数、`batch_idx_list` 非 per-row tensor(prefill-CP 的 list 按请求而非行选择)。decode 条件保持逐字节不变(`dsa_drop_wide_page_table` 依赖该条件)。

## 关键改动

```text
python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh: TopKPackedParams + topk_packed_kernel + transform_packed(USE_ROCM)
python/sglang/kernels/ops/attention/dsv4/topk.py: is_hip 时注册 topk_transform_packed wrapper,新增 topk_transform_packed_v2
python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py: packed PAGED extend 分派到 _topk_transform_v2_packed
python/sglang/srt/layers/attention/dsa_backend.py: plan 覆盖范围说明扩展到 packed PAGED extend
test/registered/kernels/ops/attention/test_topk_v2.py: 新增 packed rows 与 level boundary 测试
```

## 性能声明

测试条件:8x MI355X(gfx950),GLM-5.2-MXFP4,TP4,non-MTP,`--max-running-requests 8`,ISL 70000 / OSL 300,`SGLANG_OPT_USE_TOPK_V2=1`;baseline 为 main(v2 开、prefill legacy)。

- Kernel 级(concurrency 4,TP rank 0,按 launch grid 分组):grid 16384:896.8 → 319.7 µs(−64.3%);16383:1972.3 → 649.1 µs(−67.1%);10922:1679.8 → 568.6 µs;8191:1644.5 → 570.5 µs;5462:955.1 → 315.4 µs,单 launch 快约 66%。4 个 extend step 内 top-k 总耗时 185.3 ms → 63.8 ms(相对 2.8s extend kernel 时间)。
- E2E(conc 4-64 geomean):每 GPU token 吞吐 +4.9%(各点 +4.1%/+4.1%/+4.6%/+5.9%/+6.0%),median TTFT −4.8%,median TPOT −3.5%,interactivity +3.7%。conc 4 在当前 main(含 chunked 路由)复测两轮:吞吐 4491.4/4491.9 → 4733.1/4729.0 tok/s(+5.3%),TTFT 3302/3037 → 2846/2852 ms,TPOT 37.65/34.75 → 33.35/33.38 ms。

## 验证与回归风险

GSM8k(1200 题,同硬件):baseline 0.935 / Invalid 0.000,this PR 0.937 / Invalid 0.000。`test_topk_v2.py` gfx950 上 300 passed(含 12 个新 packed 用例);chunked 路由单独验证:4 个并发 ~78K prompt 各嵌 3 个 access code,检索 12/12 无跨请求泄漏;`test_topk_v2_packed_level_boundary` 覆盖 8192/16384 边界加 1-3 列 residue 的溢出风险。风险面:scores 会被原地修改(mask 头部 ≤3 列),调用方不得复用 buffer;kernel 仅 USE_ROCM 编译,CUDA/XPU 不受影响;分派条件含 stride/topk/dtype 守卫,不满足即回退 legacy 不报错。PR CI:Base 通过,Extra 与 AMD ROCm 10 失败(x)。

## 关联

前序:PR #36684、#36851(v2 fused top-k 上 ROCm)、#36960(MQA-logits 2 GiB 预算上限)、#37591(native coop_topk 源)。同批 AMD:PR #32340、#34837。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
