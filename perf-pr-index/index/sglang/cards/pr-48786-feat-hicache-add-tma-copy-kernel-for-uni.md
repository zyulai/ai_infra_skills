# PR #48786: [Feat] HiCache: add tma copy kernel for unified layout

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Shunkangz |
| 合并 | 2026-11-09 |
| 分类 | Attention / Kernel(tier: major) |
| Label | hicache, run-ci, jit-kernel |
| Diff | +447/-31,3 files(pack diffstat 缺失,按 pack 内 diff 文本计数) |
| 链接 | https://github.com/sgl-project/sglang/pull/48786 |

## 一句话结论

page_unified HiCache load-back 在 SM90+ 改用 TMA staging,B200 上 18.0→51.0 GB/s(2.84×)。

## 动机

`page_unified` 布局的 load-back 此前总是选择 two-CTA register-copy kernel,即使 legacy `page_first` 传输早已走 TMA,导致 host→device KV 回载带宽只有约 18 GB/s。PR 把既有 TMA staging pipeline 复用到 page_unified 路径,消除与 legacy 路径之间约 2.8× 的带宽差距。

## 优化点

为 TMA transfer kernel 增加 `kPageUnified` 模板特化:HicacheTmaParams 新增 `src_page_size`/`src_page_stride`/`src_group_stride`,host 入口把 layer/KV 分量偏移折进源基址,并复用 chunk 的 layer 轴遍历 head group。chunk 行数改为取 page size 的约数(`hicache_tma_page_unified_rows_per_chunk`),保证整页索引 run 不跨越 page 边界;索引 run 跨页时强制关闭 contiguous 快路径,退回逐行 TMA 拷贝。TMA 禁用、设备不支持或单个 group row 超过 staging slot 时保留原 register-copy fallback,MLA 路径同样覆盖。

## 关键改动

```text
python/sglang/kernels/jit/csrc/kvcacheio/hicache_tma.cuh: kPageUnified 特化、page-aware 寻址与 run_one_page_unified<kIsMLA> host 入口(TensorMatcher 校验 16B 对齐/布局)
python/sglang/kernels/ops/kvcache/hicache.py: 注册 launch_one_page_unified/launch_one_mla_page_unified,_use_page_unified_tma_kernel 门控并优先选 TMA
test/registered/kernels/ops/kvcache/test_hicache_page_unified_tma.py: 新增注册测试(TMA/register parity、跨页、fallback、CUDA graph)
```

```cuda
auto src_offset = [&](T token) -> int64_t {
  const auto index = static_cast<int64_t>(token);
  if constexpr (kPageUnified) {
    return (index / p.src_page_size) * p.src_page_stride
         + (index % p.src_page_size) * p.row_bytes;
  } else {
    return index * p.src_stride;
  }
};
```

## 性能声明

B200 上 load-back 18.0→51.0 GB/s(2.84× faster),与 legacy 持平。正文表格:legacy `page_first`+TMA 31.54/31.55 ms(51.1/51.0 GB/s);旧 page_unified register copy 89.56/89.22 ms(18.0/18.1 GB/s);新 page_unified+TMA 31.56/31.53 ms(51.0/51.1 GB/s),分别对应 1/2 个 head group,CTA 数 4 vs 2。其余测试条件(buffer 规模、page 配置)PR 未注明。

## 验证与回归风险

新增注册测试(est 40s,base-b-kernel-unit,1-gpu-large):TMA 与 register 两路径 parity,覆盖 page_size 1/3/16/64/128、bf16/fp16/uint8、int32/int64 索引、CPU/CUDA host pool、MLA、空索引与 CUDA graph replay;oversized row(group_bytes 32784)确认仍走 register fallback。风险集中于 page-unified 门控与 16B 对齐假设;kernel 要求 SM90+,ROCm 不涉及。

## 关联

无(素材未提及其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
