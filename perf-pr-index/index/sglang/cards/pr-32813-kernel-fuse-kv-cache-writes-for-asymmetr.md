# PR #32813: [Kernel] Fuse KV-cache writes for asymmetric K/V (head_dim != v_head_dim)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-07-30 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci, jit-kernel |
| Diff | 素材未注明 diffstat(diff 片段可见 6 个文件,含 1 个新增测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/32813 |

## 一句话结论

让 fused KV-cache 写入 JIT 内核支持 head_dim != v_head_dim 的非对称 K/V,MiMo-V2.5 decode 每步 KV 写内核从 102 降到 51 个、KV 写 GPU 时间 -76%。

## 动机

MHATokenToKVPool 此前仅在 `head_dim == v_head_dim` 时启用 fused `store_cache` JIT 内核;非对称 KV 模型(如 MiMo-V2.5,K/V 行宽 192/128)每层退化为两次独立的 `index_put_` 写入,decode 路径上 KV 写成为小内核堆积的瓶颈。

## 优化点

把 JIT 内核按 `(k_row_bytes, v_row_bytes)` 二元组特化(此前只有单一 `kElementBytes` 模板参数),并为 K/V 引入独立的 cache slot stride(`stride_k_cache_bytes`/`stride_v_cache_bytes`),使非对称 K/V 也能走 fused 路径。拷贝策略上新增 `copy_kv_rows_warp`:K/V 行的公共前缀宽度走原有 interleaved 双行拷贝,宽度差的部分由新增 `copy_row_warp` 串行收尾;等宽 K/V 退化为原来的单次 interleaved 拷贝,行为不变。向量化计划抽取为 `RowVecPlan` 供 interleaved 与单行拷贝共享;num_split 启发式改为要求 K/V 双侧都对齐(split 必须同时整除两行宽度)。CPU AMX 路径同步收紧为仅服务等宽行;`set_kv_buffer_prefix_valid` 的 tiled 内核只接受单一行宽,对非对称 KV 显式抛 `NotImplementedError` 而不是静默写坏。

## 关键改动

```text
python/sglang/kernels/jit/csrc/elementwise/kvcache.cuh: 内核按 (k_row_bytes, v_row_bytes) 特化;独立 K/V cache stride;新增 copy_kv_rows_warp/copy_row_warp/RowVecPlan
python/sglang/kernels/ops/kvcache/kvcache.py: store_cache 增加 v_row_bytes 参数;can_use_store_cache 按双侧行宽判定
python/sglang/srt/mem_cache/memory_pool.py: 池层以 v_row_dim(head_num*v_head_dim)替代 same_kv_dim 并透传;CPU AMX 与 prefix-valid 路径加等宽守卫
test/registered/kernels/ops/kvcache/test_store_cache.py: 非对称行宽正确性 + num_split 分支测试
test/registered/unit/mem_cache/test_asymmetric_mha_pool.py: 新增;池层接线与 set_kv_buffer 往返测试(CUDA)
```

```cuda
template <int64_t kKBytes, int64_t kVBytes>
SGL_DEVICE void copy_kv_rows_warp(...) {
  constexpr auto kCommon = kKBytes < kVBytes ? kKBytes : kVBytes;
  constexpr auto kTail = (kKBytes < kVBytes ? kVBytes : kKBytes) - kCommon;
  ...
  if constexpr (kCanInterleave) {
    copy_kv_warp<kCommon>(k_src, v_src, k_dst, v_dst);
    if constexpr (kTail > 0) { /* 单行收尾拷贝较宽一侧 */ }
  } else {
    copy_row_warp<kKBytes>(k_src, k_dst);
    copy_row_warp<kVBytes>(v_src, v_dst);
  }
}
```

## 性能声明

MiMo-V2.5 (310B),TP=4,bs=1 decode:每步 KV-write 内核数 102 → 51,KV-write GPU 时间 -76%,output throughput +2.9%,生成 token 逐字节一致(byte-identical)。

## 验证与回归风险

三层测试:内核级 `test_store_cache.py` 新增非对称行宽(192/128、128/192、1024/512 等)在 bf16/fp32 下的正确性(含未命中 slot 不被越界写坏断言)与非默认 num_split 分支;池级新增 `test_asymmetric_mha_pool.py`(注册 CUDA CI,base-b),验证 `v_row_dim` 接线、SWA 子池覆盖、`set_kv_buffer` 往返及 prefix-valid 对非对称 KV 的拒绝;基准侧 `bench_store_cache.py` 新增 `benchmark_asymmetric`。风险面:JIT 内核模板与 stride 语义变化影响所有走 fused KV 写的模型(等宽路径要求行为不变,由既有测试覆盖);CPU AMX 路径对非对称 KV 回退到朴素路径;prefix-valid 路径对非对称 KV 从"可能写错"变为显式报错。pack 内 CI 状态显示 PR Test (Extra) 为失败态。

## 关联

同区域:KV-cache 写入/内存池相关 kernel PR(素材未列出编号);上游受益模型为 MiMo-V2.5 类非对称 KV(head_dim 192 / v_head_dim 128)架构。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
