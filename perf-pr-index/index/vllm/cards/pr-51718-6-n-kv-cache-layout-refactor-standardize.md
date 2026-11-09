# PR #51718: [6/N][KV-Cache Layout Refactor] Standardize KV cache layout

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LucasWilkinson |
| 合并 | 2026-08-22 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | documentation, performance, rocm, kv-cache-manager, kv-connector, nvidia, ready-run-all-tests 等多个 |
| Diff | diffstat 缺失(pack 未提供加减行数);pack 只保留了测试/benchmark 类 diff,另有 144 个文件因超长未含于 pack(核心 runtime 改动不在本 pack 中) |
| 链接 | https://github.com/vllm-project/vllm/pull/51718 |

## 一句话结论

KV-cache 布局标准化系列的核心:所有 KV cache 统一到逻辑 [L,B,H,N,C] 词汇与单一 layout 解析点,删除各后端自带的 shape/stride-order 钩子。

## 动机

这是 KV-cache layout 标准化系列(RFC #42082)的第 6 步。此前每个 attention backend 各自暴露 `get_kv_cache_shape()`/`get_kv_cache_stride_order()`(及 `get_required_kv_cache_layout()`),allocator 与测试需要为每个后端做 shape/stride 反推与 permute,NHD/HND 等别名散落在多处,布局知识没有单一权威来源,新增 backend 或布局都要改多处代码(如 indexer backend 还需额外 override 才能避免 allocator assert)。

## 优化点

(1) 引入 `KVCacheLayout` 枚举,列出 6 种物理 stride 排列(LBHNC、LBNHC、LHBNC、BLHNC、BLNHC、BHLNC),所有 KV cache 分配统一到逻辑 `[L, B, H, N, C]` 词汇,通过 `create_kv_cache_views`/`compute_layout_strides` 从原始 buffer 生成 backend 视图。(2) layout 解析改为单一 writer:attention backend 选择时把 layout 写入 `CacheConfig`(优先级:test override > backend-required > `VLLM_KV_CACHE_LAYOUT` > connector preference > 默认 LBNHC),所有 consumer 只读它;legacy NHD/HND 名字经 `_layout_from_name` 映射到 LBHNC/LBNHC。(3) 删除 `indexes_kv_by_block_stride` 与各 backend 的 per-backend cache-shape/stride-order 钩子——标准化视图使其成为常量。pack 内 diff 显示测试与 benchmark 改造:`dense_kv_cache_views` 按生产 allocator 同路径构造缓存视图,MLA cache 逻辑形状变为 4D `(num_blocks, 1, block_size, head_size)`,indexer 侧 H=1 的 4D 视图 squeeze 后保持 contiguous。

## 关键改动

```text
vllm/v1/kv_cache_interface: KVCacheLayout 枚举、compute_layer_kv_cache_shape_bytes、compute_layout_strides、create_kv_cache_views(核心 runtime 文件未含于 pack diff)
vllm/v1/attention/backends/utils: get_supported_kv_cache_layouts / resolve_kv_cache_layout / record_kv_cache_layout 取代 get/set_kv_cache_layout 全局态
tests/...: 各 fusion/cache 测试改用 customize_spec + dense_kv_cache_views;删除 backend stride-order 相关负例测试
benchmarks/attention_benchmarks/runner.py: 按标准 allocator 路径创建 cache
```

关键代码摘录(节选,<=15 行):

```python
# MLA KV cache is 4D: (num_blocks, num_heads=1, block_size, head_size)
kv_cache = torch.zeros(
    (num_blocks, 1, self.block_size, self.head_size),
    dtype=self.kv_cache_dtype, device=self.device,
)
self.mla_attn.bind_kv_cache(kv_cache)
# legacy aliases: HND -> LBHNC, NHD -> LBNHC
assert KVCacheLayout.LBHNC.layer_view_order == (0, 1, 2, 3)
assert KVCacheLayout.LBNHC.layer_view_order == (0, 2, 1, 3)
```

## 性能声明

- PR 正文无性能数字(布局标准化属结构性重构,目的是消除 per-backend 分支与减少拷贝/permute 路径,本 pack 未提供 benchmark 数据)。
- Test Plan/Result:FULL CI Run,无 PR 相关失败。

## 验证与回归风险

- 验证为完整 CI;diff 内可见 test_minimax_m3.py 大幅重写:layout contract、AITER sparse PA、indexer 3D squeeze、HND packed head-major 分配、byte-identical 生产 allocator 对照(AC-2)等正负例;rope-kvcache/qk-norm-rope fusion 测试参数化为 LBHNC/LBNHC 两种布局。
- 回归风险面:全 repo 的 KV cache 分配与消费路径(144+ 文件)——任何依赖旧 `get_kv_cache_shape`/`get_kv_cache_stride_order` 钩子或全局 `set_kv_cache_layout` 的外部代码会失效;MLA cache 从 3D 逻辑形状改为 4D;`SlidingWindowMLASpec`、connector preference 等特例需在解析优先级中正确落位;FlashInfer TRTLLM 在 SM100 上仍要求 LBHNC(原 HND)。

## 关联

- 系列:RFC #42082(KV-cache layout 标准化);替代 #44458(同内容 restack);为该系列 6/N。
- 同区域:索引库已有 pr-56492(extensible KV cache)、pr-46205(packed HMA KV cache layout)、pr-51739(long-context MLA cache gather)等 KV cache 卡片。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
