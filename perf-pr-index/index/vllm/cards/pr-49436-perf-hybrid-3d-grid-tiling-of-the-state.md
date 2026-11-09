# PR #49436: [Perf][Hybrid] 3D-grid tiling of the state-copy Triton kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fuscof-ibm |
| 合并 | 2026-08-10 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready, v1 |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 3 个文件(mamba_utils kernel 与两个测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/49436 |

## 一句话结论

给 hybrid 模型 mamba state 拷贝 Triton kernel 加 3D grid 分块,reqs=1 时 kernel 带宽提升 6-7.7x,E2E 吞吐 +1.1-1.8%。

## 动机

前作 #48110 已用 64-bit load/store 向量化 hybrid 模型(mamba/conv 状态)的块边界状态拷贝,但小 batch 时拷贝仍远未吃满 HBM:grid 为 `(num_reqs, total_states)`,reqs=1 时只有少数 CTA,大量 SM 空转;同时 kernel 硬性要求状态张量 8 字节对齐(assert),限制了一般布局。

## 优化点

核心是把每个 temporal 状态拷贝的 u64 主体按新的 `program_id(2)` 轴分给 `TEMPORAL_TILES` 个 CTA,grid 从 2D 变为 `(num_reqs, total_states, TEMPORAL_TILES)`,小 batch(reqs=1..16)也能填满 SM。具体改动:新 device function `_memcpy_u64_tiled` 实现 head/body/tail 拷贝——fast path(src/dst 共享 sub-8B 对齐)由 tile 0 拷 ≤7 B 的 head(把 dst 对齐到 8B)与 tail,主体 u64 向量化并按 tile 切分(per-tile 向上取整到 COPY_BLOCK_SIZE 保证内层循环满宽);slow path(src/dst 对齐不一致)退化为按字节分块拷贝,规避 GB200 等硬件对非对齐 u64 load 报 `cudaErrorMisalignedAddress` 的问题。SD conv 状态(~60-80 KiB/block)并入同一 fused launch 与 `_copy_mamba_state_block`,`tile_idx > 0` 早退保持单 CTA。原 setup 阶段的 8B 对齐 assert 降级为一次性 `warning_once`(对齐只是吞吐提示而非正确性前提)。`TEMPORAL_TILES=16` 由 {8,16,32} 在 GB200 上的 sweep 确定。`TEMPORAL_TILES=1` + `tile_idx=0` 完整复现旧的 2D 行为。

## 关键改动

```text
vllm/v1/worker/mamba_utils.py: _memcpy_u64_tiled(head/body/tile 分块);_copy_mamba_state_block 接收 tile_idx;两个 fused kernel 增加 TEMPORAL_TILES;grid 改 3D;assert 改 warning_once
tests/kernels/mamba/test_memcpy_u64_tiled.py: 新增,head/body/tail 与 tile 边界的字节级恒等测试
tests/kernels/mamba/test_precopy_mamba_align.py: 参数化 TEMPORAL_TILES ∈ {1, 16}
```

tiled memcpy 主体(节选):

```python
body_u64 = body_bytes // 8
per_tile_u64 = tl.cdiv(tl.cdiv(body_u64, NUM_TILES),
                       COPY_BLOCK_SIZE) * COPY_BLOCK_SIZE
tile_start = tile_idx.to(tl.int64) * per_tile_u64
tile_end = tl.minimum(tile_start + per_tile_u64, body_u64)
for i in range(tile_start, tile_end, COPY_BLOCK_SIZE):
    mask = (i + offsets) < tile_end
    data = tl.load(src_body_u64 + i + offsets, mask=mask)
    tl.store(dst_body_u64 + i + offsets, data, mask=mask)
```

## 性能声明

Microbenchmark(NVIDIA GB200,峰值约 8 TB/s,num_reqs ∈ {1..128}):Qwen3.5-9B reqs=1 拷贝带宽 585.7 对 4519.1 GB/s(7.72x,7.3% 对 56.5% 峰值),reqs=4 为 2.74x,reqs=128 仍 1.09x(84.4% 峰值);Nemotron-3-Super-120B-A12B-NVFP4 reqs=1 为 949.9 对 5740.9 GB/s(6.04x)。kernel 延迟:Qwen3.5-9B reqs=1 178.58 对 23.15 µs(7.71x),reqs<16 增益最大。E2E(Qwen3.5-9B + qwen3_next_mtp 2 spec tokens + prefix caching,MRv2,GB200,random 500 in/5330 out、500 prompts、ignore-eos,每并发 4 次取后 3 次):输出 tok/s 在 conc=16/32/64 分别 +1.81%/+1.23%/+1.12%;median TPOT -1.39%/-1.32%/-1.22%,median ITL -1.62%/-1.24%/-0.68%,Mean E2EL -1.61%/-1.24%/-0.97%;MTP acceptance 两侧均约 98.5-98.7%,无任何指标回退。

## 验证与回归风险

单元与集成测试全部通过:`test_memcpy_u64_tiled.py` 对 copy_size ∈ [0, 4 KiB]、全部 sub-8B src/dst 对齐组合、NUM_TILES ∈ {1, 16} 做 torch slice copy 的字节恒等比对;`test_precopy_mamba_align.py` 的 V1/V2 等价测试参数化 tile 数;`tests/models/language/generation/test_hybrid.py` 覆盖 hybrid 模型生成。回归风险:kernel 语义由"仅 u64 主体"扩展为 head/body/tail 全覆盖,慢路径(misaligned src)按字节拷贝性能较低但正确;`num_accepted_tokens` 写入增加 `tile_idx == 0` guard 防止多 tile 重复写;grid 变 3D 后调用方必须传 `TEMPORAL_TILES`,默认 1 保留旧契约;影响所有使用 mamba/hybrid 状态拷贝的模型路径(Qwen3.5、Nemotron 等)。

## 关联

前序 #48110(同文件同 kernel 的 uint64 向量化,本 PR 为其严格 follow-on);microbenchmark 脚本复用 #48110 引入的 `bench_copy_mamba_state_block.py` 并扩展。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
