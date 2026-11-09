# PR #55528: [Bugfix][KV Cache][MLA] Align packed block strides for V3.2 sparse MLA

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#55528](https://github.com/vllm-project/vllm/pull/55528) |
| 作者 | 200lz |
| 合并日期 | 2026-09-25 |
| labels | bug, documentation, performance, new-model, rocm, frontend, intel-gpu, speculative-decoding, ready, ray, torch.compile, ci/build, multi-modality, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia, quantization, vllm-ir, mistral, DSv4, dflash, rust, kimi, k3, scheduler, kv-cache-manager, glm, minimax, inkling, cohere, DSv4.1 |
| 热度 | 👍 0 · 💬 31 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
V3.2 sparse MLA 未向分配器发布所选 kernel 要求的物理块 stride 对齐;由 producer 上报、字段上提到 KVCacheSpec 并统一 LCM 合并修复。

## 问题现象
V3.2 sparse MLA 所选 kernel 需要物理块 stride 对齐,但该要求从未被发布:分配器不知道要对齐。对 SM100 FlashMLA paged FP8/NVFP4,还额外要求独立的原生 TMA 对齐(`fp8_ds_mla`:656 字节;`nvfp4_ds_mla`:352 字节)且无 flattened-row warmup。B200 验证工作在修正前复现了 FP8 TMA 断言失败。另有 HiSparse hot-page 约束与 stride 约束需合并处理,否则先做的 round-up 可能使前一约束失效。

## 根因
正文披露:V3.2 sparse MLA 没有发布其选中 kernel 要求的物理块 stride 对齐。对齐要求由 backend 与 cache format 选择;旧代码里 `block_stride_alignment` 只存在于 `MLAAttentionSpec` 且分配器用 `isinstance(..., MLAAttentionSpec)` 过滤,HiSparse 热页与 stride 两轮 round-up 分开做,顺序可能互相破坏(后者为正文所述约束合并动机)。

## 修复方案
六文件 +187/-27(采纳 LucasWilkinson 在 200lz/vllm#1 的最小化建议):
- `mla_attention.py`:producer 侧设置对齐——`get_kv_cache_spec` 对 flattened-row 路径或 SM100 TMA 行路径设 `block_stride_alignment=spec.state_content_size_bytes`,并新增 `_uses_flat_kv_cache()` 统一 warmup 判定;
- `kv_cache_interface.py`:`block_stride_alignment` 字段上提到 `KVCacheSpec`(kw_only、校验正值),各 `merge` 传播该字段并断言组内一致;
- `kv_cache_utils.py`:分配器去掉 MLA 类型检查,把热页与 stride 对齐合并进一次 LCM round-up:

```python
alignments = [group.kv_cache_spec.page_size_bytes for group in kv_cache_groups
    if isinstance(group.kv_cache_spec, HiSparseHotSpec)]
alignments.extend(_get_per_layer_spec(group, layer_name).block_stride_alignment or 1
    for group in kv_cache_groups for layer_name in group.layer_names)
return round_up(bytes_per_block, math.lcm(*alignments))
```

新增 `test_sparse_mla_kv_cache_layout.py` 覆盖分配与 warmup 的一致性。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:改动触及所有 KV cache spec 的 merge 与池容量计算路径(字段上提+LCM 合并),BLHNC/BLNHC 既有支持保留;#56254 的对齐传播与池 rounding 逻辑不被重复,上游 #53781 已修 fused-writer 问题。CPU 验证 258 passed / 41 skipped;CUDA 用例在 CPU 主机跳过,历史 B200 结果支持保留的契约但非当前 main 的新全量运行时验证;生产 MTP/DFlash2 正确性、接受率与长上下文行为仍需重测,未声称生产模型质量评估。

## 关联
- [#56254](https://github.com/vllm-project/vllm/pull/56254)(既有对齐传播与 pool rounding)
- [#53781](https://github.com/vllm-project/vllm/pull/53781)(fused-writer 修复)
- [200lz/vllm#1](https://github.com/vllm-project/vllm/pull/55528#issuecomment-5628968592)(最小化建议与 GB200 报告)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
