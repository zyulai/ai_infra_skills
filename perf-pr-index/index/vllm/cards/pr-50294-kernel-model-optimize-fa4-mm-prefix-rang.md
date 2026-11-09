# PR #50294: [Kernel][Model] Optimize FA4 mm_prefix range lookup

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vhagor |
| 合并 | 2026-08-05 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | performance, ready, v1, verified |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 内容可用,4 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/50294 |

## 一句话结论

把 FA4 mm_prefix 掩码的范围扫描从每个 QxK 分数内循环移到元数据预计算+按查询 token 查表:Gemma4 视频池化负载 QPS 0.678→1.307(+92.8%),并消除 CuTe JIT 重复编译。

## 动机

Gemma4 多模态模型希望所有层(滑窗 head_dim=256 与 global head_dim=512)统一用 FA4 后端,这是 vision `mm_prefix`/PrefixLM 双向掩码路径的前提。但视频负载中每帧常贡献一个双向 range(`PlaceholderRange.extract_embeds_range()` 把一个视频按帧拆成多个 range),batch 内 `max_ranges` 随帧数(1-32,甚至更高)不断变化:一方面 `max_ranges` 经 `cutlass.range_constexpr` 烘进 CuTe JIT 特化,变化即触发冷编译、GPU 空等;另一方面即使已编译,mask_mod 对每个分数扫描全部 padded range,代价 `O(batch_max_ranges)`,整批被最大帧数拖累。

## 优化点

把范围归属查询从 FA4 掩码内循环移到元数据构建阶段(最终 diff 的实现按"每查询 token 一行"组织):(1) `FlashAttentionMetadata` 的 `mm_prefix_range_tensor (num_seqs, max_ranges, 2)` 换成 `mm_prefix_query_range_tensor (num_actual_tokens, 2)`:每个被调度的查询 token 存其所属 range 的绝对 [start, end],range 外为 (-1,-1) 哨兵;(2) builder 预分配 pinned CPU 暂存 + GPU buffer(按 max_num_batched_tokens 尺寸,build 零分配),`fill_mm_prefix_query_ranges` 在 CPU numpy 填充后 non_blocking 拷贝;(3) mask_mod 只按查询 token 读取 range 边界(`_load_q_range`,两次加载仅依赖 q,可从逐元素掩码循环中 hoist),键侧用 `kv_idx ∈ [r_start, r_end]` 判断,range 不重叠保证语义等价,复杂度从每分数 O(max_ranges) 降为 O(1);`_make_mm_prefix_mask_mod` 加 `functools.cache` 并移除 `max_ranges` 参数(不再进 CuTe compile key,避免闭包 repr 变化导致每次 forward 重编译);(4) mask_mod 是完整掩码 `(causal ∧ window) ∨ mm_prefix`,forward 时显式 `causal=False`、`sliding_window_size=None`(FA #155 后不会自动清);(5) decode-only 步不携带 mm_prefix 元数据(生成 token 必然在所有 prompt range 之后),这使 FULL CUDA graph 捕获与 replay 一致。Triton 参考路径继续用旧张量,本 PR 只优化 FA4。

## 关键改动

```text
vllm/v1/attention/backends/flash_attn.py: 核心改动。metadata 字段替换、builder 持久 buffer + fill_mm_prefix_query_ranges、_make_mm_prefix_mask_mod 缓存化 + _load_q_range(CuTe JIT)、causal/window 显式关闭
vllm/v1/attention/backends/utils.py: fill_mm_prefix_query_ranges(CPU 填充)
vllm/v1/attention/backend.py: CommonAttentionMetadata.mm_req_doc_ranges 文档明确 range 不重叠
vllm/model_executor/models/gemma4_mm.py: 全注意力层清理新增的 mm_prefix_query_range_tensor 字段
```

关键代码(新 mask_mod,替换 range 循环):

```python
r_start, r_end = _load_q_range(q_idx, seqlen_info, aux_tensors, batch_idx)
mm = (kv_idx >= r_start) & (kv_idx <= r_end)
if sliding_window > 0:
    mm = mm & ((q_abs - kv_idx) < sw)
keep = keep | mm
```

## 性能声明

单张 H100,Gemma 4 31B BF16,FA4 + vision mm_prefix;pooling embedding 工作负载(非生成):1000 真实请求、并发 32,每请求 text+video,视频 8-64 帧(均值 48),每帧 360x640,文本平均 885 token。QPS(完成的池化请求/秒):现有 FA4 range 扫描(带 CuTe JIT)0.103;不带 CuTe JIT 0.678;per-token range 方案 1.307;Triton 参考路径 0.743。即较现有 FA4 路径 +92.8%,并反超 Triton 路径。正确性:新旧行为输出 bit-exact;对参考路径平均绝对差约 0.00646(与历史 FA4/Triton 差异同量级)。冷启动收益来自避免 max_ranges 变化触发的重复 CuTe 编译。

## 验证与回归风险

PR 正文提到 `tests/v1/attention/test_mm_prefix_range_id.py`(4 passed);diff 新增 636 行 `tests/v1/attention/test_mm_prefix.py`:元数据测试随处可跑,FA4 测试需 SM100,用 dense float32 参考逐点对照(含 chunked prefill 中完全落在已计算上下文的 range 不拓宽掩码、paged KV + block_table 路径 hd128/hd512、decode-only batch 无 mm_prefix 元数据以保证 cudagraph 一致性)。风险面:改动共享 FlashAttention backend 的 metadata 与 forward 路径(仅 FA4 + mm_prefix 分支受影响);`functools.cache` 缓存 mask_mod 工厂;Gemma4 全注意力层的字段清理需与 builder 生命周期匹配。

## 关联

同区域:Gemma4 FA4 全层启用与 mm_prefix 相关的正确性/FP8 工作(PR 正文提及但未给编号);Gemma4 统一 FA4 路径见 PR #42175;分类同属 CUDA Graph / Compiler 的其他 PR 见索引。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
