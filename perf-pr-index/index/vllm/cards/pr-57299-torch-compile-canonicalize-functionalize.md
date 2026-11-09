# PR #57299: [torch.compile] Canonicalize functionalized split slices for fusion pa…

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @laithsakka |
| 合并 | 2026-09-29 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | ready, torch.compile |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 本身可用(2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/57299 |

## 一句话结论

修复 PyTorch 2.15 functionalization replay 改动导致的 torch.compile 融合失效:把完整连续 slice 分区重写回 split_with_sizes,恢复 QKNorm+RoPE 融合(0→6)并消除 slice_scatter。

## 动机

PyTorch 2.15(pytorch#194037)改变了多输出 view 的 functionalization replay:`split_with_sizes` 的各输出现在被重新生成为独立的 contiguous slice 节点。vLLM 中依赖"canonical split_with_sizes + getitem"表示的 pass 因此失配:QKNormRoPEFusionPass 报告 `norm_rope_fusion=0`(期望 6),ScatterSplitReplacementPass 在图中留下两个 `slice_scatter` 操作,相关内核融合与替换静默失效。

## 优化点

扩展 `SplitCoalescingPass`,新增 `_canonicalize_slices`:收集同一源张量、同一维度上的 `aten.slice.Tensor` 节点,当分区完整、连续、unit-step(要求具体数值边界、正宽度、覆盖全维度、无 gap 与 overlap、源与 slice 的 tensor meta 可用,不支持 dynamic slice/extent)时,重写为一个 `split_with_sizes` 节点加索引 `getitem` 用户,getitem 的 meta 从原 slice 节点复制。这样下游模式匹配 pass(QKNorm+RoPE 融合、scatter-split 替换)重新看到规范表示。同时修复该 pass 的既有正确性 bug:旧的 coalescing identity 只比较输入张量与 split sizes,会把"同 input、同 sizes 但 split 不同维度"的两个节点误判为重复(如 `split_with_sizes(x, [4,4], dim=0)` 与 `dim=1`)并互相替换;新版把 normalize 后的 split 维度纳入 identity,只有语义等价的 split 才合并。

## 关键改动

```text
vllm/compilation/passes/utility/split_coalescing.py: 新增 _canonicalize_slices(slice 分区→split_with_sizes 重写);_coalesce_splits 按 (source, normalized dim) 合并
tests/compile/passes/test_split_coalescing.py: canonicalize、unsafe 拒绝、嵌套 slice、不同维度不合并等回归测试
```

```python
# 重写前:
q = slice(qkv, 0, q_size)
k = slice(qkv, q_size, q_size + kv_size)
v = slice(qkv, q_size + kv_size, q_size + 2 * kv_size)
# 重写后:
qkv_split = split_with_sizes(qkv, [q_size, kv_size, kv_size])
q = qkv_split[0]; k = qkv_split[1]; v = qkv_split[2]
```

## 性能声明

PR 未给出吞吐/延迟数字,效果以 pass 统计与 op 计数表述。修复前:`norm_rope_fusion expected: [6], found: [0]`;`test_scatter_split_replace` 中 `aten.slice_scatter.default` 计数 fp16/bf16 均为 2(断言应为 0)。修复后上述测试全部通过。端到端冒烟:`tests/compile/fusions_e2e/test_tp1_quant.py` 的 Qwen3-30B-A3B-FP8 TP1 FP8 fusions(inductor_partition,TRITON_ATTN)用例通过。

## 验证与回归风险

新增测试覆盖:完整 QKV slice 分区被 canonicalize(3 个 slice→1 个 split_with_sizes,getitem 索引与 meta 对齐);六种 unsafe 组合(gap、overlap、缺前缀、缺后缀、单一全范围 slice、非 unit step)被拒绝;嵌套 slice 分区使用 rewired source;不同维度等尺寸 split 保持独立。另运行 `test_scatter_split_replace.py` 与 `test_qk_norm_rope_fusion.py`。回归风险:该 pass 作用于所有走 torch.compile 的图;重写条件保守(仅静态完整连续分区触发),但 split/getitem 的 meta 传递正确性影响下游 Inductor 代码生成。

## 关联

vLLM #33295(原始 duplicate-split 问题)、PyTorch #174472(上游 CSE gap)、PyTorch #194037(multi-output views replayed as individual slices)。同分类区域为 CUDA Graph / Compiler,pack 未列出其他同区域 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
