# PR #58838: [vLLM IR][RoPE] Port base RotaryEmbedding to a vLLM IR op (CUDA + ROCm/AITER)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mpashkovskii |
| 合并 | 2026-11-18 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, performance, rocm, ready, torch.compile, deepseek, nvidia, vllm-ir |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数);pack diff 段注明另有 37 个文件因超长被略去,可见文件含 csrc/pos_encoding_kernels.cu、csrc/ops.h、csrc/torch_bindings.cpp、tests/kernels/ir/test_rotary_embedding.py(新增)等 |
| 链接 | https://github.com/vllm-project/vllm/pull/58838 |

## 一句话结论

把基础 `RotaryEmbedding` 从普通 `CustomOp` 迁移为 vLLM IR op(`vllm.ir.ops.rotary_embedding`),使 QK-norm + RoPE(+KVCache)融合 pattern 在不强制 `custom_ops=["+rotary_embedding"]` 的情况下稳定匹配,并把 CUDA/ROCm(AITER)kernel 选择交给 IR op priority。

## 动机

此前 RoPE 只有在注册为 custom op 时 fusion 才能匹配,`vllm/config/compilation.py` 里留有 `# TODO: support rope native forward match` 的强制追加,对应长期未决的 pattern-match 失败 #28042;早期迁移尝试 #43199(wxsIcey)及其前身 #39488/#41773/#42264 均已关闭未合并(merge 冲突)。另外,去掉 `+rotary_embedding` 强制会静默禁用 ROCm 上的 AITER RoPE。

## 优化点

1. 新 functional RoPE IR op(`activations=["query","key"]`,`allow_inplace=True`)及 query-only 变体;RoPE 在 native/CUDA/HIP 全部前向路径发射同一 IR node。
2. Providers:`vllm_c`(`_C.rotary_embedding`)与 `aiter`(`rocm_aiter_triton_rotary_embedding`);CUDA/ROCm 平台设置 `rotary_embedding` IR-op priority(ROCm 在启用 AITER + fusion 时优先 `aiter`),kernel 选择从全局 custom-op flag 改为 lowering 时的 IR op priority。
3. 保留 `MatcherRotaryEmbedding` 并让其发射 IR node(main 在 #43199 之后新增的 `QkNormRopeKvCacheFusionPass`(#42749)依赖它,直接删会破坏 AITER 融合)。
4. 新增 out-of-place CUDA kernel `rotary_embedding_out`:从 src 读 heads、写旋转后的 heads 到 dst 并复制非旋转 tail(partial rotary 下 dst 完整定义)。
5. 清理 `vllm/config/compilation.py` 中三处 `+rotary_embedding` 强制追加;范围仅限基础 `RotaryEmbedding`(Neox + GPT-J、full/partial rope),`DeepseekScalingRotaryEmbedding`、mRoPE、flashinfer、`key=None` 跨层 KV 共享路径留待后续。

## 关键改动

```cuda
// csrc/pos_encoding_kernels.cu(新增 out-of-place kernel)
+__global__ void rotary_embedding_out_kernel(
+    const int64_t* __restrict__ positions,  // [num_tokens]
+    const scalar_t* query, const scalar_t* key, scalar_t* query_out,
+    scalar_t* key_out, const cache_t* __restrict__ cos_sin_cache, ...)
```

```python
# tests/kernels/core/test_pos_encoding.py(测试改为显式选 kernel)
+    with (
+        ir.ops.rotary_embedding.set_priority(["vllm_c", "native"]),
+        ir.ops.rotary_embedding_query_only.set_priority(["vllm_c", "native"]),
+    ):
+        out_query, out_key = rope.forward(positions, query, key)
```

## 性能声明

PR 正文未提供 benchmark 数字与测试条件(perf claims 检测为 none;pack 未提供)。收益定性为:fusion pattern 匹配不再依赖 `custom_ops`,AITER RoPE 不再因去掉 custom-op 强制而失效;`benchmarks/kernels/benchmark_rope.py` 在 diff 中被改造为按 IR priority 计时,但正文未附结果。

## 验证与回归风险

- PR 正文自述为 draft、"static-checked only":`python -m py_compile` 通过全部 11 个改动文件;运行时验证(标注 pending)包括 `pytest tests/kernels/ir/test_rotary_embedding.py`、`tests/compile/passes/test_qk_norm_rope_fusion.py`、`tests/compile/passes/test_rocm_aiter_qk_norm_rope_kvcache_fusion.py` 与一个计划中的 gsm8k parity 对照;作者声明在离开 draft 前补齐。注意 pack 元信息显示该 PR 已于 2026-11-18 合并/关闭,与正文 draft 状态并存,以 pack 记录为准。
- 回归风险:RoPE lowering 路径改变,精度 parity 必须确认;AITER bf16 cache 拷贝(`cos_sin_cache_bf16`)是设计内例外,`test_rope_module_cache` 相应放行;partial rotary(rotary_ratio 0.5)不参与 fused RoPE+KV-cache 融合(测试断言 matched_count == 0);不支持的 AITER 形状(如 head 96)必须回退 `vllm_c` 而不是失败。

## 关联

前序:#43199/#39488/#41773/#42264(均已关闭)、方向来源 #38756、问题单 #28042、依赖其 matcher 的 #42749。索引内同区域:RoPE/QK-norm 融合与 vLLM IR 迁移类卡片(pack 未提供具体编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
