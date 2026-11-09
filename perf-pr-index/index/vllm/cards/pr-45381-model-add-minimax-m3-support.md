# PR #45381: [Model] Add MiniMax M3 support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @youkaichao |
| 合并 | 2026-06-15 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, new-model, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, gpt-oss, nvidia, rust |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 内容可用,列出 4 个源码文件,另有 105 个文件因超长被丢弃) |
| 链接 | https://github.com/vllm-project/vllm/pull/45381 |

## 一句话结论

新增 MiniMax M3 全栈支持(模型实现、MTP、稀疏注意力、fused qknorm+RoPE+KV insert kernel、router GEMM 形状、Rust 推理/工具解析器)。

## 动机

实现 issue #45360:MiniMax M3 是新模型,vLLM 完全不支持。该模型涉及若干与既有实现不同的组件(带 alpha/beta 参数的 gated activation、稀疏注意力 + indexer K cache、MTP 投机解码、不同的 router GEMM 形状),需要模型实现之外的 kernel 与量化路径配套。

## 优化点

PR body 概括为三块:(1) 模型栈——config、processors、model registry、AMD/NVIDIA 模型实现、MTP(投机解码)、sparse attention(含 warmup 路径);(2) 前端——MiniMax M3 的 reasoning parser 与 tool parser,含 Rust frontend 注册(vllm-reasoning-parser/vllm-tool-parser/vllm-chat crate)与 Python 侧 wrapper;(3) kernel/量化——fused kernel、量化路径、router GEMM 形状支持与针对性测试。diff 可见的性能相关机制:(a) gated activation kernel 泛化:`silu_and_mul_clamp` 扩展为 `act(x,alpha) * (y + beta)` 形式(alpha 在激活内、beta 加在另一半上),默认 alpha=1/beta=0 精确还原 SwiGLU/GeGLU,为 M3 的激活变体在单一 fused kernel 内支持,不增加 kernel 数;(b) fp32 router GEMM 按 (num_experts, hidden_dim) 形状对显式实例化并 dispatch:M2/M2.5 的 (256,3072) 与 M3 的 (128,6144),避免运行时形状分支;(c) 新增 635 行 horizontally-fused 的 M3 attention 预处理 kernel:qk-norm + RoPE + KV/indexer-K cache 插入一次完成,支持 PDL 触发与 sparse serving 的 slot 写入(跳过 padded/未调度 token)。

## 关键改动

```text
csrc/libtorch_stable/activation_kernels.cu: gated activation 增加 alpha/beta 参数化(silu_and_mul_clamp 扩展)
csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu: 新增 fused qk-norm + RoPE + KV/indexer cache insert kernel
csrc/libtorch_stable/fp32_router_gemm.cu, fp32_router_gemm_entry.cu: router GEMM 支持 (128,6144) M3 形状并 dispatch
其余约 105 个文件(模型实现、MTP、稀疏注意力、parsers、注册表等)在 pack 中因超长被丢弃
```

关键代码(fused kernel 的 cache 插入分支):

```cpp
if constexpr (kInsertKV) {
  int64_t const sm = (isK || isV)
      ? slot_mapping[tokenIdx]
      : (isIK ? index_slot_mapping[tokenIdx] : -1);
  if (sm >= 0) {  // skip padded / unscheduled tokens
    if (isIK) { /* indexer K cache */ }
    else if (isK || isV) { /* kv_cache [num_blocks, 2, block_size, nkv, head_dim] */ }
  }
}
```

## 性能声明

PR 正文未给出 benchmark 数字(regex 未检出性能声明),仅在 Summary 中列出配套优化(fused kernel、router GEMM 形状等)。fused kernel 对比原多 kernel 路径的加速未注明。PR 未注明测试条件。

## 验证与回归风险

测试:`cargo fmt --check` 与 `cargo test -p vllm-reasoning-parser -p vllm-tool-parser -p vllm-chat`;正文称含 targeted tests(具体 Python 测试文件在丢弃的 diff 中,PR 未注明)。风险面:activation kernel 是所有模型共享路径(alpha/beta 为新增参数,默认值保持原行为,但 kernel 模板签名变化影响 libtorch_stable 构建面);fp32_router_gemm 的形状 dispatch 收紧为显式枚举的 (E,H) 对,其他形状需走回退;fused 预处理 kernel 涉及 NHD/HND 物理布局 stride 与 PDL,是新引入的专用路径。

## 关联

同区域:MiniMax M2/M2.5 的 router GEMM(本 PR 在其基础上扩展 M3 形状);speculative config 中已有 "minimax_m3_mtp" 方法(#12755 的 MTP 体系后续演进);label 中的 gpt-oss 关联同族架构复用;同分类 Model-specific Fast Path 其他 PR 见索引。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
