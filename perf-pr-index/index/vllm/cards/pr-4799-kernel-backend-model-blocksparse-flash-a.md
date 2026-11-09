# PR #4799: [Kernel][Backend][Model] Blocksparse flash attention kernel and Phi-3-Small model

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @linxihui |
| 合并 | 2024-05-25 |
| 分类 | Attention / Kernel(tier: major) |
| Label | 无 |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 19 个文件因超长被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/4799 |

## 一句话结论

在 vLLM paged attention CUDA kernel 中加入 block-sparse 跳块支持,并配套 Triton prefill kernel 与 Phi-3-Small(8K/128K)模型接入,使稀疏注意力 head 按块跳过无关 KV 计算。

## 动机

Microsoft Phi-3-Small-8K / Phi-3-Small-128K 使用 blocksparse flash attention,每个 attention head 有自己的混合稀疏模式(local + strided),vLLM 当时的 paged attention 与后端无法表达这种按块跳过的注意力,也就无法运行该模型或享受稀疏带来的 KV 访问量下降。本 PR 是 Microsoft GenAI 与 vLLM 的合作改动(linxihui、beagleski 与 zhuohan123、simon-mo、youkaichao),把 block-sparse 支持做进 CUDA paged attention kernel,并为 prefill 提供 Triton kernel。

## 优化点

机制上,给 `paged_attention_kernel` 增加编译期模板参数 `IS_BLOCK_SPARSE` 与四个稀疏参数(`blocksparse_local_blocks`、`blocksparse_vert_stride`、`blocksparse_block_size`、`blocksparse_head_sliding_step`,外加 `tp_rank`)。开启后(`blocksparse_vert_stride > 1`)在 K 和 V 的块循环里,对每个 KV block 先判定其 block-sparse block id 是否属于 remote(按 vert_stride 取模命中)或 local(位于 query block 前 local_blocks 内);既非 remote 也非 local 的块直接跳过:K 侧把对应 token logits 置为 `-FLT_MAX` 以免污染 softmax sumexp 归一化,V 侧直接 `continue`。稀疏偏移支持按 q head 或 kv head 滑动(`blocksparse_head_sliding_step` 正负号区分),从而允许每个 head 不同的稀疏 pattern。launcher 层用 `CALL_Vx_LAUNCHER_SPARSITY` 宏按 `is_block_sparse` 分发到 true/false 两个模板实例,非稀疏路径走原逻辑,无额外开销。

## 关键改动

```text
csrc/attention/attention_kernels.cu: paged_attention v1/v2 kernel 增加 IS_BLOCK_SPARSE 模板分支与跳块逻辑
vllm/attention/ops/(Triton prefill kernel 与 interface 文件在 diff 中被略去,见 PR 全文): blocksparse triton prefill kernel 与 LocalStridedBlockSparseAttn 接口
csrc/cpu/attention.cpp: CPU backend 增加 TORCH_CHECK 拒绝 blocksparse
tests/kernels/test_blocksparse_attention.py: 新增 CUDA kernel 与参考实现的数值一致性测试
docs/source/models/supported_models.rst: 登记 Phi3SmallForCausalLM
```

关键片段(K 侧跳块):

```cuda
if constexpr (IS_BLOCK_SPARSE) {
  const int k_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
  const bool is_remote =
      ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
  const bool is_local =
      (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
  if (!is_remote && !is_local) {
    // logits 置 -FLT_MAX,跳过该块的 K 计算
    continue;
  }
}
```

## 性能声明

PR 正文未提供 benchmark 数字,也未注明测试条件(perf claims 检测为 none)。收益机制是稀疏 pattern 下跳过未参与的 KV block 计算,但 PR 未量化。

## 验证与回归风险

新增 `tests/kernels/test_blocksparse_attention.py`:参数化覆盖 v1/v2、head_size 64/112、block_size 16/32、alibi、fp8 KV cache、vert_stride=8、local_blocks=16、blocksparse_block_size=64、head sliding step 0/2/-1 等,与带稀疏 mask 的参考实现对比(allclose,fp8 用放宽容差 1e-2/1e-5)。回归风险:`paged_attention` C++ 接口签名变更波及 CUDA/CPU/C HIP 各 backend(CPU 侧显式 TORCH_CHECK 拒绝 blocksparse);非稀疏路径通过模板 false 实例保持原行为,但所有调用点需要传新参数。

## 关联

索引内同区域(Attention / Kernel)卡片:#12348(ROCm paged attention kernel 优化)、#18218(Triton attention kernel);更直接相关的是后续稀疏/长序列 attention 演进,如 #53781(HiSparse sparse-MLA decode hot-buffering,同分类)。Phi-3-Small 模型接入与 kernel 改动在本 PR 同时落地。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
