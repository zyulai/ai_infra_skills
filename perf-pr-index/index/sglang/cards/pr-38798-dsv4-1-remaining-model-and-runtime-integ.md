# PR #38798: dsv4.1: remaining model and runtime integration

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-09-18 |
| 分类 | Other |
| Label | documentation, high priority, quant, dependencies, deepseek, hicache, npu, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, unified-radix-cache, memory-pool |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与文件数统计);diff 因超长被截断,另有 97 个文件未展示;可见部分涉及 7 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/38798 |

## 一句话结论

DeepSeek V4.1 模型与 runtime 集成收尾:补齐 SM100 小批量 decode attention、c2 pair-pooling、fp4 indexer、router gate 等 dsv4 kernel。

## 动机

为 SGLang 增加 DeepSeek V4.1(dsv4.1)支持,本 PR 是系列中的"剩余模型与 runtime 集成"部分。正文很简短:作者说明该分支处于重构中、不稳定,建议使用指定 commit 的镜像;并声明依赖同系列 PR #39666(`dsv4.1-engram`)。原始 restack 中包含的 standalone NVLink benchmark 已归档到 gist 并从分支移除,生产代码与测试不受影响。

## 优化点

从可见 diff 看,主要是 dsv4 专用 kernel 的补齐与数值对齐:

- `c2_decode_pool.py`(新增):ratio-2 decode pair-pooling 的 fused Triton kernel,声称与 torch `pool_pairs` 路径 bitwise 一致;支持 `CompressStatePool` 环形 buffer(ring_size)或每请求单行两种状态布局;用 `libdevice.exp`/`libdevice.div_rn` 与 `t0 + 0.0` 防 FMA 收缩来精确复现 torch 的舍入点,因为舍入差异会改变 indexer 的 top-k 选择。
- `decode_attention_sm100.py` + `decode_attention_sm100_gluon.py`(新增):面向 SM100(Blackwell tcgen05)的小批量(b≤8)paged attention,16 heads × head_dim 512,V4 layout(每 token 448 字节 fp8 nope + 64 bf16 rope,附 ue8m0 scale);用 Triton Gluon 写 native tcgen05 MMA,SWAP_AB,并做 fp32 低概率补偿(COMPENSATE,`prob - p_hi` 二次 MMA)降低 bf16 量化误差;attention sink 恰好并入一次;调用方负责 inverse RoPE。
- `fp4_indexer.py`:新增 `fp4_index_logits_decode`,直接从 fp4 index-K pool(e2m1 payload + e8m0 block scale)计算 decode index logits,舍入点与 torch reference 对齐。
- `dsv4_attn_metadata_kernels.py`:SWA page indices 构建支持 `swa_replay_start`(窗口绝对下界 floor);新增 `late_layer_tail_layout`,为 prefill extend 的尾部行构造 layout,三个长度向量一次 H2D 拷贝,launch 数不随 batch 增长。
- `moe_fused_gate.py`:router Triton kernel 扩展——按 `input_ids` 选择的 token bias(`bias_alt`)、`num_token_non_padded` padding 掩码、`packed_out`(idx<<16|bf16 weight 的 int32 打包)、`sqrtsoftplus_log1p`(V4.1 数值)、`renormalize_epsilon`,并调整 NaN 排序以匹配 `torch.topk`。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsv4/c2_decode_pool.py: 新增,fused ratio-2 decode pair-pooling
python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100.py: 新增,SM100 小批量 paged attention 入口
python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100_gluon.py: 新增,tcgen05 native MMA 核心
python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py: 新增 fp4_index_logits_decode
python/sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py: swa_replay_start 与 late_layer_tail_layout
python/sglang/kernels/ops/moe/moe_fused_gate.py: router kernel 扩展 V4.1 语义
```

```python
if COMPENSATE:
    p_lo = (prob - p_hi.to(gl.float32)).to(gl.bfloat16)
    p_smem.store(p_lo)
    fence_async_shared()
    tcgen05_mma(kv_smem.permute((1, 0)), p_smem, out_tmem, use_acc=True)
```

## 性能声明

PR 正文未提供 benchmark 数字(regex 亦未检出);仅说明 standalone NVLink benchmark 已归档至作者 gist 并从分支移除。SM100 attention 限定 MAX_BATCH=8、16 heads、bf16、head_dim 512 的使用条件在代码中注明。

## 验证与回归风险

PR 正文未注明新增测试;CI 状态显示 Base/Extra/AMD ROCm 10 测试均为未运行(no_entry_sign)。diff 有 97 个文件未在 pack 中展示,改动面覆盖 deepseek、quant、hicache、npu、memory-pool、unified-radix-cache 等多个子系统(label 可佐证)。数值敏感路径(c2 pool、fp4 indexer、router gate)在代码注释中要求与 torch reference bitwise/舍入一致,任何精简都可能改变 top-k 选择。依赖 #39666。

## 关联

依赖:PR #39666(dsv4.1-engram)。同系列 dsv4.1 stack 的其他 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
