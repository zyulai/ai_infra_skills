# PR #14423: [NPU] perf update with kvcache nz & w4a8 quant

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @liupeng374 |
| 合并 | 2025-12-13 |
| 分类 | Quantization(major) |
| Label | deepseek, npu, run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/14423 |

## 一句话结论

Ascend 上启用 NZ 格式 KV cache 加速 FIA 算子(TPOT 约省 2 ms),MoE w4a8 改 per-channel 量化,prefill 优化提升超 10%。

## 动机

四项并行:KV cache 以 NZ 格式存放可加速 FIA(fused infer attention)算子;MoE w4a8 之前按 group(256)量化,padding 与 scale 处理开销大;prefill 的 MHA 预处理(rope/kv-cache 写入)可由 `npu_interleave_rope` 等融合算子加速;`num_token_non_padded_cpu` 存在重复调整的 bug。

## 优化点

开启方式为 `SGLANG_USE_FIA_NZ=1`,且必须与 mlapo 的 `SGLANG_NPU_USE_MLAPO=1` 同开(`is_fia_nz()` 内 assert)。NZ 路径:decode/MTP/FIA 的 KV cache 视图从 `(blocks, page, heads*dim)` 改为 `(blocks, 1, heads*dim//16, page_size, 16)` 的 NZ 分块,`mla_preprocess` 的 `cache_mode` 由 `krope_ctkv`/`PA_BNSD` 切换为 `nzcache`/`PA_NZ`,decode graph 的 input_layout 相应由 BNSD 改为 BSND。MoE w4a8:`NPUW4A8Int4DynamicMoEMethod` 的 `group_size` 256→0 走 per-channel 权重量化,grouped 的 `*_scale_second`/`*_offset_second` 参数仅在非 per-channel 时创建,per-channel 的 scale 打包为 uint32 视角的 uint64 张量返回。MHA prefill:`forward_mha_prepare_npu` 改用 `npu_interleave_rope` 处理 q_pe、`npu_kv_rmsnorm_rope_cache` 一步完成 kv rmsnorm+rope+cache 写入(PA_NZ/PA_BNSD 双模式),核心 bmm 换 `torch.ops.npu.batch_matmul_transpose`。bugfix:删除 `adjust_num_token_non_padded_for_attn_tp` 中对 `num_token_non_padded_cpu` 的二次计算。

## 关键改动

```text
python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py: FIA NZ 的 KV cache reshape 与 BSND layout
python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py: is_fia_nz() 门控、mla_preprocess 的 nzcache 模式
python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py: w4a8 per-channel(group_size=0)support
python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py: npu_interleave_rope + npu_kv_rmsnorm_rope_cache 融合预处理
python/sglang/srt/model_executor/forward_batch_info.py: num_token_non_padded_cpu 重复计算修复
```

## 性能声明

PR 文本:FIANZ 可将 TPOT 提速约 2 ms;prefill 优化可提升性能 10% 以上。未注明模型、硬件型号、并发与 workload 等测试条件;精度证据为一张截图(素材中不可读取文字数字)。

## 验证与回归风险

PR 勾选了格式项,未勾选单元测试、文档与精度/性能 benchmark 项;精度结果仅有截图。风险面:多条路径由环境变量门控且要求两个开关同开,误配会被 assert 拦截;NZ reshape 对 `heads*dim` 可被 16 整除有隐含要求;w4a8 per-channel 的 scale 打包(uint32→int64 视角)与 checkpoint 格式耦合;`get_cos_sin_cache` 在 `rotary_embedding.py` 引入的全量 cos/sin 缓存(`cos_cached_total` 等)增加常驻显存;`forward_batch_info.py` 的删除会影响 attn TP 场景下 num_token_non_padded 的语义(作者视为 bugfix)。

## 关联

无(素材未列出前序/后续 PR 编号;`mla_preprocess` 中留有 sgl-kernel-npu issue #78 的 dummy-input TODO)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
