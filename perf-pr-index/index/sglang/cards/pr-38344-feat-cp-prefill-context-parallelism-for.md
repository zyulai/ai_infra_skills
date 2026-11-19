# PR #38344: [Feat] CP: Prefill context parallelism for Kimi K3

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Shunkangz |
| 合并 | 2026-11-17 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 摘录含 7 个文件、其中 1 个测试文件删除) |
| 链接 | https://github.com/sgl-project/sglang/pull/38344 |

## 一句话结论

为 Kimi K3 打开 prefill context parallelism(CP-TP group sharing,zigzag):只有 MLA 层看 token 分片,KDA/MoE 全序列跑;Phase 0 默认实现把 8x B200 上 CP8 的开销从 +45%/+87% 压到 +0.3%/+2.4%,Phase 0 + Phase 1(`SGLANG_K3_CP_SP_RESIDUAL=1`)反超 plain TP8 达 −16.6%/−15.4%。

## 动机

K3 = 93 层:69 KDA(chunked delta rule 线性注意力)+ 24 MLA(softmax attention);除 layer 0 外每层接 latent MoE(896 NVFP4 experts,top-16)。分支的 prefill CP 是"full-sequence CP":只有 MLA 层看 token shard(zigzag,平衡因果负载),KDA 与 MoE 在每个 rank 上跑全序列并走常规 TP8 集合通信。CP8 意味着 A = tp/cp = 1:每个 rank 为自己的 N/8 行 query 持有全部 96 个 MLA 头,MLA o_proj 的 all-reduce 消失。结构上,CP 不能降低每 rank 的 attention/KDA/expert FLOPs(heads × queries 守恒),也不触碰 KDA/MoE 集合通信,只能拿 MLA all-reduce 换 K/V 展开复制的代价——要让 CP8 赢,必须改两件事。

## 优化点

- `cp_tp_group_sharing_hook.py`:新增 `_KIMI_LINEAR_MODELS`(KimiLinearForCausalLM / KimiK3LinearForCausalLM / KimiK3ForConditionalGeneration)并入 `_SUPPORTED_MODELS`;要求 `--attn-cp-size == --tp-size`;K3 要求 prefill attention backend fa3/fa4(fa4 还要求 SM100/SM110 且 kv_lora_rank=512、qk_rope_head_dim=64,用于 absorbed MLA);kimi linear 额外拒绝 attn-dp-size > 1 与任何非 null 分解模式,并沿用 Qwen4Exp 的 unsupported 列表(moe-dp-size > 1、enable-mixed-chunk、disaggregation-mode decode 等);prefill CUDA graph 置 DISABLED(序列集合通信仍有 batch 相关尺寸)。
- `models/kimi_k3.py`:删除"Kimi-K3 does not support --enable-prefill-cp yet"的拒绝;`_stage_facts` 给 `declare_attn` 传 `in_token_order`(KDA 层且 zigzag 时——interleave KDA 在 attention 内部恢复序列序,zigzag 把该传输留给 stage 边界)。
- `models/kimi_linear.py`:forward 里接入 `cp_split_before_forward` / `cp_gather_after_forward`(含 aux_hidden_states);`stage_facts` 的 `declare_attn` 按 `linear_attn_parallel_group()` 给 SumGroup.TP/ATTN_TP 与 in_token_order;qkv_proj 的 parallel_group 参数化。
- 测试:新增 `TestKimiLinearPrefillCP`(Kimi-Linear-48B-A3B,TP2 + prefill CP zigzag,GSM8K 200 题、阈值 0.88);删除 `test_platform_prefill_cp_deprecation.py`。

## 关键改动

```text
python/sglang/srt/arg_groups/cp_tp_group_sharing_hook.py:
+_KIMI_LINEAR_MODELS: set[str] = {
+    "KimiLinearForCausalLM",
+    "KimiK3LinearForCausalLM",
+    "KimiK3ForConditionalGeneration",
+}
python/sglang/srt/arg_groups/model_overrides/kimi_k3.py:
-    if cfg.enable_prefill_cp:
-        raise ValueError(
-            "Kimi-K3 does not support --enable-prefill-cp yet (Kimi-K3.1 does)."
-        )
python/sglang/srt/models/kimi_k3.py:
+        declare_attn(
+            **attn_ops,
+            # Interleave KDA restores sequence order inside the attention;
+            # zigzag delegates that transport to the stage boundary.
+            in_token_order=config.is_kda_layer(layer_idx)
+            and get_parallel().cp_strategy == "zigzag",
+        ),
python/sglang/srt/models/kimi_linear.py:
+        if cp_active:
+            hidden_states, positions = cp_split_before_forward(hidden_states, positions, forward_batch)
```

## 性能声明

(数字全部照抄 PR 正文,未独立核实。8x B200,real-weight 8 层 Kimi-K3-NVFP4 proxy(6 KDA + 2 MLA 层,7 MoE blocks),单请求,`--attention-backend fa4`,flashinfer_trtllm MoE runner,median of 5 forwards。)

| configuration | 32K prefill | 64K prefill |
|---|---|---|
| TP8 (baseline) | 121.7 ms | 244.5 ms |
| TP8 + CP8, before this PR (branch default) | 177.0 ms (+45%) | 457.2 ms (+87%) |
| TP8 + CP8, before this PR, expanded-MHA opt-in | 124.2 ms (+2.1%) | 254.4 ms (+4.0%) |
| TP8 + CP8, Phase 0 (this PR, default) | 122.1 ms (+0.3%) | 250.4 ms (+2.4%) |
| TP8 + CP8, Phase 0 + Phase 1 (SGLANG_K3_CP_SP_RESIDUAL=1) | 101.5 ms (−16.6%) | 207.0 ms (−15.4%) |

正文声明:"This PR makes Kimi K3 prefill with context parallelism (CP) faster than plain TP8 on 8x B200"(对应 Phase 0 + Phase 1 行)。

## 验证与回归风险

Accuracy Tests 小节在正文中为空(仅模板占位,本 PR 未提供独立精度数据);回归依赖新增的 `TestKimiLinearPrefillCP`(GSM8K 200 题、阈值 0.88、TP2 + `--enable-prefill-cp --attn-cp-size 2 --cp-strategy zigzag`),并将 test_kimi_linear_models.py 的 CI est_time 从 276 调到 438。CI 状态:Base 失败(x),Extra 跳过(未加 label),AMD 运行中(hourglass)。回归风险:(1) CP-TP sharing 的组合约束多:kimi linear 拒绝 attn-dp > 1、任何分解模式、moe-dp-size > 1、enable-mixed-chunk、disaggregation-mode decode;fa4 后端还受 SM100/SM110 与 kv_lora_rank/qk_rope_head_dim 硬编码约束;(2) 该路径强制 prefill CUDA graph DISABLED;(3) KDA zigzag 的 in_token_order 正确性依赖 stage 边界的序列传输;(4) 删除了平台 prefill CP 弃用策略的整套测试文件,平台策略覆盖减少;(5) 默认只含 Phase 0(仍在 +0.3%/+2.4% 开销),Phase 1 需显式 `SGLANG_K3_CP_SP_RESIDUAL=1`。

## 关联

同模型:PR #50842(KDA DWDP,其 Kimi-K3.1 对比表用 TP8 + `--enable-prefill-cp` 8,即本 PR 打开的 CP-TP group sharing 路径)。`_SUPPORTED_MODELS` 中已有 Glm5NextForConditionalGeneration、Qwen4ExpForConditionalGeneration、KimiK31ForConditionalGeneration;`_INTERLEAVE_MODELS`(DSA 层 interleave CP)不含 Kimi,K3 用 zigzag。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
