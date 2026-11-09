# PR #46263: [Perf] [NVIDIA] deepseek-v4.1: Default aggregated servers to MegaAttention only when each rank holds 64 query heads

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @weireweire |
| 合并 | 2026-10-26 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, deepseek |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/46263 |

## 一句话结论

聚合部署且 attention TP>1 时,DeepSeek-V4.1 的 `--dsv4-attn-backend auto` 改默认 `flashmla`,避免 MegaAttention 64-head tile 填充浪费,GB300 上交互性 P90 最高 +27%。

## 动机

自 #43922 起,SM100/SM103 + fp8_e4m3 KV cache 下 `auto` 一律解析为 `flashmla_mega`,不区分 attention TP 宽度。MegaAttention 恒跑 64-head tile(`kernel_num_heads = 64 if n_local_heads <= 64 else 128`),缺失头由 `pack_query` 零填充;DeepSeek-V4.1-Flash 有 64 个 query head,TP2(每 rank 32 头)会多做 2 倍、TP4(每 rank 16 头)多做 4 倍 attention 计算。而 `flashmla` 路径有面向 16/32 头的小批量 SM100 kernel(`can_use_swapab_attention`),在这些宽度下更优。TP4 上 MegaAttention 使每步 decode 多花约 1.1 ms(mega kernel 约 1.45 ms 加 fp8 `wo_a` GEMM 约 0.29 ms,减去被丢弃的 FlashMLA combine/wo_a 融合/Q-norm-RoPE/quant 约 0.85 ms)。

## 优化点

改动是纯后端选择逻辑:对聚合服务器(`--disaggregation-mode null`),`_default_dsv4_attention_backend` 在 attention 按张量并行分片(`attn_tp_size > 1`,经新辅助函数 `_attention_is_tensor_parallel` 由 tp/attn_cp/attn_dp 推导)时返回 `flashmla`,与 `_kernel_num_heads` 仅在 attn_tp>1 时 pad 到 64 头 tile 的行为对齐。TP1、DP attention(attn TP 1)与 prefill CP 不分片,仍解析为 `flashmla_mega`(prefill CP 直接视为 attn TP 1,因 `validate_deepseek_v4_cp` 在本默认之后才设 `attn_cp_size`)。显式 `--dsv4-attn-backend` 不受影响。PD 分离角色保持现默认(#45075):无论 attention TP 宽度,前后角色都用 `flashmla_mega`,因为两个后端 KV 布局不同(`flashmla` 存 V4、`flashmla_mega` 强制 V4.1),按头数 gate 会让 TP prefill 与 DP-attention decode 解析不一致,在 KV-layout 握手时失败(GB300 上曾出现 `KV layout mismatch` 首请求报错)。

## 关键改动

```text
python/sglang/srt/arg_groups/deepseek_v4_hook.py: 新增 _attention_is_tensor_parallel,聚合 + attn_tp>1 时默认改 flashmla
python/sglang/srt/arg_groups/fields/exec_.py: 更新 --dsv4-attn-backend help 文本
docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx: PD 角色默认说明改为"无论 attention TP 宽度"
```

```python
+        or (cfg.disaggregation_mode == "null" and _attention_is_tensor_parallel(cfg))
```

## 性能声明

DeepSeek-V4.1-Flash FP4,GB300,aggregated + Dynamo 前端,DSpark(MTP),agentic 多轮 coding trace 回放,60 分钟测量窗口,单 server 单行:

| 点位 | backend | 总 tok/s/GPU | 交互性 P50/P90 | TTFT P90 |
|---|---|---|---|---|
| TP4 c4 | auto(mega) / flashmla | 9,881 / 9,736 | 537.6/465.8 vs 683.7/590.8 | 0.288 vs 0.254 s |
| TP4 c16 | auto(mega) / flashmla | 29,051 / 29,371 | 473.8/403.7 vs 593.2/468.6 | 1.888 vs 1.401 s |
| TP2 EP2 c16 | auto(mega) / flashmla | 57,766 / 57,999 | 376.2/309.0 vs 414.8/315.4 | 1.310 vs 1.176 s |

P90 交互性分别 +27%、+16%、+2%(TP2 P50 +10%),吞吐持平。TP2 c32 与 TP4 c64 大批量下 P90 分别 +3.1%/+5.0%,固定批量 TP4 EP1 下 `flashmla` decode step 在 57–256 running 全部更快(-1.9% 到 -4.1%);MegaAttention prefill 略快(C=128 中位 TTFT 4.54 vs 5.22 s),端到端 bench 吞吐差 <1.7%。GSM8K 1319 题:sgl-eval 0.1.3、聚合 TP4 EP1、DSpark 真实 acceptance,flashmla 96.82% vs mega 96.59%(差距 3 题,在一个标准误内)。

## 验证与回归风险

- 改动仅选择既有后端,无 kernel/模型代码变化;用 `ServerArgs.resolve_once` 验证了 TP2/TP4/attn-dp/DP-attention/prefill CP/PD 各组合的解析结果符合预期表。
- 副作用:聚合 TP2/TP4 改存 V4 KV 布局而非 V4.1,`--mem-fraction-static 0.8` 下 KV pool 66.86M vs 67.19M tokens(-0.5%);与 V4.1 布局 server 共享的 HiCache store 会看到不同布局(store key 不含布局)。`--enable-cp-decode-attn-tp` 未测量,仍解析 mega。GB300 TP2 CI 套件(`--attn-dp-size 1`)从此跑 `flashmla`。CI 显示三条(含 Base)均为失败。

## 关联

- 前序:#43922(auto 默认解析为 flashmla_mega)、#45075(PD 角色默认)。
- 同区域:DeepSeek V4.1 attention 后端选择系列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
