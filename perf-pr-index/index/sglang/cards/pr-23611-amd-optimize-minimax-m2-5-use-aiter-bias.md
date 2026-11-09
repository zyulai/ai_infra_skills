# PR #23611: [AMD] Optimize MiniMax-M2.5 - use aiter biased_grouped_topk for sigmoid scoring in MoE routing

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yctseng0211 |
| 合并 | 2026-04-25 |
| 分类 | MoE(major) |
| Label | 无 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示仅改动 topk.py 的 fused_topk sigmoid 分支(+17/−7 左右) |
| 链接 | https://github.com/sgl-project/sglang/pull/23611 |

## 一句话结论

AMD 上 MiniMax-M2.5 等带 correction bias 的 sigmoid scoring MoE 改用 aiter biased_grouped_topk(ASM kernel),单次调用 ~9.3us→~6us,输出吞吐 +2.0%..+2.4%。

## 动机

对使用 sigmoid scoring + correction bias 的模型(如 MiniMax-M2.5),AMD GPU 上此前走 `sgl_kernel.topk_sigmoid`,其单次调用约 9.3us;aiter 的 `biased_grouped_topk`(ASM kernel)同类调用约 6us,每次调用可降低约 35% 的 MoE routing 开销。

## 优化点

在 `fused_topk` 的 `scoring_func == "sigmoid"` 分支上加平台判定:当 `_use_aiter` 且 `correction_bias is not None` 时改调 `aiter_biased_grouped_topk`,把 correction_bias cast 到 gating_output 的 dtype,以 `num_expert_group=1, topk_group=1`(即不做分组,只取 biased top-k)、`need_renorm=renormalize` 的配置运行;不满足条件(非 aiter 平台或无 correction bias)时保持原 `topk_sigmoid` 路径不变。改动只在 dispatch 层,不引入新 kernel。

## 关键改动

```text
python/sglang/srt/layers/moe/topk.py: fused_topk 的 sigmoid 分支在 AMD/aiter 且有
  correction_bias 时切换到 aiter_biased_grouped_topk(num_expert_group=1,
  topk_group=1),否则回落 topk_sigmoid
```

关键摘录(diff):

```python
if _use_aiter and correction_bias is not None:
    aiter_biased_grouped_topk(
        gating_output,
        correction_bias.to(dtype=gating_output.dtype),
        topk_weights, topk_ids,
        num_expert_group=1,
        topk_group=1,
        need_renorm=renormalize,
    )
else:
    topk_sigmoid(...)
```

## 性能声明

MI355X,MiniMax-M2.5 FP8,TP=4,ISL=8192,OSL=1024:输出吞吐 conc=64 时 +2.0%,conc=32 时 +2.4%;conc=4..128 各并发级别均无回退。Kernel 级:aiter kernel ~6us/call vs sgl_kernel 变体 ~9.3us/call,routing 开销每次调用降低约 35%。精度:GSM8K 93.4% vs baseline(topk_sigmoid)93.3%。

## 验证与回归风险

PR 正文仅给出 GSM8K 精度对照(93.3%→93.4%),无新增单元测试;diff 无测试文件。回归风险:分支条件严格(`_use_aiter` 且 `correction_bias is not None`),其他平台与无 bias 模型路径不变;`num_expert_group=1/topk_group=1` 假定 MiniMax-M2.5 式 routing 无分组语义,若其他带 bias 的 sigmoid 模型实际依赖分组 top-k,该假设需要另行确认;`need_renorm=renormalize` 保持原有 renormalize 语义。

## 关联

同系列:[AMD] Optimize MiniMax-M2.5 系列的 #23620(aiter decode FP8 KV cache 融合写入);同日合并、同作者、同模型。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
