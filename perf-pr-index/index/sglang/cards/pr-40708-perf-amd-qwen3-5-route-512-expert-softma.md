# PR #40708: [Perf] [AMD] qwen3.5: Route 512-expert softmax MoE routing through aiter topk_gating

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @omirosh |
| 合并 | 2026-11-01 |
| 分类 | Speculative Decoding (tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/40708 |

## 一句话结论

Qwen3.8 的 (512,10) softmax MoE 路由从 vendored `topkGatingSoftmax` 改走 aiter `topk_gating` 的 E=512 专用 kernel,8x MI355X 上输出 tok/s +1.5%(无 MTP)/+2.2%(MTP)。

## 动机

Qwen3.8 每层 92 层都要在 512 个专家里选 10 个;ROCm 上这一步占每 decode step 0.878 ms(4.8% 的 decode kernel 时间),而 B300 上只要 0.440 ms。`aiter.fused_moe.fused_topk` 的 ASM router 只覆盖 `(experts, topk)` ∈ {(128,4)...(640,8)},不含 (512,10),于是落到 vendored `vllm::moe::topkGatingSoftmax`。aiter 后来通过 #4460 增加 E=512 kernel 并支持 softmax 的 `need_renorm`(#5334),使路由换到专用 kernel 成为可能。2025 年 8 月 #28399 试过同样的接线但无收益;#39986 则把 Qwen3.5(hidden 4096)路由到 Triton kernel。

## 优化点

`topk.py` 新增 `_use_aiter_topk_gating_softmax()` 判定与 `_AITER_TOPK_GATING_MAX_ROWS = 4096` 行数上限:命中 gfx95 + AITER + hidden 8192(Qwen3.8)或 2560(Qwen3.8-Flash-Next)+ BF16 + 512 专家 + topk=10 + 连续 gating 输出 + 无 correction_bias/共享专家融合/packed_out 且 ≤4096 行时,softmax 分支改调 `aiter.topk_gating`(传 `need_renorm=renormalize`、`score_func="softmax"`);其余形状仍走 `aiter_fused_topk`。4096 行上限来自微基准:`topk_gating` 的 E=512 kernel 在 ≤1024 行 GPU 时间领先,2048/4096 行与 legacy 打平,5120 行起明显落后(其共享内存 kernel 输给 `topkGatingSoftmax`,16384 行 28.58 vs 18.56 us);设上限使 decode/verify/draft-extend/短 prefill 命中新 kernel,大批量 prefill 留在 legacy。新增回退开关 `SGLANG_ROCM_DISABLE_AITER_TOPK_GATING`(默认 off)。

## 关键改动

```text
python/sglang/srt/layers/moe/topk.py: _use_aiter_topk_gating_softmax 判定 + softmax 分支改调 aiter topk_gating
python/sglang/srt/environ.py: SGLANG_ROCM_DISABLE_AITER_TOPK_GATING 回退开关
test/registered/amd/test_qwen38_moe_softmax_topk_gating.py: 与 legacy launcher 数值一致性 + 派发边界测试
```

```python
+    return (
+        _use_aiter
+        and _use_aiter_topk_gating
+        and _is_gfx95
+        and hidden_states.shape[1] in (2560, 8192)
+        and gating_output.shape[1] == 512
+        and topk == 10
+        and gating_output.shape[0] <= _AITER_TOPK_GATING_MAX_ROWS
+        ...
+    )
```

## 性能声明

8x MI355X、TP8、Qwen3.8-2.4T-A95B-Quark-MXFP4,`chunked-prefill-size 16384, page-size 64`,12 个 serving shape(8k/1k、60k/600、128k/1k 多并发)输出 tok/s 几何均值:

- 本 PR:non-MTP 1.0148(+1.48%),NEXTN MTP(3 steps/eagle-topk 1/4 draft tokens)1.0222(+2.22%);对 merge-base(aiter pin `285579b62`)为 +1.27%/+2.22%。
- SGLang v0.5.20 默认 page size:无 MTP +1.03%,MTP +0.40%(review comment)。
- Qwen3.8-Flash-Next(reviewer 实测):输出 tok/s +2.17%,median TPOT 7.71→7.54 ms(-2.25%);accept length 高 0.7%,按每 accepted token 折算 -1.4%(17.84→17.59 ms)。
- 微基准(gfx950,E=512,topk=10,bf16,CUDA graph 回放):graph 模式下 `topk_gating` 1 行 4.74 vs legacy 9.26 us;5120 行起 legacy 反超。
- 与 #39986 Triton 路由端到端打平(相差 ≤0.25%,在 run-to-run spread 内);未测其他 512-专家模型、短 prompt prefill 端到端、gfx942、TP8/EP1 以外配置。

## 验证与回归风险

- 新测试覆盖:6 个行数档(含 1025 与 4096 边界)× 2 个 hidden size 与 legacy launcher 数值一致(BF16 tie 允许等 logit 专家换序),以及派发 envelope 的每个负例(hidden 4096、fp16、非连续、topk 8、correction_bias、fused shared experts、packed_out、>4096 行、开关关闭)。
- 精度:MI355X TP8 上 GSM8K 5-shot(0.978 上下)与 NIAH 65k/131k 基本持平。
- 回归风险:派发条件收得很窄,其他 512-专家模型不受影响;依赖 aiter #5334——没有它 `topk_gating` 退化为 8.0–9.2 us 的通用 kernel,几乎无收益。CI 显示 Extra 与 AMD ROCm 10 失败,Base 通过。

## 关联

- 相关:#39986(Qwen3.5 Triton 路由,hidden 4096,本 PR 与其互补)、#28399(早期尝试,无收益)、#45498(aiter pin)。上游:ROCm/aiter#4460、#5334。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
