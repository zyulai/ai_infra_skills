# PR #41533: [ROCm] Use the fused MLA absorb + RoPE + KV-write kernel for decode-sized forward modes only

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jiaryang |
| 合并 | 2026-10-02 |
| 分类 | Attention / Kernel(tier: major) |
| Label | amd, run-ci |
| Diff | 素材未提供 diffstat |
| 链接 | https://github.com/sgl-project/sglang/pull/41533 |

## 一句话结论

把 gfx950 融合 MLA kernel(#38340)限定在 decode 尺寸的 forward mode,EXTEND/MIXED/SPLIT_PREFILL 回退两 kernel 路径,MI355X 上 8192:1024 吞吐 geomean +4.2%、TTFT +15.3%。

## 动机

#38340 引入的 `fused_fp8_bmm_rope_cat_and_cache_mla` 把 absorbed q BMM 与 RoPE/cat/KV-write 合成一次 launch,在 decode 与 verify 尺寸下是明显收益(两个独立 kernel 都填不满 CU);但在 prefill 形状(8k tokens × 4)下,main 上各自调优的两个分离 kernel 反而更快。即融合 kernel 并非在所有 forward mode 下都是最优,需要按模式区分。

## 优化点

在 `forward_mla_rocm.py` 的 `forward_absorb_rocm_prepare()` 中,给融合 kernel 的启用条件增加 `not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()` 门控:DECODE、TARGET_VERIFY、DRAFT_EXTEND_V2 继续走融合 kernel,EXTEND、MIXED、SPLIT_PREFILL 改回 main 已有的两 kernel 路径(absorb BMM + RoPE/KV-write)。特意不用 `is_extend()` 判断,因为它包含 TARGET_VERIFY——该模式形状上是 extend 但尺寸上是 decode。无新数值路径;CUDA graph 只对 DECODE 和 TARGET_VERIFY 捕获,捕获的图仍含融合 kernel。

## 关键改动

```text
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py: 唯一改动文件,给 fuse_bmm_rope_cache 判定加 forward mode 门控
```

```python
# The fused kernel wins at decode-sized batches (decode, target verify,
# draft extend); prefill shapes run faster as the two separate launches.
fuse_bmm_rope_cache = (
    not q_replicate_active
    and not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
    and _can_fuse_bmm_rope_cat_and_cache(self)
)
```

## 性能声明

MI355X TP4,GLM-5.2-MXFP4,fp8_e4m3 KV cache,`--chunked-prefill-size 16384`、dsa triton 前后端、aiter allreduce fusion;`bench_serving` 随机数据集,ratio 0.8,每点单次。No MTP:8192:1024 吞吐 geomean +4.2%(conc 64 单点 +5.4%)、TTFT geomean +15.3%(conc 64 单点 +48.8%)、TPOT +3.5%;70000:300 吞吐 geomean +6.5%;1024:1024 各项约持平(99.5–100.7%)。MTP(EAGLE):8192:1024 吞吐 geomean +3.5%,70000:300 +7.1%。Profile:8192:1024 c4 下 TP rank 0 的 EXTEND kernel 时间 1531.6→1413.4 ms,融合 kernel 的 166.1 ms 被 absorb BMM(32.8 ms)与 RoPE/KV-write(14.1 ms)替代。精度:GSM8K 5-shot × 3 runs,no MTP 0.938→0.929,MTP 0.932→0.936。

## 验证与回归风险

未新增单测;依赖 AMD CI(AMD ROCm 10)与上述 GSM8K/bench_serving 手动验证。风险面:decode 尺寸路径(DECODE/TARGET_VERIFY/DRAFT_EXTEND_V2)不变仍走融合 kernel;prefill 模式回退的是 main 既有两 kernel 路径,无新数值路径;no MTP 的 GSM8K 有 −0.9 个百分点的波动(0.938→0.929),MTP 侧略升。影响范围限于走 `forward_mla_rocm` 的 DeepSeek 系 MLA 模型在 ROCm/gfx950 上的行为。

## 关联

前序:#38340(引入 `fused_fp8_bmm_rope_cat_and_cache_mla` 融合 kernel,本 PR 是其适用范围的修正)。同区域其他 PR 未在素材中提及。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
