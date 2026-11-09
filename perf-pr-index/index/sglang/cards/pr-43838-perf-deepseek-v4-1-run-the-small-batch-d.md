# PR #43838: [Perf] DeepSeek-V4.1: Run the small-batch decode fusions at TP2

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @weireweire |
| 合并 | 2026-10-20 |
| 分类 | Model-specific Fast Path |
| Label | deepseek, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/43838 |

## 一句话结论

把 DeepSeek-V4.1-Flash 小批量 decode 三类融合(AR+mHC、fused WO-A、swap-AB attention)从 TP4 放宽到 TP2,GB300 TP2 部署 c1 P90 提速 +19.9%。

## 动机

DeepSeek-V4.1-Flash 的小批量 decode 融合(融合 all-reduce + mHC epilogue、融合 inverse-RoPE + WO-A + MXFP8 kernel、SM100 swap-AB decode attention)此前被 gate 在 TP4 形状(tp_size==4、2 个 local o_groups、16 local heads),TP2 decode 在所有 batch size 都走 unfused 路径;而 TP2 才是该模型在 GB300 上中等并发下更好的延迟/吞吐点。

## 优化点

三类融合分别解除 TP4 限制:fused all-reduce + mHC 直接准入 tp_size 2(kernel 已按 world size 模板化,新增 `fused_mhc_tp_ok(tp_size) in (2,4)` 统一判定,替换 `_can_fuse_attn_mhc`/`_can_fuse_ffn_mhc`/预编译注册处的 `== 4` 判断);fused WO-A 把 local group 数(TP4 为 2、TP2 为 4)做成 `WoATrait<kMaxTokens, kGroups>` 模板参数,输出列数改为 `kGroups * 1024`,JIT wrapper 名带 groups;swap-AB decode attention 把 TP2 的 32 个 local heads 作为两个 16-head group 在 partial kernel 的 grid z 轴上跑(gluon kernel 新增 `HG` constexpr,query/输出/scale 按 `h0 + h` 偏移读写)。所有 gate 检查的是形状(16/32 local heads、2/4 local o_groups、tp 2/4)而非 TP size 本身,具备这些 local 形状的部署都会走这些路径;目前仅验证过 TP2 与 TP4。

## 关键改动

```text
python/sglang/kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh: WoATrait 增加 kGroups 模板参数
python/sglang/kernels/ops/attention/dsv4/wo_a.py: GROUPS=(2,4) 与 can_use_fused_wo_a
python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100*.py: 32 heads 拆两个 16-head group(HG)
python/sglang/srt/models/deepseek_v4_mhc.py: fused_mhc_tp_ok(2,4)
python/sglang/srt/models/deepseek_v4.py / deepseek_v4_backend.py: gate 与 head 数适配
```

```python
_FUSED_MHC_TP_SIZES = (2, 4)

def fused_mhc_tp_ok(tp_size: int) -> bool:
    return tp_size in _FUSED_MHC_TP_SIZES
```

## 性能声明

GB300 x2 GPU,DeepSeek-V4.1-Flash TP2,AgentX trace replay(semianalysis cc-traces weka,每点 1 小时,MTP 接受长度钉在 3.51),P90 tok/s/user @ tok/s/GPU:conc=1:431.7@12,443 → 517.7@13,423(P90 +19.9%,吞吐 +7.9%);conc=2:+12.0%/+9.3%;conc=4:+8.9%/+2.1%;conc=16:+10.6%/+1.9%;conc=32:+3.2%/+1.8%。GSM8K(1319 题 5-shot,max_tokens 16384,真实 MTP 验证):1 thread 0.968→0.967,64 threads 0.970→0.973。

## 验证与回归风险

`test_wo_a_fused.py` 参数化 groups 2/4(数值对 reference、UE8M0 scale 逐位一致、scale padding 清零);`test_swapab_attention.py` 新增 `test_tp2_head_groups`(32 heads 两组各读各的 query/sink、graph replay);GB300 上两测试通过。风险面:TP2 fused WO-A 启动 64 clusters/512 CTAs、约 1.7 waves(代码注释推断这会限制 WO-A 增益);gfx950 HIP kernel 保持 16-head 契约(head-groups 用例 skipIf is_hip);mHC gate 放宽影响所有命中形状的部署。pack 显示 CI Extra 失败,Base/AMD ROCm 10 待跑。

## 关联

同区域:同批 #43786(DeepSeek-V4 FP8 wo_a 融合 inverse RoPE,引用的 #39957 为 BF16 wo_a 融合前序)。三者共同构成 DeepSeek-V4/V4.1 wo_a 小批量融合系列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
