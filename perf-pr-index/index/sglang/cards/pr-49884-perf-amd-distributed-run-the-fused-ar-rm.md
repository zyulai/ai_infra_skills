# PR #49884: [Perf] [AMD] distributed: Run the fused AR + RMSNorm + per-group FP8 quant on gfx942

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jiaqiang-dot-liu |
| 合并 | 2026-11-17 |
| 分类 | Speculative Decoding(major) |
| Label | run-ci, run-ci-amd-extra |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/49884 |

## 一句话结论

把 #24651 引入的融合 all-reduce + RMSNorm + per-group FP8 量化的单 aiter 内核从 gfx95-only 放宽到 gfx942(MI300X/MI325X):Qwen3.5 FP8 attention/GDN 输入投影每层少一次内核启动和一次激活读写,每 profiled decode step 省约 90 us GPU 时间(约 1.5%),端到端 MI300X TP2 吞吐 +1.7%(在波动范围内)。

## 动机

#24651 为 Qwen3.5 FP8 attention 与 GDN 输入投影加了融合 AR + RMSNorm + per-group FP8 量化路径,但用 `is_gfx95_supported()` 把该 aiter 内核只对 gfx95 开放。gfx942 上同一调用回退到"融合 AR + RMSNorm 内核 + 独立 per-1x128 量化内核"两步,每层输入投影多付一次内核启动与一次归一化激活的读写。aiter 内核本身并非 gfx950 专属:它是通用 HIP(arch 选择 FP8 类型,gfx942 上为 `e4m3fnuz`),aiter 自带的 `test_fused_ar_rms_per_group_quant.py` 在 2× MI300X TP2 全形状全选项组合通过。

## 优化点

- `GroupCoordinator.fused_allreduce_rmsnorm_quant_per_group`(`parallel_state.py`):gate 由 `is_hip() and is_gfx95_supported()` 改为 `is_hip() and (is_gfx95_supported() or is_gfx942_supported())`;形状、尺寸、world-size 检查不变。
- `parallel_state.py` 与 `layernorm.py` 的 docstring 更新为 gfx942 与 gfx95。
- 下游零改动:gfx942 上两步回退本就产出相同的 `(fp8, scale)` / `(bf16, fp8, scale)` 元组给投影,行主 scale 布局不变(column-major `transpose_scale` 路径仅 gfx95 bpreshuffle GEMM 使用)。
- 效果:gfx942 上开 `--enable-aiter-allreduce-fusion` 后,Qwen3.5 FP8 attention/GDN 输入投影的 FP8 输入来自单个融合内核;`SGLANG_DISABLE_FUSED_AR_QUANT=1` 仍可退出。
- 新增单元测试钉住 gfx95/gfx942 调用融合内核、其他 HIP arch 返回 None。

## 关键改动

```python
# python/sglang/srt/distributed/parallel_state.py
-        if not (is_hip() and is_gfx95_supported()):
+        if not (is_hip() and (is_gfx95_supported() or is_gfx942_supported())):
             return None

# test/registered/unit/distributed/test_parallel_state.py (新增)
@pytest.mark.parametrize("gfx95, gfx942, expect_fused",
    [(True, False, True), (False, True, True), (False, False, False)])
def test_fused_allreduce_rmsnorm_quant_per_group_runs_on_gfx942_and_gfx95(...):
    out = coordinator.fused_allreduce_rmsnorm_quant_per_group(
        x, torch.zeros_like(x), weight, 1e-6)
    assert out == ("fused" if expect_fused else None)
```

## 性能声明

(全部照抄 PR 正文,未独立核实;PR 正文的正则未检出明确百分比,数字均取自正文表格与叙述)

- 内核级 2× MI300X(Qwen3.5-122B-A10B-FP8,TP2,`--enable-aiter-allreduce-fusion`,EAGLE MTP,30 步 decode torch profile,AgentX trace 并发 24):融合 AR+RMSNorm(无量化)内核 49.0→25.5 次/步;独立 per-group quant 102.0→78.5 次;融合 AR+RMSNorm+per-group quant(7.25 us)0→23.5 次。每个 attention/GDN 输入从 6.45 + 4.64 us 两内核变为一个 7.25 us 内核,每 profiled step 省约 90 us GPU 时间,约 1.5% decode GPU 时间。
- 内核精度(aiter 多卡测试,TP2,group 128):6 个 (M,K) 形状 × 8 组合全过,per-group 最大误差 1.0e-05 至 2.4e-04;CUDA graph 延迟 5.9–583.6 us。
- 端到端 MI300X TP2(AgentX replay 并发 24,1200 s,AL 3.39,各一 run):吞吐/GPU 16,977→17,270 tok/s(+1.7%),TPOT p90 140.9→107.8 ms,TTFT mean 5.66→6.70 s——含 TTFT 增加在内均在 replay 波动内,内核计数是直接证据。
- 端到端 Qwen3.5-397B TP8 8× MI325X(同 trace):吞吐/GPU 10,440→10,522 tok/s(+0.8%),TPOT p90 12.1→11.3 ms,亦在波动内但方向符合。
- 精度:GSM8K 5-shot 1319 题,122B TP2:flexible 0.9742→0.9765、strict 0.9750→0.9780、accept length 3.564→3.552(stderr ±0.004 内);397B TP8:flexible 0.9765(±0.0042)、strict 0.9773(±0.0041)、accept 3.527。

## 验证与回归风险

改动极小(gate 一行 + docstring + 单测);单元测试用 monkeypatch 钉住三种 arch 组合的融合/回退行为。风险面:aiter 内核在 gfx942 上的形状覆盖依赖 aiter 仓库测试(PR 引用其结果);`SGLANG_DISABLE_FUSED_AR_QUANT=1` 退出开关保留。Base CI 与 AMD ROCm 10 CI 通过,Extra 显示失败图标(pack 未提供细节)。

## 关联

#24651(引入融合路径并 gate 到 gfx95)、aiter 多卡测试 `test_fused_ar_rms_per_group_quant.py`、#49883/#40872/#7842(397B 测量中同时携带的 unified_attention 路由与 CK tuning)、Hyperloom(发现来源)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
