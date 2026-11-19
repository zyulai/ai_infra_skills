# PR #62590: [ROCm][Perf][Qwen4Exp] Tune sparse GQA prefill profile on gfx942 and gfx950

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tc3oliver(与 @eppaneamd 共同署名) |
| 合并 | 2026-11-17 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | rocm, ready, qwen |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/62590 |

## 一句话结论

在 CDNA3(gfx942)与 CDNA4(gfx950)上重调 `qsa_sparse_paged_attention` 的 prefill 启动 profile(超过 512 programs 时改为每 program 一个 warp,gfx942 BF16 K/V 且 `BLOCK_M >= 16` 例外用 `BLOCK_N=32`),使 Qwen3.8-Flash-Next 的 prefill 稀疏注意力快 1.2-2.2x,长 prompt 服务在 TP2 下再快 2-3%,精度不变。

## 动机

`>512-program` 的 prefill 分支此前是 `BLOCK_N=64, 1 split, 2 warps`(在 GB300 上调的)。该 profile 在 MI325X / MI355X 上并非最优:实测一个 warp 更快;唯一例外是 gfx942 BF16 K/V 且 `BLOCK_M >= 16`(TP1/TP2)时 `BLOCK_N=32` 更快(2.2x vs 约 1.5x)。两条规则不能合并:`BLOCK_N=32` 配单 warp 在 gfx942 BF16 `BLOCK_M=16` 会触发 `HSA_STATUS_ERROR_MEMORY_FAULT`,所以用 `elif` 分开。gfx950 的单 warp 规则来自被关闭并折叠进本 PR 的 #65341;draft #65474 叠在本 PR 之上继续调同一分支,#57969 只改 gfx11 的 `BLOCK_M`,故均非重复实现。

## 优化点

- 在 `qsa_sparse_paged_attention` 的 `>512 programs` 分支里按 `get_cdna_version()` 选 profile:gfx942 BF16 且 `BLOCK_M >= 16` 用 `(BLOCK_N=32, 2 warps)`;gfx942 FP8(或 BF16 `BLOCK_M < 16`)与 gfx950 全部用 `(BLOCK_N=64, 1 warp)`;其他 ROCm 架构、CUDA 及 `<=512 programs` profile 不变((64, 2))。
- FP8 不进 `BLOCK_N=32` 分支:kernel 接收 FP8 K/V 时该配置在 TP2 520 rows 会回归(0.78x),单 warp 则给 1.8-1.9x。
- 新增 6 个单测 shape(TP1/TP2/TP8 的 `_split1` 形状,各配 BF16/FP8),超过 512 programs、覆盖每个架构分支。
- 局限(正文明示):`.buildkite/test-amd.yaml` 只有 `:amd: (MI355 DPX) Qwen4 Exp` 跑 `test_qsa_amd.py` 且为 `optional: true`,无 MI300 步骤;端到端仅测 TP2;MI300X/MI350X 未测;`BLOCK_N=16` + 单 warp 在 >=2,048 rows 还能再快 1.2-1.3x 但 520-1,024 rows 更慢,留给 #65474。

## 关键改动

```python
# vllm/models/qwen4_exp/amd/ops/qsa.py(>512-program 分支)
from vllm.platforms.rocm import get_cdna_version

# One warp per program is faster on CDNA3 and CDNA4, except BF16
# K/V with BLOCK_M >= 16 on CDNA3, where BLOCK_N=32 is faster
# (BLOCK_N=32 with one warp faults there).
cdna_version = get_cdna_version()
if cdna_version == 3 and block_m >= 16 and not is_fp8:
    block_n = 32
elif cdna_version in (3, 4):
    partial_warps = 1
```

```diff
 # tests/models/qwen4_exp/test_qsa_amd.py:新增 _split1 形状
+        pytest.param(257, 24, 2, 1792, id="tp1_split1"),
+        pytest.param(513, 12, 1, 1792, id="tp2_split1"),
+        pytest.param(513, 3, 1, 1024, id="tp8_split1"),
```

## 性能声明

- Kernel(8,192 rows,520-16,384 rows 的 min-max;MI325X 与 MI355X,BF16 与 FP8 K/V;无任何实测格子慢于 main):TP1 2.05x(2.05-2.34, gfx942 BF16)/1.87x/1.63x/1.56x;TP2 2.23x(1.42-2.37)/1.94x(1.08-1.94)/1.70x(1.54-1.72)/1.58x(1.41-1.58);TP4 1.20x/1.33x/1.38x/1.78x;TP8 1.17x/1.28x/1.25x/1.24x(顺序 gfx942 BF16 / gfx942 FP8 / gfx950 BF16 / gfx950 FP8)。TP2 绝对值:5.35 → 2.39 ms(gfx942 BF16)、2.76 → 1.42 ms(gfx942 FP8)、2.51 → 1.47 ms(gfx950 BF16)、2.11 → 1.33 ms(gfx950 FP8)。
- 端到端(Qwen3.8-Flash-Next-FP8 TP2,长 prompt 格子 32k 与 128k prefill、60k/600):吞吐 +1.5 至 +3.4%(MI325X BF16 KV)、+2.0 至 +3.4%(MI325X FP8 KV)、+1.6 至 +3.1%(MI355X);decode-heavy 格子(1k/1k、8k/1k)±1% 内,除 MI325X 8k/1k 并发 32/128 时 +2.4-2.5%;median ITL 不变。MI325X BF16 KV 32k/8/8:12,190 → 12,602 tok/s(+3.4%)、TTFT 12,393 → 11,993 ms(-3.2%)。
- 测试条件:CUDA events、10 次预热 50 次计时、两臂 3 轮交错取中位数;MI355X 跑两轮并交换 GPU 对;GT 为 `vllm/vllm-openai-rocm:nightly-60bd6dd2edd175a1c55fd3488f871e4c3d2ef27d`(vLLM 0.34.0rc1.dev571)。
- 精度:除 gfx942 BF16 TP1/TP2(`BLOCK_N=32`,最大绝对差 2.0e-3,与 main 相对 fp64 参考误差相同)外逐位相等;GSM8K 5-shot 0.9682 → 0.9697(10 fixed / 8 broken,McNemar p = 0.82),RULER niah_single_2 在 64k/128k 均为 1.000。

## 验证与回归风险

- 单测:`pytest tests/models/qwen4_exp/test_qsa_amd.py` MI325X 与 MI355X 各 29 passed(main 为 23)。服务命令含 `VLLM_ROCM_USE_AITER=1` 等,AITER MoE 与 AITER custom all-reduce 关闭(block-FP8 checkpoint 在 TP2 下二者都会 raise)。
- 回归风险:gfx942 BF16 TP1/TP2 走新 `BLOCK_N=32` 分支,输出不再逐位等于 main(≤2.0e-3);CI 覆盖缺口——MI300 步骤不跑该测试文件,gfx942 分支只靠手测;24q/1kv 形状(`BLOCK_M=32`)非 Qwen3.8-Flash-Next 形状且不在测试内。

## 关联

- Co-authored-by: @eppaneamd <rony.leppanen@amd.com>;gfx950 单 warp 规则源自 #65341(已关闭并折叠)。
- 叠于其上的 draft #65474 对 BF16 K/V 按程序数选 splits 继续调优;#57969 改 gfx11 `BLOCK_M`,均非重复。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
