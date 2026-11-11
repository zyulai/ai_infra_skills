# PR #45264: [Perf] [NVIDIA] mhc: Use BLOCK_K=32 for the hc_mix kernel on SM120

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @moria97 |
| 合并 | 2026-11-10 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, run-ci, jit-kernel |
| Diff | +154/-8,2 files(pack diffstat 缺失,按 pack 内 diff 文本计数) |
| 链接 | https://github.com/sgl-project/sglang/pull/45264 |

## 一句话结论

mHC `hc_mix` 统计 kernel 在 SM120 上改用 BLOCK_K=32,大 batch 最高 1.55×,单请求 32K prefill +2.5%。

## 动机

mHC 的 `hc_mix` stats kernel 此前在所有平台统一使用 BLOCK_K=64;在 SM120 上实测 BLOCK_K=32 一致更快,值得按平台分派。

## 优化点

新增 `functools.lru_cache` 的 `_hc_mix_stats_sinkhorn_block_k(device)`:按输入 device 的 capability 懒查询(避免 fork 前初始化 CUDA),仅精确匹配 (12,0) 且非 HIP 时返回 32,否则保持 `_HC_MIX_BLOCK_K=64`;GB10(SM121)未测量故明确排除。`_num_slices_for` 改为接收 block_k 参数,slice 数仍只依赖 K 与 tile 宽度、不依赖 batch M,保住注释中的 batch-invariance 契约。BLOCK_K=32 时把 `block_m` 提升为 `max(16, block_m)`(SM120 的 8-row tile 相对 16/32-row batch 改变舍入)。bf16x3 变体拆出独立常量 `_HC_MIX_BF16X3_BLOCK_K=64`,不受本次调参影响。

## 关键改动

```text
python/sglang/kernels/ops/layernorm/mhc.py: SM120 的 BLOCK_K 选择、block_m 下限、bf16x3 常量拆分
test/registered/kernels/ops/layernorm/test_mhc_sm120.py: 新增注册测试
```

```python
@functools.lru_cache(maxsize=None)
def _hc_mix_stats_sinkhorn_block_k(device: torch.device) -> int:
    if (
        device.type == "cuda"
        and not _is_hip
        and torch.cuda.get_device_capability(device) == (12, 0)
    ):
        return 32
    return _HC_MIX_BLOCK_K
```

## 性能声明

RTX PRO 5000,DeepSeek-V4.1-Flash shapes,kernel 延迟(μs):M=1 23.2→22.5(1.03×)、M=256 37.4→27.2(1.38×)、M=1024 125.6→88.3(1.42×)、M=4096 683→451(1.51×)、M=16384 2818→1819(1.55×)。端到端:单请求 32K prefill 11443→11734 tok/s(+2.5%);decode 不变。kernel 级相对 diff ≈3e-7;GSM8K 0.913 vs 0.905(噪声)。合并分支验证:upstream a90e9c5d + #42371 + 本 PR + 三个 SM120 companion PR(DeepGEMM metadata、b12x MXFP8、RTX PRO 5000 configs),8×RTX PRO 5000、DP8 attention+EP8、`--fp8-gemm-backend flashinfer_cutlass`:clean 分支 GSM8K 0.905/32K prefill 12020 tok/s,env-gated dev build 0.913/11734 tok/s。

## 验证与回归风险

新测试断言:模块体不做 eager capability 查询(reload 检查);tile 选择按输入 device 且排除 SM121/SM100/Hopper/HIP;SM120 上 BLOCK_K=32 与 64 输出在 rtol 1e-5/atol 2e-6 内一致,并对 m=1..2049 的全部行做 bit-exact batch 不变性(逐 8 行走真实小 batch launch);bf16x3 输出与 partial tile 宽度无关(bit-exact)。风险:capability 仅精确匹配 (12,0),SM121 及其他 Blackwell 走 64;BLOCK_K=32 配套 block_m≥16 改变小 batch 舍入,已有测试覆盖。

## 关联

与 #42371 及三个 SM120 companion PR(DeepGEMM metadata、b12x MXFP8、RTX PRO 5000 configs)合并验证;素材未给出这些 PR 的完整编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
