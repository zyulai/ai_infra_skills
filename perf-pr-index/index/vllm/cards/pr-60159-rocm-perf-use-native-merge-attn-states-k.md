# PR #60159: [ROCm][Perf] Use native merge_attn_states kernel on gfx942/gfx950

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @amd-mghanimi |
| 合并 | 2026-10-08 |
| 分类 | Attention / Kernel (tier: major) |
| Label | rocm, ready |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际仅改动 1 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/60159 |

## 一句话结论

ROCm CDNA3+ 上把 merge_attn_states 从 Triton fallback 切到原生 kernel,单测中位 2.4x,chunked prefill 吞吐 +3.3~3.9%。

## 动机

`merge_attn_states`(merge 前缀与后缀两段 attention 输出与 LSE)在 CUDA 侧早已有原生 kernel(PR #16173 引入),但 ROCm 上 Python dispatch 只在 `current_platform.is_cuda()` 时走原生路径,导致 gfx942/gfx950 一直回落到 Triton 实现。该 kernel 在 chunked prefill(每个 extend 都要 merge)与 MLA 模型的两段式 prefill 中反复调用,Triton 版本明显慢于已有的原生实现。

## 优化点

不新增 kernel——原生 merge_attn_states kernel 在 ROCm 上已构建并注册,只是 dispatch 被门在 `is_cuda()`。本 PR 在 `vllm/v1/attention/ops/merge_attn_states.py` 中新增 `supported_platform()` 判断:CUDA 直接为真;ROCm 上用 `get_cdna_version() >= 3` 判定,即 CDNA3 及更新(gfx942、gfx950,gfx1250/CDNA5 也被选中但未实测);gfx90a(CDNA2)与 RDNA 继续留在 Triton 路径。dtype 与 headdim 的既有约束(`supported_dtypes`/`supported_headdim`)不变,不满足时仍回落 Triton。

## 关键改动

```text
vllm/v1/attention/ops/merge_attn_states.py: 新增 supported_platform(),CDNA3+ 放行原生 kernel,替换原先仅 is_cuda() 的门控
```

关键代码摘录(节选,<=15 行):

```python
def supported_platform() -> bool:
    if current_platform.is_cuda():
        return True
    if current_platform.is_rocm():
        from vllm.platforms.rocm import get_cdna_version
        return get_cdna_version() >= 3
    return False
```

## 性能声明

单元测试(tests/kernels/attention/test_merge_attn_states.py,2592 个计时 case/GPU,原生 vs Triton):
- MI325X(gfx942):2598 passed,min 1.10x / median 2.39x / max 8.58x,0 个更慢。
- MI355X(gfx950):2598 passed,min 1.64x / median 2.36x / max 8.30x,0 个更慢。
- FP8 输出时:MI325X 上原生快 2.1x–2.9x,MI355X 上快 2.2x–3.5x。

端到端(Kimi-K3,8x MI355X,TP=8,FP8 KV cache,concurrency 16):
- merge kernel 时间:60k/600 输入输出下 118.90 ms → 29.30 ms(4.1x);128k/1k 下 424.50 ms → 98.31 ms(4.3x)。
- serving 指标(TTFT/TPOT/吞吐)在 run-to-run 噪声内不变,因 merge 仅占 prefill kernel 时间的 0.26%(60k)/0.42%(128k)。

Qwen2.5-7B 单卡 MI350X(chunked prefill,input 65536/output 100,concurrency 8):
- 吞吐 +3.3%(1024 budget)至 +3.9%(8192 budget);median TTFT 下降 3.5%–4.2%。
- niah_single_2 准确率 8k/16k/32k 与 Triton 完全一致。

## 验证与回归风险

- 依赖现有测试 tests/kernels/attention/test_merge_attn_states.py(MI325X/MI355X 均 2598 passed);端到端用 lm_eval niah_single_2 与 vllm bench serve 验证。
- 回归风险面:gfx90a(CDNA2)与 RDNA 不受影响仍走 Triton;gfx1250 虽被 `>=3` 条件选中但未在实测覆盖中。FP8 输出 dtype 在 gfx942 为 float8_e4m3fnuz、gfx950 为 float8_e4m3fn,均验证与 Triton 输出一致。

## 关联

- 前序:PR #16173(CUDA 引入 merge_attn_states 原生 kernel,本 PR 复用其测试行)。
- 同区域:ROCm attention/merge 相关优化;PR 未列出其他直接后续。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
