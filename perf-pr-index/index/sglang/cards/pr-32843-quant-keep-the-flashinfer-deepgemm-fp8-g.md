# PR #32843: [Quant] Keep the flashinfer_deepgemm FP8 GEMM to 1 <= M < 32

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zhendonghua |
| 合并 | 2026-08-01 |
| 分类 | Quantization (tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/32843 |

## 一句话结论

限制 `flashinfer_deepgemm` FP8 GEMM 只在 1 <= M < 32 生效,避开 flashinfer 在 M >= 32 上最慢(1.42x slower)的区间,转回 DeepGEMM/triton。

## 动机

`--fp8-gemm-backend flashinfer_deepgemm` 把所有 dense block-FP8 linear 都路由到 `flashinfer.gemm.fp8_blockscale_gemm_sm90`,但这一个 Python 入口底下实际是两个不同的 kernel:在 H200 上 profiling 显示 M < 32 走 `deep_gemm::fp8_gemm_kernel_swapAB`,M >= 32 走 `deep_gemm::fp8_gemm_kernel`。只有 swapAB 那一半值得用;M >= 32 区间 flashinfer 反而比 DeepGEMM 慢,且在某些 checkpoint 上还有精度问题。vLLM 的 `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` 已对同一 kernel 加了 `input.shape[0] < 32` 的门控。此外 M == 0(空 batch)会让该 kernel 硬报错,而 DP attention 下空闲 rank 的零 token forward 是常态输入。

## 优化点

在 `fp8_utils.py` 的 `flashinfer_deepgemm_w8a8_block_fp8_linear_with_fallback` 中引入 `m_supported = 1 <= M < 32` 的形状门控,并折入既有的 shape/dtype fallback,形成单一 triton 出口(保留 UE8M0 scale 解包)。当 `m_supported` 不满足且 JIT DeepGEMM 可用时,直接转交 `deepgemm_w8a8_block_fp8_linear_with_fallback`——DeepGEMM 覆盖两端(M=0 与 M>=32),对它无法服务的形状自带 triton fallback。最终路由(H200 上对 M in {0,1,8,31,32,64} x DeepGEMM 开/关验证):M=0 与 M>=32 任何情况下都不会进 flashinfer;1-31 走 flashinfer;其余走 deep_gemm(可用)或 triton(不可用)。该文件此前已有同类形状门控先例(`aiter_w8a8_block_fp8_linear` 的 per-shape M 上界)。

## 关键改动

```text
python/sglang/srt/layers/quantization/fp8_utils.py: 为 flashinfer_deepgemm 路径加 1 <= M < 32 门控,越界转 DeepGEMM 或 triton
```

关键片段:

```python
m_supported = 1 <= input.view(-1, input.shape[-1]).shape[0] < 32

if not m_supported and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
    return deepgemm_w8a8_block_fp8_linear_with_fallback(
        input, weight, block_size, weight_scale, input_scale, bias
    )

if not (shape_supported and dtype_supported and m_supported):
    if weight_scale.dtype == torch.int32:
        weight_scale = _unpack_ue8m0_scale_for_triton(...)
```

## 性能声明

H200 上的 per-shape 微基准(flashinfer vs DeepGEMM):

- M=16(decode):flashinfer 快 1.05x
- M=32-64:flashinfer 慢 1.42x(其最差区间,恰在阈值上方)
- M=850(prefill):flashinfer 慢 1.10x

精度方面(次要、依赖 checkpoint):Qwen3.6-27B-FP8 上不加门控的 gsm8k 掉 3.8 分(两次重复噪声 0.15 分),复现于 prefill;但 Qwen3-4B-Instruct-2507-FP8 上三者(0.9016/0.8955/0.8940)在噪声内一致,PR 明确说明性能论证才是主要依据。另注明 M=0 时 flashinfer 路径直接 `RuntimeError`(N=2560, K=2048, block [128,128] 实测)。

## 验证与回归风险

两套测试均在 H200(SM90)运行:`test/registered/quant/test_fp8_blockwise_gemm.py`(DeepGemm 0.9016 / FlashinferDeepGemm 0.8955)、`test/registered/ep/test_deepep_small.py`(6 个非 MTP 类通过,3 个 MTP 类因权重不可公开访问未跑)。PR 强调 CI 并不覆盖该 backend(唯一选择它的测试注册在 4-gpu-b200 runner 上且被 SM90 skipIf 变成 no-op),需要 Hopper 手工验证。回归风险:该 backend 保持 opt-in,auto 模式默认不变;但 PR 记录了与 `--moe-a2a-backend deepep` + async NaN assertion 组合下服务启动失败(NaN)的预存问题,明确不在本 PR 修复,并建议用户暂勿组合这两个 flag;还记录了一例 M=8 干净输入仍出 NaN 的未解释观察。

## 关联

vLLM 侧同构实现 `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`(同门控,触发条件 `input.shape[0] < 32`);文件内先例 `aiter_w8a8_block_fp8_linear` 的 CK M 上界门控。pack 素材未给出同分类其他 sglang PR 编号,DeepEP NaN 问题 PR 称将另行提出,编号未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
