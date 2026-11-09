# PR #32593: [Kernel] Enable Helion backend for Kimi Delta-Attention

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ethche |
| 合并 | 2026-08-15 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, quant, dependencies, Multi-modal, blackwell, npu, run-ci, jit-kernel |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及 3 个文件并另注明 11 个文件因超长被裁剪 |
| 链接 | https://github.com/sgl-project/sglang/pull/32593 |

## 一句话结论

为 Kimi Delta-Attention 新增 opt-in Helion 后端:decode 1.1x、prefill 1.6x(对 Triton 默认),Kimi-Linear-48B 端到端吞吐 +5.04%–5.27%。

## 动机

为 Kimi Delta Attention(KDA)的 packed decode 与 prefill 增加 opt-in Helion 后端(与已有的 Inkling silu_mul fusion 用法同类),同时保留现有默认后端、公共契约、状态突变语义与回落行为。Helion==1.4.0 作为可选依赖,仅 helion 后端需要,与其他 KDA 后端的做法一致。目标是把 KDA 的 recurrent decode 与 chunkwise prefill 迁到 Helion 代码生成,获得比手写 Triton 更好的调度而不在热路径引入 autotune 抖动。

## 优化点

新增 Helion 实现覆盖 packed recurrent decode 与 fixed/packed-varlen prefill;支持 FP32/BF16/FP16 recurrent states、padded cache indices(`ssm_state_indices == -1` 写零输出且不动 state)、bounded sigmoid gate(`lower_bound`,ALU/寄存器压力低于 unbounded softplus)、intermediate states 与 opt-in Newton-Schulz 研究路径。经 `--linear-attn-backend helion` 注册,phase 可用 `--linear-attn-decode-backend` / `--linear-attn-prefill-backend` 独立选择。generic decode、speculative verification 与 ReplaySSM 保持既有 Triton 回落。内核使用固定 Helion 配置与静态 shapes,模型热路径无 runtime autotuning;decode 按 state dtype、是否 bounded gate 与小 value-head 数(K3 在 TP=8 下本地 12 头,阈值 `KDA_SMALL_VALUE_HEAD_THRESHOLD=12`)在三个预调 config 变体中选择。decode 的收益约三分之二来自消除 packed 路径的 eager CPU dispatch(split + 3 次 unflatten + 额外 launch),benchmark 特意用 wall-clock CUDA-event 计时以计入这部分。

## 关键改动

```text
python/sglang/kernels/ops/attention/helion/kda_decode.py: Helion packed KDA decode 内核(3 个固定 config)
python/sglang/kernels/ops/attention/helion/kda_prefill.py: Helion chunk_kda prefill(l2norm/gate cumsum/fixed+varlen)
benchmark/bench_linear_attention/bench_kda_decode.py 等: 三方对照与正确性检查接入 Helion
```

```python
def _select_decode_kernel(
    *,
    is_bf16_state: bool,
    num_v_heads: int,
    use_lower_bound: bool,
) -> helion.Kernel:
    if (
        is_bf16_state
        and use_lower_bound
        and num_v_heads <= KDA_SMALL_VALUE_HEAD_THRESHOLD
    ):
        return _helion_fused_recurrent_kda_packed_decode_bf16_small_head
    if is_bf16_state:
        return _helion_fused_recurrent_kda_packed_decode_bf16
    return _helion_fused_recurrent_kda_packed_decode
```

## 性能声明

NVIDIA 2xGB200。microbenchmark geomean:decode 对 Packed Triton FP32/FP32 1.134x、BF16/BF16 1.067x、FP32 state/BF16 act 1.118x,对 FlashInfer 1.393x;prefill 对 Triton 1.603x(FP32 state)/1.595x(BF16)。H=12、lower_bound=-5.0(K3 本地形状):decode 对 Packed Triton 1.552x、对 custom CUDA decode 0.721x(CUDA kernel 融合 conv+recurrence+RMSNorm,Helion 落后,作者注明后续 PR 将在同等融合下评估);prefill 对 Triton 1.385–1.390x、对 Triton ReplaySSM 1.707x/1.225x。整 KDA wrapper 对 CuTe DSL:2.031x(50-shape geomean);作者注明 `bench_kda_prefill_cutedsl.py` 预计算 CuTe 元数据并绕过生产适配器,该脚本上 Helion 仅 0.86x,非忠实表征。端到端 Kimi-Linear-48B-A3B-Instruct,TP=2,input 8192,batch 1/4/16 × output 64/256(每格 16 测量):FP32 state geomean 吞吐 +5.04%、TTFT -5.22%、TPOT -4.83%;BF16 state +5.27%、-4.06%、-5.68%。

## 验证与回归风险

accuracy:decode 对 packed Triton——FP32 state 最大输出差 7.63e-6、state 差 2.24e-8;BF16 state 7.63e-6/1.22e-4。prefill 对 recurrent Triton:输出 7.32e-4、state 4.90e-3。覆盖 fixed/packed-varlen prefill、FP32/BF16/FP16 state、padded -1 index、partial chunks、`lower_bound=-5.0` safe gates、稳定 sub-chunk 归一化;全部输出/状态有限。Kimi-Linear GSM8K smoke(helion decode)0.915。新增注册 CUDA kernel 测试、CPU dispatcher 测试与 benchmark 覆盖;默认后端不变,opt-in 才启用。pack 中 CI:Base 通过,Extra 失败标记。

## 关联

正文未引用其他 PR 编号;新增 `helion==1.4.0` 可选依赖,与其它 KDA 后端的可选依赖方式一致。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
