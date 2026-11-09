# PR #64417: [Perf] Skip custom-op dispatch for FlashInfer dense GEMMs in eager

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @YangXu1990uiuc |
| 合并 | 2026-11-03 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | performance, ready, nvidia, quantization |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,涉及 6 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/64417 |

## 一句话结论

eager 模式下绕过 FlashInfer dense GEMM 系 custom op 的 dispatcher,每次 linear 调用省 31-53µs host 开销。

## 动机

`vllm::mm_mxfp8`、`vllm::flashinfer_mm_fp4` 等 FlashInfer GEMM/量化 op 用的是 `@torch.library.custom_op` 慢注册路径,每次 eager linear 调用(如无 CUDA graph 的 prefill)都要走完整 dispatcher。实测一个同签名 no-op:custom_op 形式 8.3µs、`direct_register_custom_op` 3.2µs、普通函数调用 0.15µs;真实 GEMM 经过 op 比直调多 32-35µs(FlashInfer 自身的 Python dispatch 在 dispatcher 下也更慢)。配套改动 flashinfer-ai/flashinfer#7788(已合并)已从 FlashInfer 侧省掉 25-50µs,本 PR 解决 vLLM 侧的附加开销。

## 优化点

核心是新增 `eager_bypass_custom_op()`:包一层 wrapper,只在 `torch.compiler.is_compiling()` 时走 dispatcher(op 作为 Dynamo graph 边界需要保留),eager 调用直接落到实现函数。配套:(1) 把 `mm_mxfp8`、`flashinfer_mm_fp4`、`bmm_fp8`、`flashinfer_mm_bf16_fp4`、三个 quantize op 从 `torch.library.custom_op` 迁到 `direct_register_custom_op`(op 名、schema、fake impl 不变),已直接注册的 `cuda_flashinfer_bf16_gemm` 与 FP8 block-scale 两个 op 也接上 bypass;(2) `mxfp8_e4m3_quantize` 非 compile 时直调 impl;(3) 用 `functools.cache` 缓存"是否用 FlashInfer MXFP8 quantize"的判定(原来每次调用做 2 个 import + capability 查询 + `has_flashinfer()`)。明确不做的:`skip_check=True`(再省 8-15µs 但丢校验)与 quantize+GEMM 折叠成一个 op(会藏起 `_C.scaled_fp4_quant`,影响 attention/quant fusion)。

## 关键改动

```text
vllm/utils/torch_utils.py: 新增 eager_bypass_custom_op(compile 走 op,eager 直调)
vllm/utils/flashinfer.py: 7 个 FlashInfer op 迁移注册 + 接 bypass
vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py: FP8 block-scale op 接 bypass
vllm/model_executor/layers/utils.py: cuda_flashinfer_bf16_gemm 接 bypass
vllm/model_executor/layers/quantization/utils/mxfp8_utils.py: quantize 判定缓存 + 非 compile 直调
vllm/model_executor/kernels/linear/nvfp4/flashinfer.py: 改调 bypass 后的 flashinfer_mm_bf16_fp4
```

关键片段(torch_utils.py):

```python
def wrapper(*args, **kwargs):
    if torch.compiler.is_compiling():
        return op(*args, **kwargs)
    return op_func(*args, **kwargs)
```

## 性能声明

Host µs/call(enqueue rate,warm,无 per-call sync;FlashInfer cute-dsl 后端即 SM100 默认;Kimi K3 投影形状 M=4096,含 FlashInfer #7788;15 轮 × 150 calls 取中位数,IQR ≤ 2µs):MXFP8 q_b 18432×1536 裸调 75.8 / vLLM main 141.7 / 本 PR 91.0(省 50.7);MXFP8 o_proj 69.7 / 136.8 / 85.4(省 51.4);NVFP4 q_b 63.3 / 130.3 / 97.3(省 33.0);NVFP4 o_proj 66.1 / 134.5 / 101.8(省 32.6)。cutlass 后端同幅省 31-51µs;M=8 与 M=4096 差几µs 内。孤立 op 微基准(64×256×512):mm_mxfp8 56.5→41.7、flashinfer_mm_fp4 58.0→43.7、bmm_fp8 58.2→42.6、quantize 类省 2-5µs;bf16 gemm 108.3→93.1、FP8 block-scale(H100)20.1→17.3。torch.compile 下 MXFP8 GEMM 每调用再省约 17-23µs(direct_register)。局限:数字来自逐行复刻 vLLM 调用路径的 harness,非端到端 serving benchmark;NVFP4 用同 schema op 替代了 C++ `scaled_fp4_quant`,绝对值可能偏高至多约 13µs(before/after 差值不受影响)。

## 验证与回归风险

验证:`torch.library.opcheck`(schema/autograd/faketensor/aot_dispatch)对重注册的 `vllm::mm_mxfp8`、`vllm::flashinfer_mm_fp4` 通过;eager 路径 / op 路径 / 直调 FlashInfer 三者输出 `torch.equal` 逐位一致(MXFP8 与 NVFP4 的 cutlass、cute-dsl);`torch.compile(fullgraph=True)` 仍记录 `vllm.mxfp8_quantize` 与 `vllm.mm_mxfp8` 无 graph break。风险面:dispatcher 被绕过意味着 eager 下不再经过 Dispatcher 层的 mode/代理(如 meta 设备、调试 hook);依赖 `torch.ops.vllm.*` 调用点的代码若未同步替换仍在慢路径;CI 覆盖 `tests/kernels/quantization/test_flashinfer_mxfp8_trtllm.py`。

## 关联

配套:flashinfer-ai/flashinfer#7788(上游 dispatch 提速)。同分类 CUDA Graph / Compiler:PR #29207;区域与 PR #19830(CustomOp 抽象)同属 op dispatch 开销链路。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
