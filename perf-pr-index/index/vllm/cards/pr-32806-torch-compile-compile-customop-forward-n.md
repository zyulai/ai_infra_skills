# PR #32806: [torch.compile] Compile `CustomOp.forward_native` for `SiluAndMul` and `QuantFP8` to avoid raw torch ops inside opaque custom ops

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ProExpertProg |
| 合并 | 2026-01-23 |
| 分类 | MoE (tier: major) |
| Label | ready, torch.compile, ready-run-all-tests |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 7 个文件(custom_op 基类、activation/quant 层与测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/32806 |

## 一句话结论

在 opaque custom op 内部手动 torch.compile `forward_native`(SiluAndMul/QuantFP8),llama4 平均延迟 -15.8%,DSR1-nvfp4 -3.7%。

## 动机

当 `CustomOp.forward_native` 被另一个 opaque torch custom op(如 `fused_moe`、`unified_attention`)调用时,它对 model 级 torch.compile 不可见,内部只会执行 eager PyTorch 算子,性能损失明显。典型受害者是 llama4 与 DeepSeek 的 shared expert:它们在 `fused_moe` custom op 内部用 `SiluAndMul` 和 `QuantFP8` 做激活与量化。而单纯改用 CUDA kernel 收益很小(正文实测 custom op 比 compiled native 更慢)。

## 优化点

在 `CustomOp` 基类新增 `maybe_compile`:`dispatch_forward` 增加 `compile_native` 参数,当该 op 未启用(走 `forward_native` 回退路径)且 `compile_native=True` 时,用 `torch.compile(fn, dynamic=True, backend=current_platform.simple_compile_backend, options=maybe_disable_graph_partition(...))` 手动编译 `forward_native`,避免其在 opaque custom op 内执行 raw torch ops。多层防护:全局 `CompilationMode.NONE`、backend 为 "eager"、或 `enable=False` 时不编译;已在编译区域内的调用自然被 torch.compile 忽略,不影响现有可见的 CustomOp 调用;注明这不带来跨 op 融合,opaque custom op 仍应尽量展开。默认仅对 `SiluAndMul` 与 `QuantFP8` 开启(两者 `__init__` 默认 `compile_native=True`),避免引发大量编译问题;`matcher_utils` 的 act+quant 融合路径显式传 `compile_native=False` 以免与融合 pass 冲突;`CustomOp.__init__` 改为 keyword-only 参数。

## 关键改动

```text
vllm/model_executor/custom_op.py: 新增 maybe_compile;dispatch_forward(compile_native) 在 forward_native 回退路径上手动编译
vllm/model_executor/layers/activation.py: SiluAndMul 默认 compile_native=True
vllm/model_executor/layers/quantization/input_quant_fp8.py: QuantFP8 默认 compile_native=True
vllm/compilation/matcher_utils.py: 融合路径构造 QuantFP8 时传 compile_native=False
tests/compile/test_silu_mul_quant_fusion.py、tests/kernels/core/test_activation.py: 测试适配
```

编译包装(节选):

```python
def maybe_compile(self, fn, *, enable: bool = True):
    if not enable:
        return fn
    compilation_config = get_cached_compilation_config()
    if compilation_config.mode == CompilationMode.NONE:
        return fn
    if compilation_config.backend == "eager":
        return fn
    return torch.compile(
        fn, dynamic=True,
        backend=current_platform.simple_compile_backend,
        options=maybe_disable_graph_partition(
            current_platform.simple_compile_backend),
    )
```

## 性能声明

Llama4(`vllm bench latency --model nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 --max_num_seqs=256 --max_model_len=1024`):平均延迟 main 1.5058 s 对 PR 1.3005 s,平均提升 15.8%;启动时间基本不变(冷启动 169.5 对 168.0 s)。DSR1-nvfp4(`vllm bench latency/throughput --model=nvidia/DeepSeek-R1-NVFP4 -tp=4 --enable-expert-parallel`,仅 silu 受益):平均延迟 2.2826 s 对 2.2007 s,提升 3.7%;吞吐基本持平(约 19.8 req/s);若改用 CUDA custom op 而非编译 native 为 2.2709 s,说明 compiled native 更优;GSM8K(5-shot, limit 100)两侧均 0.96。llama3.1-fp8(RedHatAI/Meta-Llama-3.1-8B-FP8)不受影响,延迟差在 1.59400e-3 内。测试条件:PR 未注明 GPU 型号。

## 验证与回归风险

Test Plan 为 CI 与 eval,正文 Test Result "Looks good";DSR1 附 lm_eval GSM8K 数字佐证精度无损。风险面:仅 SiluAndMul/QuantFP8 两类 op 的 forward_native 会额外被 torch.compile(dynamic=True),首次调用有编译开销(dynamic=True 用于避免 recompilation);其他 CustomOp 行为不变(compile_native 默认 False);`CustomOp.__init__` 参数改 keyword-only 可能影响按位置传 `enforce_enable` 的子类(rotary_embedding 已同步改为 keyword 形式);与 act+quant 融合 pass 的冲突通过 `compile_native=False` 显式规避。作者定位为过渡方案,后续将迁移到 vLLM IR 或 PluggableLayer。

## 关联

同区域 torch.compile 优化:tests/compile/test_silu_mul_quant_fusion.py 对应的 silu_mul+quant 融合 pass(素材中未给编号)。PR 内未引用其他 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
