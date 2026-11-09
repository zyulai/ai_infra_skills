# PR #39505: [compile] Add FlashInfer FP8 async TP fusion and preserve allreduce fusion ordering #27893

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @baonudesifeizhai |
| 合并 | 2026-05-01 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready, torch.compile, nvidia, model-bash |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/39505 |

## 一句话结论

为 FlashInfer FP8 GEMM 启用 AsyncTP 融合:把 `all_gather + bmm_fp8` 与 `bmm_fp8 + reduce_scatter` 重写为 SymmetricMemory 支撑的融合 op(解决 #27893)。

## 动机

vLLM 的 AsyncTP 编译 pass 已有 BF16 GEMM、scaled_mm、cutlass_scaled_mm 的 all-gather/reduce-scatter 融合模式,但 FlashInfer FP8 GEMM(`vllm.bmm_fp8`)不在其中,FlashInfer FP8 路径在 TP 下无法享受通信-计算融合;此前 e2e 测试甚至在 Blackwell 上禁用 `FlashInferFP8ScaledMMLinearKernel` 来绕开"FlashInfer scaled_mm FP8 不支持 async tp patterns"的限制(issue #27893)。

## 优化点

机制:新增两个 pattern matcher 模式,把 `all_gather → vllm.bmm_fp8` 匹配为 `fused_all_gather_flashinfer_scaled_matmul`,把 `vllm.bmm_fp8 → reshape → reduce_scatter` 匹配为 `fused_flashinfer_scaled_matmul_reduce_scatter`;两者通过 PyTorch SymmetricMemory 的 `_fused_all_gather_matmul_impl`/`_fused_scaled_matmul_reduce_scatter_impl` 实现,注入自定义 `mm_out_op` 适配器。适配器依赖新增的 `flashinfer_scaled_fp8_mm_out`(out 变体,复用 FlashInfer `bmm_fp8` 的列主序 weight 视图,仅支持 tensor-wise scale、无 bias/scale_result/fast_accum)。仅在 model_dtype==bfloat16 且 `torch.ops.vllm.bmm_fp8` 存在时注册。附带把 Blackwell(sm100/sm103)折叠进 capability family 100 的 sequence parallelism 阈值(hidden_size>=8192,per-GPU 32MB),并让测试 conftest 支持 `is_device_capability_family`,移除 `VLLM_DISABLED_KERNELS` 绕行补丁。

## 关键改动

```text
vllm/compilation/passes/fusion/collective_fusion.py: 两个 FlashInfer FP8 AsyncTP pattern + 自定义 op 注册
vllm/utils/flashinfer.py: flashinfer_scaled_fp8_mm_out(out 变体适配器)
vllm/compilation/passes/fusion/sequence_parallelism.py: Blackwell family SP 阈值(100: 8192 / 32MB)
tests/compile/fusions_e2e/test_tp2_async_tp.py: 移除 Blackwell 上禁用 FlashInfer FP8 kernel 的绕行
```

## 性能声明

PR 正文未给出文字性能数字(Test Result 仅为截图,链接外部 paste);未注明测试条件。性能收益定性为 FlashInfer FP8 GEMM 在 TP 下可用 AsyncTP 通信-计算融合,消除了独立 all_gather/reduce_scatter 与 GEMM 之间的等待。

## 验证与回归风险

依赖 `tests/compile/fusions_e2e/test_tp2_async_tp.py`(TP2 FP8 fusion 匹配测试,现覆盖 FlashInfer 后端)与 `tests/compile/conftest.py` 的 mock platform 扩展;FLASHINFER 后端在无 trtllm attention 支持时将 `attn_quant_fusion` 匹配数置 0。风险面:仅影响 BF16 模型 + TP + torch.compile AsyncTP 路径;FlashInfer 适配器限制较多(2D、tensor-wise scale、gather/scatter dim=0、M 可被 world_size 整除),不满足时模式不匹配即回退原有分离算子;SP 阈值变化会改变 Blackwell 上 sequence parallelism 的自动启用点(TP8 更保守)。

## 关联

同区域:#27893(对应 issue);AsyncTP 既有模式:GEMMReduceScatter/AllGatherGEMM(BF16)、ScaledMM/CutlassScaledMM 系列;#52239(SymmetricMemory multicast 方向的 NVIDIA 优化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
