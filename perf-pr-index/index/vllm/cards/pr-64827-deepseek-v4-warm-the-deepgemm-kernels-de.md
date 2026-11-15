# PR #64827: [DeepSeek-V4] Warm the DeepGEMM kernels DeepSeek-V4/V4.1 JIT-compile at runtime

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @HangruiCao2000 |
| 合并 | 2026-11-14 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | ready, deepseek, DSv4 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/64827 |

## 一句话结论

DeepSeek-V4/V4.1 会在 live request 上 JIT 编译被漏掉的 DeepGEMM kernel(o_proj einsum 与 mHC prenorm GEMM),DP+EP 下全体 rank 等待编译 rank、几步即可超 300 s 超时导致 EngineDeadError;本 PR 把两类 kernel 纳入启动 warmup,V4.1 的 DeepGEMM warmup 时间 118/130 s 降到 52/52 s。

## 动机

DeepSeek-V4/V4.1 在 live request 上 JIT 编译若干 DeepGEMM kernel。DP+EP 下每个 rank 都要等编译中的 rank,一步里出现几次即超过 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`(300 s),触发 RPC `sample_tokens` timed out → EngineDeadError,即 #41530 报告的症状。正文强调对生产 DeepSeek serving 影响极强——服务客户时不允许出现 JIT 编译。

## 优化点

1. **o_proj einsum 从未被 warm**:`fp8_einsum("bhr,hdr->bhd")`;`wo_a` 是 `is_bmm` linear,`DeepGemmFp8BlockScaledMMKernel` 刻意不把它注册为 `fp8_gemm_nt` warmup provider,而 `deep_gemm_warmup` 此前只覆盖 fp8_gemm_nt、grouped contiguous GEMM 与 Mega-MoE。M 采样:`relax` 模式在既有 boundary heuristic 之上补采每个 M % 16 == 1——Batched einsum 枚举的每个 block_m 都是 16 的倍数(SM100 swap-A/B 分支按 16 步进),config 只可能在 M % 16 == 1 处切换;旧采样照搬 SM90 的 [64, 128, 256] 列表,漏掉 SM100 的 BLOCK_M=48/112(@adenzhou1350 在 B300 上复现同样两个)。V4.1 mega attention 的 FP8 输出是不同 kernel(FP8QuantizationEpilogue,`use_128x4_sfd_layout=True`),warmup 数据与 cache key 均携带 `quantize_output` 并构建该 layout。
2. **mHC prenorm GEMM 的 DeepGEMM 路径(`tf32_hc_prenorm_gemm`)从未被 warm**:JIT registry 只 warm 了它所取代的 TileLang kernel。该 kernel 按 (n, k, num_splits) 实例化,M 是 launch 参数,故每个可达 split count 编译一次即覆盖全部 M;`HCPrenormGemmKernel`(`VllmJitKernel`,`kernels/mhc/warmup.py`)由各模型在自己的 mHC warmup 旁注册,并传入 forward 实际使用的 split 函数:V4 传 `compute_mhc_num_splits`(raw compute_num_split,现在 `_hc_prenorm_gemm_outputs` 也用它),V4.1 传 `compute_mhc_pre_num_splits`(1/4/16 取整)。token 上界取单 rank 可见值:sequence parallel 下为 `cdiv(max_num_batched_tokens, tp)`(mHC 跑在 sp_shard 上)。broadcast 权重(k = hidden_size)只在首个 PP rank 的 layer 0 注册。
3. 正文称 draft kernel warmup 不在本 scope;但 diff 中 `kernel_warmup.py` 仍把 `worker.get_draft_model()` 纳入 `deep_gemm_warmup`(drafter 在 target 模块树之外,其 o_proj 可能落在与 target 不同的 einsum kernel 上)——见"scope not covered":无 spec-decode run,该改动仅由 source 与 unit tests 验证。

## 关键改动

```text
vllm/model_executor/warmup/deep_gemm_warmup.py: FP8_EINSUM_WARMUP_CACHE + _fp8_einsum_warmup_data + _get_fp8_einsum_m_values(relax 补采 M%16==1)+ deepgemm_fp8_einsum_warmup 接入 deep_gemm_warmup
vllm/model_executor/kernels/mhc/warmup.py: compute_mhc_num_splits + HCPrenormGemmKernel(VllmJitKernel)+ HC_PRENORM_GEMM_KERNEL
vllm/model_executor/kernels/mhc/tilelang.py: _hc_prenorm_gemm_outputs 改用 compute_mhc_num_splits
vllm/model_executor/warmup/kernel_warmup.py: deep_gemm_warmup 遍历 (target, draft) 两个模型
vllm/models/deepseek_v4/nvidia/model.py / vllm/models/deepseek_v41/nvidia/model.py: 注册 HC_PRENORM_GEMM_KERNEL warmup(V4 raw splits;V4.1 1/4/16 rounding)
tests/model_executor/test_deep_gemm_warmup.py / test_mhc_prenorm_gemm_warmup.py(新增): einsum M 采样边界与 prenorm split 覆盖测试
```

```python
    if envs.VLLM_DEEP_GEMM_WARMUP == "relax":
        n = weight.size(1)
        m_values = set(_generate_optimal_warmup_m_values(max_tokens, n, weight.device))
        # Block-M candidates are 16-multiples, so configs change only at M % 16 == 1.
        m_values.update(range(1, max_tokens + 1, FP8_EINSUM_BLOCK_M_GRANULARITY))
        return sorted(m for m in m_values if 1 <= m <= max_tokens)
```

## 性能声明

- 单元:`pytest test_deep_gemm_warmup.py test_mhc_prenorm_gemm_warmup.py`:**25 passed**。
- Prenorm kernel 级(max_tokens 16384,两种权重形状;把 warmup keys 编译进空 `DG_JIT_CACHE_DIR`,再对 M=1..16384 用各模型 runtime split 函数逐个调用并计新编译):

| GPU | model | previous revision compiles | this PR compiles | M=1..16384 sweep 新编译 |
|---|---|---|---|---|
| GB200 | V4.1 | 37(31 个 V4.1 永不使用) | **6** | **0** |
| GB200 | V4 | 37 | 37 | **0** |
| H100 | V4.1 | 36(30 个永不使用) | **6** | **0** |
| H100 | V4 | 36 | 36 | **0** |

  一个 GB200 process 上,V4.1 编译集耗时 68.4 s → 11.2 s。
- E2E(`vllm serve`,previous vs this,每 run 一个空 `DG_JIT_CACHE_DIR`,默认配置 CUDA graphs on;`/health` 时快照已提交 cache entries,然后 1→16384 每 32 token 发一个 prefill,concurrency 1 再 16,共 1,026 requests):GB200 V4.1-Flash TP4+EP:previous(×2)prenorm kernels 40, 40 → this PR(×3)6, 6, 6,runtime DeepGEMM JIT 全为 0,errors 0;GB200 V4-Flash DP4+EP / TP4+EP:37 / 37,runtime 0;TP4+EP+Mega-MoE(SP):37 → **35**,runtime 0;H100 V4-Flash TP8:36 / 36,runtime 0。
- **V4.1 冷启动**(仅计 TileLang/Triton/FlashInfer cache 已热、只清空 DeepGEMM cache 的 run):DeepGEMM warmup 118 s, 130 s → 52 s, 52 s;time-to-ready 390 s, 381 s → 310 s, 320 s。(pod 首次 run 1333 s 因还支付其他 cache,排除在外。)
- 历史参照(DeepGEMM warmup 禁用,vLLM 0.33 nightly,eager,GB200):H100 TP8 出现 34 次 runtime 编译(21 fp8 GEMM、12 prenorm、1 indexer);GB200 TP4 78 次(52 fp8/fp4 GEMM、17 prenorm、9 einsum)。

## 验证与回归风险

- 环境:精确 merge-base wheel `0.34.0rc1.dev144+g68776b023` + 本 PR 文件(每个覆盖文件的 md5 与 merge-base 核对);专用 MLP pod(4 × GB200 SM100,152 SMs,nvcc 13.1;8 × H100 SM90,nvcc 13.0),前后无外来 GPU 进程;`VLLM_DEEP_GEMM_WARMUP=relax`、`VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0`、`--max-model-len 20480`。
- Kernel 级测试另含数值核对:对照 fp32 reference。
- Scope not covered(正文原列):`sm90_fp8_mqa_logits`(indexer)不 warm——SM90 上首次调用每 indexer shape 编译一次,其 key 无 M,且 `logits_dtype` 属于 key 但无法从 module 推断,猜错会 warm 一个永不运行的 kernel;无 spec-decode run,draft-model 改动仅由 source 与 unit tests 验证;prenorm warmup 改为跟随 `kernel_config.enable_jit_warmup`(与其他 mHC warmup 一致)而非 `VLLM_DEEP_GEMM_WARMUP`。

## 关联

- #41530(EngineDeadError 症状来源)。
- @adenzhou1350 在 B300 上独立复现 BLOCK_M=48/112 两个缺失 shape。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
