# PR #21986: [AMD] Simplify fused allreduce + RMSNorm and remove hidden_dim allowlist

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hubertlu-tw |
| 合并 | 2026-04-12 |
| 分类 | Attention / Kernel |
| Label | documentation, amd, run-ci |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见改 communicator.py、parallel_state.py、基准脚本并扩展 AMD 注册测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/21986 |

## 一句话结论

简化 `--enable-aiter-allreduce-fusion` 路径:修边界 off-by-one、删除 Python 侧 hidden_dim 白名单改由 AITER C++ dispatch 决定,Qwen3.5-397B TP4 上总吞吐 +3.7%、TPOT −3.7%。

## 动机

两个问题:(1) `communicator.py` 的激活门用 `<` 比较 byte-size 阈值,而 AITER 的 `should_custom_ar` 用 `<=`(inp_size <= max_size/2)——hidden_size=4096、bf16、8192 tokens 时 total_bytes 恰好等于阈值 67,108,864,`<` 错误地拒绝;(2) `parallel_state.py` 的 1-stage vs 2-stage 选择硬编码 `hidden_dim ∈ {512, 1024, 2048, 4096, 2880}` 白名单,与 AITER C++ 侧的判定(#2453 后为 `n % pack_size == 0 && n/pack_size <= 1024`)重复,新模型每加一个 hidden_dim 都要改 Python(#21947 给 2880 加白名单即是例子),维护负担重。AITER C++ dispatch 本就决定哪些 hidden_dim 有 1-stage 支持,不支持的 dim 传 `use_1stage=True` 会被安全地覆写为 2-stage。

## 优化点

两处门控修改:激活门改为 `total_bytes <= 8*1024*8192`,与 AITER 边界一致;1-stage 选择删除 hidden_dim 集合判断,只保留 128 KB byte-size 阈值——1-stage kernel 每 token 一个 block、被 `kMaxBlocks=80` 硬封顶(hidden_dim=4096 bf16 时 80x4096x2=655,360 字节),128 KB 阈值确保只有 decode 类小 batch 走 1-stage,大 prefill batch 自动落到 2-stage。同时删除了 `SGLANG_ENABLE_DETERMINISTIC_INFERENCE` 强制 1-stage 的分支(简化策略,仅 env 显式覆盖优先)。依赖上游 AITER 的两个 PR:#2586(修 1-stage 精度——f32 累加后在 residual 加法前才舍入 bf16,与 unfused 路径的先舍入不一致,1-ULP 逐层放大导致 −2.6pp GSM8K 回归)与 #2453(allreduce dispatch 重构,1-stage 接受任何 pack_size 倍数的 hidden_dim,2-stage 用 n_packs 分发去掉 n_bytes%1024 对齐要求)。测试新增 `_run_residual_accuracy_check`(多 GPU torchrun,验证 fused residual 输出与 unfused 路径 bit-identical)与 `test_fused_ar_rms_multi_hidden_dim`(2880/4096/5120/6144/7168/8192 全部 shape fused available)。

## 关键改动

```text
python/sglang/srt/layers/communicator.py: 激活门 < 改 <=,对齐 AITER should_custom_ar
python/sglang/srt/distributed/parallel_state.py: 删 hidden_dim 白名单与 deterministic 分支,保留 128KB 阈值
test/registered/ops/test_aiter_allreduce_fusion_amd.py: 新增 residual bit-exact 与 multi-hidden-dim 测试
benchmark/kernels/all_reduce/benchmark_fused_ar_rms_amd.py: 默认形状加入 2880
```

简化后的选择逻辑:

```python
if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
    use_1stage_ar = envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
else:
    total_bytes = input_.numel() * input_.element_size()
    use_1stage_ar = total_bytes <= 128 * 1024
```

## 性能声明

条件:Qwen3.5-397B-A17B-FP8(60 层,hidden=4096),4x MI355X TP4 全互联,`SGLANG_USE_AITER=1` + `--attention-backend aiter`,AITER main a530b92bf(含 #2586 与 #2453)。Serving benchmark(random 8192-in/1024-out、concurrency=1、8 prompts):fusion ON vs OFF,总吞吐 766.4 vs 739.1 tok/s(**+3.7%**)、输出吞吐 85.2 vs 82.1 tok/s(+3.7%)、median E2E 12,019 vs 12,476 ms(−3.7%)、median TPOT 11.46 vs 11.90 ms(−3.7%)、TTFT 基本持平(+0.8%)。内核微基准(TP4 bf16):decode(graph-captured)所有 hidden_dim 提速 1.3-2.2x(如 16x2880 为 2.10x、1x7168 为 2.24x),GPT-OSS 的 hidden_dim=2880 decode 最高 2.10x;prefill(eager)128-token 形状提速 1.27-1.49x。Trace 验证:阈值修复后 EXTEND trace 上 121 处 allreduce 有 119 处走 fused 路径(`cross_device_reduce_2stage` 121→2、`add_rmsnorm_quant_kernel` 120→1、fused 两步 kernel 各新增 119),其余 2 处为 64 宽小 hidden_dim 层低于 AITER 内部阈值。

## 验证与回归风险

精度:GSM8K(1319 题、5-shot、parallel=1319)三轮,fusion ON 均值 94.7% vs baseline 92.4%(+2.3pp),无回归,确认 #2586 精度修复正确集成。新增测试:`_run_residual_accuracy_check` 覆盖 6 个 hidden_dim x 7 个 token 数的 1-stage/2-stage 残加输出 bit-exact(回归 #2586);`test_fused_ar_rms_multi_hidden_dim`(TP=4,stage-c-test-large-8-gpu-amd suite)断言所有 shape fused available;8-GPU 基准测试保留含 oversized 回退断言。风险点:删除 deterministic inference 强制 1-stage 的分支,需要逐位可复现输出的部署失去了该保证;Python 不再预判 hidden_dim,unsupported dim 依赖 AITER 内部覆写行为(若有 AITER 版本差异会直接落到 2-stage 而非报错);激活门放宽到 `<=` 使 8192-token bf16 4096 hidden 恰好进入融合路径。

## 关联

同区域优化:上游 ROCm/aiter #2586(1-stage 精度修复)与 #2453(dispatch 重构,硬依赖);sglang PR #21947(给 2880 加白名单,被本 PR subsume);`--enable-aiter-allreduce-fusion` 与 `SGLANG_USE_1STAGE_ALLREDUCE`、`SGLANG_USE_AITER` 环境变量体系。PR 正文未列出其他 sglang 侧关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
