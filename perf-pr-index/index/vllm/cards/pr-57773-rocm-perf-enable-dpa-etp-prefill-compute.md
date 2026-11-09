# PR #57773: [ROCm][Perf] Enable DPA+ETP prefill compute/comm dual-batch-overlap for DeepSeek-V4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shen-shanshan |
| 合并 | 2026-10-15 |
| 分类 | MoE (tier: major) |
| Label | rocm, deepseek, DSv4 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),实际改动覆盖 MoE prepare/finalize、ubatch 基础设施、DSv4 AMD 模型与测试 |
| 链接 | https://github.com/vllm-project/vllm/pull/57773 |

## 一句话结论

让 ROCm 上 DP 部署 DeepSeek-V4 的 prefill 走 compute/comm 双流 DBO:高并发 TTFT 降约 4-8%,agentic 吞吐 +6.97%,并修复此前 DBO 约 30% 回归的根因。

## 动机

`--enable-dbo` 把一个 step 拆成两个 ubatch,以两线程 ping-pong 跑在 compute stream 和 comm stream 上,但只有 DeepEP-HT 的 prepare/finalize 会把 collective 移到 comm stream;不带 `--enable-expert-parallel` 的 DP 部署(`allgather_reducescatter` naive DP/EP 路径)的 MoE all-gather(dispatch)与 reduce-scatter(combine)同步跑在 compute stream 上,两个 ubatch 因此串行,DBO 在 prefill 上毫无收益。该路径在 ROCm 上还被 microbatching 的 config 检查直接拒绝。

## 优化点

三层改动。其一,comm overlap:`MoEPrepareAndFinalizeNaiveDPEPModular` 的 `dispatch()`/`combine_into_output()` 包进新的 `_comm_region()` 上下文——DBO 激活时先 yield CPU 给对端 ubatch,切到 comm stream 跑 collective,退出时同步回 compute stream(与 DeepEP-HT 相同的 handoff);非 DBO 与 cudagraph capture 下为 no-op,非 ROCm 平台行为不变(`supports_dbo()` 仅在 ROCm 返回 True)。配套在 `FusedMoEPrepareAndFinalize` 新增 `supports_dbo()` 能力(默认跟随 `supports_async()`,避免重载会同时驱动 shared-expert overlap 的 `supports_async()`)。其二,修 DBO 正确性/性能问题:ubatch 线程补 `torch.accelerator.set_device_index(self.device)`——此前每个线程首个 stream 建在 GPU 0 上,所有 DP rank 把工作压到 rank 0,这是此前 DBO 约 30% 吞吐回归的主因;ubatch 切片边界显式转 `int`(numpy 标量泄漏导致 Triton 特化失败);`_make_metadata_with_slice` 补切 `positions`(C128A metadata 必需);新增 per-ubatch `token_offset` 并让 DSv4 AMD 的 `_mtp_hidden_buffer` 按偏移写入(修复两个 ubatch 线程从 row 0 互踩,MTP drafter 读错位)。其三,ROCm config 放行 `allgather_reducescatter` 与 microbatching 组合。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py: _comm_region 上下文把 dispatch/combine 移到 comm stream,supports_dbo() 仅 ROCm
vllm/model_executor/layers/fused_moe/modular_kernel.py: 新增 supports_dbo() 能力,sync 路径 assert 放行 supports_dbo 的实现
vllm/v1/worker/gpu_ubatch_wrapper.py / ubatching.py / ubatch_utils.py: ubatch 线程绑定设备、token_offset 机制、切片边界转 int、positions 切片
vllm/models/deepseek_v4/amd/model.py: _mtp_hidden_buffer 按 dbo_current_ubatch_token_offset() 写入
vllm/config/vllm.py: ROCm 允许 allgather_reducescatter 与 microbatching
```

关键 handoff 代码:

```python
if not dbo_enabled() or torch.cuda.is_current_stream_capturing():
    yield
    return
dbo_yield_and_switch_from_compute_to_comm()
try:
    yield
finally:
    dbo_switch_to_compute_sync()
```

## 性能声明

硬件:DeepSeek-V4-Pro FP4,8x MI350X,DP8(TP1),`--moe-backend aiter`,MTP(3 speculative tokens),prefix caching,INT4 quick reduce。固定长度 prefill(SA InferenceX `8k1k`,OSL=1):conc=64 时 Mean TTFT 5858.52 → 5477.98 ms(-6.49%),P99 TTFT -6.87%;conc=128 Mean TTFT -7.52%;conc=16/4/1 反而 +3.03%/+1.10%/+8.41%(小 batch 隐藏不了 collective,建议调高 `--dbo-prefill-token-threshold`)。逐层 trace:conc=64 prefill 快 9.25%/层。Agentic replay(conc=64,平均约 97K prompt / 620 output tokens,729 vs 766 请求):输出吞吐 703.40 → 752.46 tok/s(+6.97%),Mean TTFT -2.85%,Mean TPOT 44.69 → 43.15 ms(-3.45%)。作者提示读作 parity 或温和收益而非决定性加速(仅少量 step 被拆分)。精度:gsm8k 5-shot,flexible-extract 0.9492 → 0.9530,strict-match 0.9500 → 0.9538。

## 验证与回归风险

新增 `tests/kernels/moe/test_naive_dp_ep_dbo.py`(_comm_region handoff、非 DBO no-op、graph capture 跳过、出错恢复 compute stream、supports_dbo 仅 ROCm、默认实现跟随 supports_async 的契约)与 `tests/v1/worker/test_ubatch_offsets.py`(per-thread token offset、切片边界为 Python int),全部通过。风险面:同步 prepare/finalize 路径的 assert 放宽依赖 `supports_dbo()` 正确申报;`supports_async()` 语义保持不变以避免影响 MK 内部 shared-expert overlap 调度;decode 因 `--dbo-decode-token-threshold 8192` 显式留在 FULL cudagraph 路径(microbatched decode 每 token 约慢 1.8x);CUDA 等其他平台 config 检查仍拒绝该组合,行为不变。

## 关联

同区域:DBO/ubatching 基础设施(见索引中 DBO 相关条目)、DeepSeek-V4 AMD 支持见 pr-61837(Step 5 preview on ROCm)、pr-51473(DSV4 ROCm MXFP4);ROCm MoE 路径见 pr-59478(AITER FlyDSL INT4 quick reduce,本次基准即用 INT4 quantization)。同分类 MoE 其他 PR 见索引。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
