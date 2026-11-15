# PR #63483: [Core][DBO] Dual batch overlap under pure tensor parallelism (split PR of M3.1)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benenzhu |
| 合并 | 2026-11-14 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, rocm, ready, mrv2 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/63483 |

## 一句话结论

M3.1(#63232)的 engine 侧拆分 PR:Model Runner V2 上让 DBO 支持纯 TP(一个 DP rank、无 EP)——每个 microbatch 的 TP all-reduce 在 comm stream 上与另一 microbatch 的 compute 重叠;MI355X TP4 上 16K/32K 单请求 prefill TTFT −35%/−39%,8 并发 16K prompt 吞吐 +61%。

## 动机

M3.1 的 ROCm 工作中 engine 部分单独拆出评审合并(model 本体已随 #64111 进 main)。此前 DBO 只覆盖 DP+EP(MoE all-to-all 重叠);纯 TP 部署下唯一可重叠的通信是 TP all-reduce。Related:#44677 在 Model Runner V1 上做过同一目标(其代码被 #63373 移除)。

## 优化点

1. `--enable-dbo` + `--tensor-parallel-size N`(N>1)且无 DP/EP 在 MRV2(`VLLM_USE_V2_MODEL_RUNNER=1`)可用:超过 `--dbo-prefill-token-threshold` 的 step 拆成两个 microbatch,每个 TP all-reduce 变成 comm stream 上的一次 microbatch hand-off。新增 `dbo_schedule` 配置:纯 TP 默认 `all-reduce-overlap`,其余默认 `dispatch-combine-overlap`;后者在非纯 TP 下显式拒绝。
2. **All-reduce hand-off**:`init_dbo_tp_comms` 为每个 microbatch 建一个 `PyNcclCommunicator` 并在 `GroupCoordinator._all_reduce_out_place` 装 hook(运行在 `vllm.all_reduce` custom op 之下,编译图也能到达);microbatch 线程内先 yield 给对方、在 comm stream 上用本 microbatch 的 communicator 跑 all-reduce 再切回,其余情况走原路径。每 microbatch 一个 communicator 使两个 microbatch 的 collective 不挤同一 NCCL queue(`pynccl.bind_stream` / `_comm_for`)。
3. **拆分决策**:单 DP rank 时 `dispatch_cg_and_sync_dp` 本地决策,split step 以 eager(`CUDAGraphMode.NONE`)运行;`--dbo-decode-token-threshold` 高于 decode batch 时 decode step 保留 CUDA graphs。
4. **Fused all-reduce + norm**:`fused_allreduce_gemma_rms_norm` 在 microbatch 内跳过 FlashInfer/AITER fused kernel,保证每次 all-reduce 都是 hand-off;未拆分 step 保留 fusion。
5. **MoE**:`MoEPrepareAndFinalizeNoDPEPModular.supports_dbo()` 返回 True——无 all2all,hand-off 就是 TP all-reduce;`supports_dbo` 语义收窄到 `dispatch-combine-overlap`。
6. **Microbatch 输出**:`merge_ubatch_outputs` 透传 `None`(DSpark 的 auxiliary 输出需要);两个 microbatch 同一个 stream 上 compute,workspace manager 保持一套(`num_ubatches=1`)。
7. **ROCm comm stream 优先级**:comm stream 取高优先级(-1)——默认优先级下 comm 与 compute 可能共享同一硬件 queue 而串行化。
8. **hipBLASLt hang 修复(ROCm)**:`hipblasLtCreate` 会分配 Stream-K flag buffer 并用 null-stream `hipMemset` 延迟 3–33 ms 清零;microbatch 线程每步都是新线程会新建 handle,当 comm stream 上的 Stream-K GEMM 先于迟到的清零启动时,清零会抹掉 partner workgroups 已写的 flag,owner 在 `label_SK_Fixup` 永久自旋(rocgdb 定位)。修复:`UBatchRunner` init 时填满 handle pool——每个 microbatch 线程在 compute/capture/comm stream 上各跑一个小 GEMM。
9. Multimodal 模型在该模式允许(vision encoder 不参与拆分);encoder-decoder 仍不支持;MRV1 以显式 error 拒绝该模式。

## 关键改动

```text
vllm/config/parallel.py: DBOSchedule 类型 + dbo_schedule 默认/校验(非纯 TP 拒绝 all-reduce-overlap)
vllm/config/vllm.py: all-reduce-overlap 跳过 a2a backend 检查;multimodal 限制只对非 all-reduce-overlap
vllm/distributed/device_communicators/pynccl.py: bind_stream / _comm_for —— 每 stream 独立 NCCL communicator
vllm/distributed/parallel_state.py: _all_reduce_out_place 在 _TP 上执行 microbatch hand-off
vllm/engine/arg_utils.py: --dbo-schedule CLI
vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py: microbatch 内跳过 fused AR+norm
vllm/model_executor/layers/fused_moe/modular_kernel.py: dbo_enabled("dispatch-combine-overlap") 门控
vllm/platforms/interface.py / rocm.py: prepare_ubatch_streams 钩子;ROCm 预填 hipBLASLt handle pool
vllm/v1/worker/gpu/{dp_utils,ubatch_utils,gpu_worker}.py / ubatching.py / workspace.py: dp_size==1 本地拆分、comm stream 高优先级、单 workspace 集
tests/v1/worker/test_dbo_pure_tp.py(新增)/ docs/design/dbo.md: 测试与 all-reduce-overlap 文档
```

(注:pack 的 diff 截断,有 2 个文件未展示,以上仅覆盖已展示部分。)

```python
        from vllm.v1.worker.ubatching import (
            dbo_switch_to_comm_sync,
            dbo_yield_and_switch_from_comm_to_compute,
        )

        # Launch before yielding: the comm stream then waits on a compute event
        # recorded after every read of whichever freed block the output reuses.
        dbo_switch_to_comm_sync("all-reduce-overlap")
        out = self.device_communicator.all_reduce(input_)
        dbo_yield_and_switch_from_comm_to_compute("all-reduce-overlap")
        return out
```

## 性能声明

- 单元:`pytest tests/v1/worker/test_dbo_pure_tp.py test_gpu_dp_utils.py test_gpu_ubatch_slicing.py test_workspace.py test_ubatch_offsets.py tests/test_config.py::test_dbo_thresholds_cover_a_request_per_microbatch`:在 main `36ed4327d2` 合并后 **85 passed**(MI355X,ROCm image + 本树挂 `PYTHONPATH`);`pre-commit run --files <changed files>` 全部 hook 通过。
- **E2E(MI355X,TP4,M3 + MXFP4 experts,agentic multi-turn trace,3600 s,每点一 run;在携带本改动的 M3 serving branch 上,`--dbo-prefill-token-threshold 8192 --dbo-decode-token-threshold 32768`)**:

| concurrency | DBO | interactivity P90 (tok/s/user) | TTFT P90 (s) | total tok/s/GPU |
|---|---|---|---|---|
| 32 | off | 75.4 | 1.33 | 44573 |
| 32 | on | 73.7 | 1.11 | 44009 |
| 64 | off | 23.6 | 5.23 | 53692 |
| 64 | on | 30.8 (+31%) | 3.21 | 59281 (+10%) |

- **E2E(公开模型,本 branch 合并 main `36ed4327d2`,DBO on)**:MiniMax-M3.1-Flash(`amd/MiniMax-M3.1-Flash-Quark-MXFP4`),TP4 MI355X,InferenceX 镜像 `vllm/vllm-openai-rocm:minimax-m31-flash`(`sha256:ae868121`),AgentX recipe c16/c32 lane(NVFP4 KV,`--dbo-prefill-token-threshold 16384 --dbo-decode-token-threshold 32768`,max-num-seqs 2 × concurrency),DSpark 3 与 7(InferenceX 合成 acceptance lengths 2.88 / 4.04),AgentX trace 前 20 分钟,每点一 run;每台 server 均记录 DBO on、per-microbatch TP communicator 与 comm-stream all-reduce hook;无 request 因 server error 失败(每点 1–2 个返回无内容)。
- **Serving TTFT, M3.1-Flash TP4**(命令 + `--max-num-batched-tokens 32768`,V2 runner,单请求,3 次中位数):

| prompt | base | DBO |
|---|---|---|
| 16K | 468 ms | 304 ms (-35%) |
| 32K | 952 ms | 582 ms (-39%) |
| 64K | 1930 ms | 1545 ms (-20%) |
| 8 concurrent 16K prompts | 35.0K tok/s | 56.4K (+61%) |

## 验证与回归风险

- 局限(正文原列):split step 不带 CUDA graphs(eager 运行)。
- C32 时 DBO 略差(interactivity P90 73.7 vs 75.4;tok/s/GPU 44009 vs 44573),C64 才显著为正——阈值由 `--dbo-prefill/decode-token-threshold` 控制。
- microbatch 内禁用 fused all-reduce+norm fusion(仅未拆分 step 保留);hipBLASLt workaround 标注 TODO,待 `hipblasLtCreate` 不再发 null-stream memset 后移除。
- microbatch 线程失败时先记日志再 join,避免兄弟线程在 hand-off 上永久等待(diff 中新增 `logger.exception`)。

## 关联

- #63232(M3.1 原始 PR,本 PR 为其 engine 侧拆分)、#64111(M3.1 model 本体)。
- #44677(MRV1 上的同一目标)、#63373(移除 MRV1)、#61464(DBO 下的 speculative decoding 与 tuple 输出合并)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
