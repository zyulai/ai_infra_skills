# PR #53196: [Kimi K3][Perf] Export FlashKDA checkpoints in one packed call

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ZeldaHuang |
| 合并 | 2026-08-21 |
| 分类 | KV Cache / Memory |
| Label | ci/build, kimi, k3, scheduler |
| Diff | diffstat 缺失(pack 未提供 +/− 统计,仅含源码 diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/53196 |

## 一句话结论

FlashKDA prefill checkpoint 导出从 per-request 拆分调用改为单次 packed 调用,kernel 快 22.62x、TTFT −10.7%。

## 动机

#52789 的 checkpoint 导出把每条请求在其 checkpoint 位置拆成两段、分别调用 FlashKDA;64 请求且 tail 非空的 batch 要发起 128 次顺序 FlashKDA API 调用。FlashKDA#7 已在 forward API 中加入 `checkpoint_state`/`checkpoint_offsets`,同样的操作可以表达为一次 packed 调用。

## 优化点

保留 #52789 的 checkpoint block 分配、生命周期与 cache-store 逻辑,改变 FlashKDA 产出方式:一次 packed 调用中 kernel 在处理整个 batch 时同时写出 output、final state 与中间 recurrent checkpoint state,随后的 Triton store kernel 照旧把 recurrent/conv checkpoint 写入 request-owned cache block。约束处理:checkpoint offsets 必须满足 FlashKDA 的 16-token chunk 对齐,metadata 构建把该条件加进 validity predicate,不满足的 offset 直接省略(fail-closed,如 offset 47)而非传给 kernel;cache manager 仅在 request start 为 Mamba-block aligned 时才分配内部 checkpoint block,否则 checkpoint slot 保持真实 null block、不接收 cache hash。scheduler 侧的对齐改进明确留给 follow-up。本 PR 直接 stacked on #52789,依赖 FlashKDA#7(pin 更新至 `ee0be888`),与 #52971(更广的 standalone checkpoint 实现)不重复。

## 关键改动

```text
csrc/flashkda_registration.cpp: _flashkda_C forward schema 增加 checkpoint_state/checkpoint_offsets
cmake/external_projects/flashkda.cmake: FlashKDA pin 升至含 #7 的 merge commit
vllm/models/kimi_k3/nvidia/kda.py: 删除 per-request split 循环,改单次 packed _flashkda_prefill
vllm/models/kimi_k3/nvidia/kda_metadata.py: KDACheckpointMetadata 改为 checkpoint_offsets;validity 加 16-token 对齐
```

```python
flashkda(
    packed_batch,
    cu_seqlens=cu_seqlens,
    checkpoint_state=checkpoint_state,
    checkpoint_offsets=checkpoint_offsets,
)
```

## 性能声明

- Microbench(NVIDIA GB300、CUDA 13.0、BF16、head dim 128、12 heads、64 sequences,CUDA events,10 warmup / 9 trials 中位数):64×512 tokens(checkpoint 384, tail 128):split loop(128 次调用)8.4425 ms → packed 单次 0.3733 ms(**22.62x**,kernel 延迟 −95.58%;无 checkpoint 的 packed baseline 0.3587 ms,checkpoint 导出开销 +4.08%);64×2048 tokens(checkpoint 1536):10.1720→1.3040 ms(**7.80x**,−87.18%,开销 +2.17%)。split 与 packed 的输出、final state、checkpoint state 最大绝对差均为 0.0。
- E2E(Kimi K3,TP8,两台 GB300 节点,32 并发,15,360 token 共享前缀 + 每 request 2,048 token 独立后缀,各 3 次):P50 TTFT 2,121.29→1,893.82 ms(**−10.72%**),request throughput 10.239→11.229 req/s(+9.66%),input throughput 178,246→195,466 tok/s。

## 验证与回归风险

4 个测试通过(`test_flashkda_correctness`、metadata 的 aligned/unaligned 两测、prefix cache block-aligned 要求),另有 partial-prefix lifecycle 与 Mamba aligned-split CPU 套件 68 passed;pre-commit 全过;GB300 Slurm job 复现。风险:依赖外部 FlashKDA pin 升级;unaligned offset 走 fail-closed(放弃 checkpoint 导出),该路径行为依赖 scheduler 对齐 follow-up 改善;作为 stacked PR,其 diff 相对 #52789,单独回溯时需连前驱一起看。

## 关联

#52789(直接前驱,stacked)、FlashKDA#7(依赖的上游 PR)、#52971(不重复的更广 checkpoint 实现)。同批 Kimi-K3 方向:#52606。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
