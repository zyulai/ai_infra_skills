# PR #64165: [Bugfix] Fix FlashInfer plan buffer race causing illegal memory access with PP

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64165](https://github.com/vllm-project/vllm/pull/64165) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-11-11 |
| labels | bug, nvidia, ready |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
FlashInfer plan() 在 GPU 未拷走上一步 plan 时覆写 page-locked buffer,PP 下内核越界读崩溃;重新 plan 前同步 CUDA event 修复。

## 问题现象
Qwen3.8-27B-FP8 + `--pipeline-parallel-size=3` + chunked prefill + prefix caching 下,首个长 cold prefill 触发 `CUDA error: an illegal memory access` 杀死引擎(#63505)。复现(3×A10 23 GB sm_86,vLLM `0.32.1rc1.dev505+g00b6c4c56`):基线 `00b6c4c56` 5/5 失败(`CHECK cold-prefill: FAILED HTTPError HTTP Error 500: Internal Server Error`,随后 EngineDeadError),本 PR 2/2 通过(27,423 prompt tokens,正确复述 Record 500);同服务器另外 7 个 seed 的 cold prompt(27,344 至 27,429 tokens)全部 ok,其中 3 个同时运行,decode-plan 路径也跑到;control:基线加 `--enforce-eager` ok。nightly 镜像里同一错误在 GDN kernel `chunk_delta_h.py` 处浮现(issue 中的表象)。PP=2 Qwen2.5-0.5B:基线 1/3 crash,本 PR 0/3。

## 根因
正文披露:FlashInfer `plan()` 把 plan(request、tile、merge indices)写进 wrapper 拥有的 page-locked host buffer,再用 `cudaMemcpyAsync` 异步拷到 GPU 且不等待;`FlashInferMetadataBuilder` 每步复用同一批 wrapper,若 host 在 GPU 执行上一次拷贝前再次 `plan()`,上一步的 attention kernel 会读到下一步的 plan,越界读 page indices 并 fault(与 issue 中的 Xid 31 `FAULT_PDE ACCESS_TYPE_VIRT_READ` 吻合)。为何此时才暴露:#57214(v0.31.0 引入,v0.30.0 没有)之前 `build()` 每步调用 `common_attn_metadata.seq_lens.cpu()`,阻塞拷贝顺带排空 stream;#57214 在无 speculative decoding 时改用 CPU 上界,不再排空。PP=3 时 scheduler 最多 3 步 in flight,race 要求 host 领先至少两个 plan,所以改动时序(eager、`CUDA_LAUNCH_BLOCKING=1`)会掩盖它。定位:CUDA core dump(`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`)指认 PP rank 0 上 `flashinfer::BatchPrefillWithPagedKVCacheKernel`(bf16 query, fp8 KV)发生 `CUDBG_EXCEPTION_WARP_MMU_FAULT`;debug 打点确认 767-token 末块的 plan 写入时 3136-token plan 的拷贝仍在排队,随后即 illegal memory access。

## 修复方案
`vllm/v1/attention/backends/flashinfer.py` 6 行改动:builder 构造 `self._plan_copied = torch.cuda.Event(blocking=True)`;`build()` 内当下一步 plan 之前 `self._plan_copied.synchronize()`,全部 plan 完成后 `self._plan_copied.record()`;record/synchronize 都只在 `needs_paged_kv_indices` 为真(即 `build()` 会调 wrapper 的 plan())时发生,覆盖 prefill、DCP prefill 与 decode(`fast_plan_decode`,也 plan CUDA-graph decode wrappers);TRTLLM/XQA 路径不调 plan(),既不等待也不 record。每个 builder 有自己的 wrappers 与 event;对从未 record 的 event synchronize 立即返回。做法对应 `gpu_model_runner.py` 的 `synchronize_input_prep`。成本:单 GPU async scheduling 0/2000 builds 阻塞;PP=2 时 69/1000 阻塞、每次最多 8.5 ms,等待不超过一个 plan step,小于 #57214 之前的全量 stream drain。新增测试 `test_flashinfer_plan_waits_for_previous_plan_copy`(prefill/decode 两参)无修复时失败(`assert [False, False, True] == [False, False, False]`);`test_flashinfer_avoids_seq_lens_copy_without_spec_tokens` 通过,证明未恢复 #57214 移除的 seq_lens 拷贝。

## 影响与回归风险
- 影响窗口:v0.31.0(#57214)起的 native FlashInfer 路径;issue 报告 v0.28.0 good、v0.31.0 bad。
- 回归面:`FlashInferMetadataBuilder.build()` 的三处 plan 调用引入条件阻塞等待(仅 host 领先多于一个 plan 时,恰为 race 可能发生的时机);未改动:MLA FlashInfer prefill wrappers(`vllm/v1/attention/backends/mla/prefill/flashinfer.py`)、sparse SM90 MLA backend、step5 sparse ops——它们同样在复用对象上调 plan() 但有各自 builder、未在本 PR 验证;MLA prefill 目前安全只因传 GPU indptr 给 plan() 触发 FlashInfer 内部阻塞拷贝,#52657 移除该拷贝后会暴露同样 race;issue 还报告 `--no-enable-prefix-caching` 下的 gibberish output,但 3×A10 上该配置在 cold prompt 处 OOM(与修复无关),本 PR 对其不做声明;未跑 accuracy eval 与 throughput benchmark。

## 关联
- [#63505](https://github.com/vllm-project/vllm/issues/63505)(修复的 issue)
- [#57214](https://github.com/vllm-project/vllm/pull/57214)(移除阻塞拷贝的回归源头)
- [#52657](https://github.com/vllm-project/vllm/pull/52657)(计划移除 MLA indptr 拷贝,将暴露同类 race)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
