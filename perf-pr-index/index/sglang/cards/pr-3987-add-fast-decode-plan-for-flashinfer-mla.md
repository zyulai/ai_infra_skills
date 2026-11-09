# PR #3987: Add fast decode plan for flashinfer mla

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Fridge003 |
| 合并 | 2025-03-03 |
| 分类 | Attention / Kernel(tier: major) |
| Label | high priority |
| Diff | diffstat 缺失(素材未统计行数);diff 覆盖 attention backend、schedule_batch、cuda_graph_runner、forward_batch_info 等文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/3987 |

## 一句话结论

为 flashinfer MLA decode 新增 fast plan,跳过 cuda graph replay 中 plan 的 D2H 传输/同步,长输出场景吞吐提升 1%-2%。

## 动机

flashinfer MLA backend 与 cuda graph 一起使用时,`BatchMLAPagedAttentionWrapper.plan` 内部需要在 CPU/GPU 之间传输 indptr 张量,导致 graph replay 直接 hang。另外 #3906、#3917 报告该后端在长输出场景下表现不如 triton backend,本 PR 希望一并解决该问题。

## 优化点

在 `ScheduleBatch.get_model_worker_batch` 中,decode 分支提前把 `seq_lens` 拷贝到 CPU 作为 `decode_seq_lens`,经 `ForwardBatch.decode_seq_lens_cpu` 传入 cuda graph replay。replay 时用这份 CPU seq_lens 在 CPU 侧 cumsum 出 `kv_indptr` 与 `kv_len_arr`,再调用新写的 `fast_mla_decode_plan`:它绕过原 `plan` 中触发流同步与 GPU->CPU 拷贝的路径,直接以 CPU 张量调用 `_cached_module.plan` 完成 workspace 规划,从而消除 replay 前等待 memcpyAsync 的时间。同时把各 attention backend 的 `init_forward_metadata_capture/replay_cuda_graph` 签名调整为 `**kwargs` 透传,新增 `init_metadata_replay` 标志区分 capture 与 replay,使 replay 路径复用预分配的 `fast_decode_kwargs`(含预建的 `kv_indices` 缓冲)而不再临时分配。

## 关键改动

```text
python/sglang/srt/layers/attention/flashinfer_mla_backend.py: 新增 fast_mla_decode_plan,replay 路径改用 CPU indptr 做 plan;新增 cuda_graph_*_cpu 缓冲与 fast_decode_kwargs
python/sglang/srt/managers/schedule_batch.py: decode 时提前将 seq_lens 拷贝到 CPU(decode_seq_lens 字段)
python/sglang/srt/model_executor/cuda_graph_runner.py: 维护 seq_lens_cpu 缓冲并在 replay 时传入 metadata 更新
python/sglang/srt/model_executor/forward_batch_info.py: ForwardBatch 增加 decode_seq_lens_cpu 字段
python/sglang/srt/layers/attention/{triton,flashinfer}_backend.py: 适配新接口签名(kwargs 透传)
```

```python
def fast_mla_decode_plan(self, qo_indptr_cpu, kv_indptr_cpu, kv_indices,
                         kv_len_arr_cpu, num_heads, ...):
    """A faster version of BatchMLAPagedAttentionWrapper::plan,
    for skipping the stream synchronization in original plan function
    during cuda graph replaying."""
    self._causal = causal
    self._page_size = page_size
    self._sm_scale = sm_scale
    with self.device as device:
        stream = torch.cuda.current_stream(device).cuda_stream
        self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_cpu, kv_indptr_cpu, kv_len_arr_cpu, ...)
```

## 性能声明

DeepSeek-V2-Lite,TP 8,NVIDIA H200,flashinfer mla 开启,长输出 workload 各跑 5 次取平均吞吐(正文总结:提升 1% 到 2%):

- Input-4096-Output-2048(60 prompts,同 #3917):Decode 4038.21 tok/s(PR 前 3925.06,Triton 3416.69);Prefill 7757.64(PR 前 7540.28)
- Input-180-Output-400(600 prompts,request-rate 40,max-concurrency 40,同 #3906):Decode 3532.19(PR 前 3480.68,Triton 3413.85)
- Input-100-Output-2000(request-rate 2,120 prompts):Decode 1751.71(PR 前 1742.90)

torch profiler 显示等待 memcpyAsync 的时间被消除;正文指出 MLA with weight absorption 是 compute bound、GPU 已打满,故对 e2e 吞吐影响不显著。

准确性:DeepSeek-R1(TP 8)GSM8K 8-shot accuracy 0.956(invalid 0.000),MMLU accuracy 0.871。

## 验证与回归风险

无新增单元测试(Checklist 中未勾选),依赖上述 GSM8K/MMLU 手动基准。风险点:attention backend 的 `init_forward_metadata_capture/replay_cuda_graph` 公共签名被改为 kwargs 透传,triton/flashinfer backend 同步修改,其他自定义 backend 需要适配;decode 路径新增每 batch 一次 `seq_lens.cpu()` 拷贝,属于所有 decode 请求的公共路径;`SpecInfo`/`encoder_lens` 传参方式变化可能影响 speculative decoding 与 encoder-decoder 路径。

## 关联

前序问题报告:#3906、#3917(flashinfer MLA 长输出性能差于 triton)。同分类其他 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
