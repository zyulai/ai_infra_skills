# PR #27689: [Perf] FlashInfer MLA: remove blocking D2H in spec-decode plan

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @elvischenv |
| 合并 | 2026-08-12 |
| 分类 | Speculative Decoding(classify.py 给出,tier: major) |
| Label | performance, deepseek, run-ci |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 仅 1 个文件(flashinfer_mla_backend.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/27689 |

## 一句话结论

消除 FlashInfer MLA 投机解码 plan() 中每次 verify/draft-extend 的阻塞 D2H 拷贝,H200 上 DeepSeek-R1+MTP 吞吐从与标准解码持平恢复到 +26.7%。

## 动机

H200 上 DeepSeek-R1 开 MTP/EAGLE(steps=2/topk=1/draft=3)竟不比标准解码快。nsys 显示 target-verify 与 draft-extend 的每次 attention `plan()` 有 3 次同步 device→host 拷贝,阻塞启动下一个 CUDA graph 的 host 线程,破坏 overlap 调度,stall 抵消了投机收益。根因:`BatchMLAPagedAttentionWrapper.plan()` 在收到 GPU 张量时用阻塞 `.to("cpu")` 拷贝 `qo_indptr`/`kv_indptr`/`kv_len_arr`;普通 decode 路径早已通过 `fast_mla_decode_plan`(传预计算的 CPU 张量)规避,但 spec-decode 的 verify 与 draft-extend 路径仍在用 stock plan。

## 优化点

仅改 `flashinfer_mla_backend.py`:

- `__init__` 分配 pinned host 缓冲 `fast_plan_{qo_indptr,kv_indptr,kv_len_arr}_cpu`(支持 prefill 的 backend 才分配)。
- Target-verify(cuda graph):新增 `fast_mla_prefill_plan`,作为 `fast_mla_decode_plan` 的 prefill 兄弟实现(与已合并 #28854 的 `fast_prefill_plan` 同构):把 host 侧已知 indptr/长度直接交给 `_cached_module.plan`(replay 时零 D2H),并按 stock plan 的 use_cuda_graph 分支方式 non-blocking 刷新 cuda-graph 缓冲;capture 完成真实 `plan()` 后安装到 verify wrapper。host 数组由 `seq_lens_cpu` + 固定 `num_draft_tokens` 树形状构建并 assert 校验(sync-free plan 无 device-readback fallback,错即 fail-fast)。
- Eager `DRAFT_EXTEND_V2`:`init_forward_metadata` 构建同样的 host 数组传入 `plan()`,使其内部 `.to("cpu")` 变成 no-op(flashinfer-MLA 无 draft-extend cuda graph,该路径是 eager)。
- `kv_indices` 全程留在 GPU;DCP(dcp_kv_indptr/dcp_kv_indices 存在)时 host 数组与分片不匹配,自动回退 GPU 数组。decode、普通 prefill、ragged 路径不变。

作者特别说明:该方案不消除 `seq_lens_cpu` 依赖——FlashInfer-MLA 的 `plan()` 是 host-side(tile schedule 在 CPU 构建),彻底移除需 GPU 侧 planning(如 #26824 对 trtllm_mla 的做法),属单独的 backend 级改动;残余每迭代一次 `seq_lens_cpu` 拷贝是 host-planned attention + 数据依赖投机的固有成本。

## 关键改动

```text
python/sglang/srt/layers/attention/flashinfer_mla_backend.py: pinned host 缓冲、fast_mla_prefill_plan、verify/draft-extend 路径接入
```

```python
def fast_mla_prefill_plan(self, qo_indptr_cpu, kv_indptr_cpu, kv_indices,
                          kv_len_arr_cpu, num_heads, head_dim_ckv, head_dim_kpe,
                          page_size, causal, sm_scale, q_data_type, kv_data_type):
    self._qo_indptr_buf.copy_(qo_indptr_cpu, non_blocking=True)
    self._kv_indptr_buf.copy_(kv_indptr_cpu, non_blocking=True)
    self._kv_indices_buf[: len(kv_indices)].copy_(kv_indices, non_blocking=True)
    self._kv_len_arr_buf.copy_(kv_len_arr_cpu, non_blocking=True)
    self._cached_module.plan(..., qo_indptr_cpu, kv_indptr_cpu, kv_len_arr_cpu, ...)
```

## 性能声明

DeepSeek-R1-0528 FP8,TP8,H200,ISL/OSL=1024,并发 8,无 profiler 干净运行:Total token throughput——main+STP 1320.6 tok/s、main+MTP(未修复)1319.8、本 PR+MTP **1671.9(+26.7%,对比 MTP-no-fix 与 STP 均是)**;median TPOT 11.87→9.35 ms;avg_spec_accept_length 2.253→2.257(不变,收益为纯 overlap 恢复)。nsys(profiled 窗口,bs=8)前后对比:`cudaStreamSynchronize` 1120→160 次,`qo`+`kv_indptr` 的 36B D2H 633→0 次;残余约 160 次为固有的每迭代 `seq_lens_cpu` 拷贝。精度:GSM8K(lm_eval local-completions greedy,同配置)flexible-extract 0.954±0.006、strict-match 0.951±0.006。

## 验证与回归风险

投机解码本身无损(target 验证每个 draft),本 PR 只改 attention metadata 到达 `plan()` 的方式,不改模型输出(accept length 不变佐证)。Checklist 显示未新增单元测试、未更新文档;CI 快照 Base/Extra 均失败(以合并时为准)。风险:target-verify cuda graph replay 要求 `seq_lens_cpu` 为 host-resident(有 assert fail-fast);DCP 路径回退 GPU 数组的行为依赖 `attn_dcp_metadata` 判断顺序;与 #28854 的 `fast_prefill_plan` 同构,后续重构需保持两者一致。

## 关联

同区域:#28854(normal prefill 的 fast_prefill_plan,本 PR 引用其模式)、#26824(trtllm_mla 的 `needs_cpu_seq_lens=False` GPU 侧 planning)、#21452/#17706(PCG/FlashInfer 元数据线)。PR 正文未列出其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
