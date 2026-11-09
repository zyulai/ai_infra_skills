# PR #28782: [Spec] Support FlashInfer CUDA graph for EAGLE draft-extend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-06-21 |
| 分类 | Speculative Decoding |
| Label | run-ci, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/28782 |

## 一句话结论

EAGLE draft-extend 在 FlashInfer 后端下改为捕获 CUDA graph 回放,在线 median E2E 延迟降低约 10%。

## 动机

EAGLE v2 的 draft-extend graph 白名单此前只含 TritonAttnBackend、TRTLLMMLABackend、TokenspeedMLABackend 等,使用 FlashInfer attention 后端时 draft-extend forward 每层 eager 发射 kernel。draft-extend 是投机解码每个 decode step 都要走的路径,图捕获收益直接。

## 优化点

三处改动:(1) graph 捕获——在 `flashinfer_backend.py` 的 `_prepare_cuda_graph_metadata` 与 `init_forward_metadata_out_graph` 增加 `is_draft_extend_v2()` 分支:创建 `use_custom_mask=False` 的 causal paged-prefill wrapper(draft-extend 是对完整序列的 causal paged prefill,无 mask),按 bs 存入 `draft_extend_cuda_graph_metadata`,replay 时用捕获的 wrapper 更新索引;(2) 固定 qo 布局——`EagleDraftExtendInput.generate_attn_arg_prefill` 的 `qo_indptr` 从按 `num_accept_tokens` 累加改为 `arange(0, (bs+1)*num_tokens_per_req, step=num_tokens_per_req)` 的常量每请求数布局(graph capture/replay 要求形状恒定),与 padded draft-extend 输入一致,捕获时传入 `num_tokens_per_req=self.num_tokens_per_bs`;(3) 去掉每步 D2H 同步——`FlashInferIndicesUpdaterPrefill.update` 新增 `extend_prefix_lens_cpu` 参数,用 host 端 sum 替代 `paged_kernel_lens.sum().item()` 的设备同步。限制:FR-Spec(`speculative_token_map`)回退 eager——draft-extend graph buffer 按完整 draft vocab 分配,与缩减的 hot vocab head 不兼容,故 `flashinfer_graph_supported` 要求 `speculative_token_map is None`。

## 关键改动

```text
python/sglang/srt/layers/attention/flashinfer_backend.py: draft_extend_v2 图捕获分支 + extend_prefix_lens_cpu 免 D2H
python/sglang/srt/speculative/eagle_info.py: generate_attn_arg_prefill 改常量 qo 布局
python/sglang/srt/speculative/eagle_worker_v2.py: 白名单加入 FlashInferAttnBackend(FR-Spec 除外)
python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py: 捕获时传 num_tokens_per_req
```

关键片段(常量 qo 布局):

```python
# Constant num_tokens_per_req qo layout (required for cuda-graph capture).
qo_indptr = torch.arange(
    0, (bs + 1) * self.num_tokens_per_req,
    step=self.num_tokens_per_req, dtype=torch.int32, device=device,
)
```

## 性能声明

PR 正文:约 10% 在线 median E2E 延迟改善(条件:online benchmark,request-rate 8,Llama-2-7B EAGLE topk=4);accept length 不变。PR 未注明该测试的硬件配置。

## 验证与回归风险

diff 中无新增测试文件,依赖既有 CI(CI 状态:Base 失败 x,Extra no_entry_sign,合并时未全绿)。风险面:whitelist 扩展只影响 EAGLE v2 worker + FlashInfer 后端组合,且 FR-Spec 显式排除;`generate_attn_arg_prefill` 的 qo 布局变更是共享路径,其他调用该函数的后端(Triton/TRTLLM MLA 等)依赖其语义与 padded 输入一致;`update_single_wrapper` 的 host-side sum 分支仅在传入 `extend_prefix_lens_cpu` 时生效,其余路径行为不变。

## 关联

同分类(Speculative Decoding):PR #10090(Standalone speculative decoding,同样触及 eagle_draft_extend_cuda_graph_runner)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
