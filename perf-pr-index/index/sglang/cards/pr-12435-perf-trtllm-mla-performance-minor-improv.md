# PR #12435: perf: trtllm mla performance minor improvements

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @cicirori |
| 合并 | 2025-11-01 |
| 分类 | Speculative Decoding(major) |
| Label | high priority, run-ci |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/12435 |

## 一句话结论

清理 trtllm_mla 后端冗余计算:把 seq_lens 运算移出 CUDA graph replay、K+K_rope 拼接延迟到 prefill,decode 提速约 2%,投机解码场景最高 4-5%。

## 动机

`trtllm_mla` 后端存在若干小低效与易混淆逻辑:一是 `seq_lens` 的逐元素运算在每次 CUDA graph replay 中重复执行,而非在 metadata 初始化时做一次;二是 `draft_extend`、`draft`、`verify` 路径中存在冗余的 `torch.cat([k, k_rope])` 拼接;三是 seq len 元数据逻辑混乱(`max_seq_len_q = num_tokens_per_bs + 1` 疑似不必要、`accept_length_cpu` 的 +1 处理不一致)。作者在早期调试中还遇到奇数 batch size 性能差、`draft_extend` 用 CUDA graph 性能差等问题,这次清理旨在同时改善稳定性与运行效率。

## 优化点

核心是把每步重复的元数据计算挪到初始化/捕获阶段。给 `TRTLLMMLADecodeMetadata` 增加 `seq_lens_k` 字段:target_verify 与 draft_extend 模式下在 capture 阶段就分配好 int32 `seq_lens_k` 缓冲,replay 阶段只做 `copy_` 更新,不再在图内重算 `seq_lens + num_draft_tokens` 等算式;`init_forward_metadata`(非图路径)同样统一为"先算 seq_lens_k,再据此创建 block_kv_indices"。draft_extend 的 seq_len 语义被统一成 `seq_lens = prev_context + max_draft_tokens`(对齐 pad_draft_extend_query),`max_seq_len_q`/`sum_seq_lens_q` 改为基于 `accept_length_cpu` 加一致的 +1(分别 +1 与 +bs),并删掉多余的 `max_seq_len_q = num_tokens_per_bs + 1`。另外 `forward_extend` 中 `torch.cat([k, k_rope])` 从所有投机路径移到仅 prefill 执行,投机路径直接使用 metadata 里的 `seq_lens_k`,图内不再做拼接与 seq_lens 算术。

## 关键改动

```text
python/sglang/srt/layers/attention/trtllm_mla_backend.py:
  TRTLLMMLADecodeMetadata 增加 seq_lens_k 字段并在 capture 阶段初始化
  init_forward_metadata_replay_cuda_graph: replay 仅 copy_ 更新,移除图内
    seq_lens 算术
  forward_extend: k/k_rope 拼接移到 prefill-only 分支
```

## 性能声明

PR 正文:preliminary benchmarks 显示 decode 延迟约 2% 提升;更快的投机解码场景最高 4-5% 收益。给出的具体对比:DeepSeek-V3-0324-FP4(modelopt_fp4),`SGLANG_ENABLE_SPEC_V2=1`,EAGLE 投机(num_steps 3, topk 1, draft tokens 4),TP4,trtllm_mla 注意力 + flashinfer_trtllm MoE,page-size 64,`sglang.test.send_one` 单请求:PR 侧 1.895 s / 512 tokens / acc length 2.876 / 270.21 token/s,main 侧 1.939 s / 264.11 token/s,即 270.21/264.11 = 1.023。精度:gsm8k 10 次运行(正文仅附截图,无数字表格)。

## 验证与回归风险

精度验证为 gsm8k 10 次运行(结果以截图形式给出,正文无数字);Checklist 中"Add unit tests"未勾选,pack diff 未含新增测试。风险集中在投机解码元数据语义:draft_extend 的 `max_seq_len_q`/`sum_seq_lens_q` 由 `num_tokens_per_bs`(capture)与 `accept_length_cpu + 1`(replay)两套来源混合决定,`seq_lens_k` 的 `prev_context + max_draft_tokens` 对齐规则若与 pad_draft_extend_query 假设不符,会导致 query/KV 错位(作者自己在 Notes 中请 reviewer 校正 accept_length 语义);replay 中 `seq_lens[:bs]` 截断改变了 `create_flashmla_kv_indices_triton` 的输入形状。影响范围仅 `--attention-backend trtllm_mla` 的投机与非投机 extend 路径。

## 关联

同区域:#14162 / #22316(DeepSeek w4a8 DeepEP FP8 dispatch,同为 DeepSeek 性能链路)。trtllm_mla 后端相关的前序 PR 编号在素材中未给出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
