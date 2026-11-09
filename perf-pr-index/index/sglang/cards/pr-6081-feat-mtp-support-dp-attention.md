# PR #6081: feat: mtp support dp-attention

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @u4lr451 |
| 合并 | 2025-06-17 |
| 分类 | Speculative Decoding(tier: major) |
| Label | high priority |
| Diff | diffstat 缺失(pack 未记录);diff 可用,另有 5 个文件被截断(详见 PR) |
| 链接 | https://github.com/sgl-project/sglang/pull/6081 |

## 一句话结论

让 DeepSeek MTP(NextN)投机解码支持 DP-attention,启用 target/draft 双侧 CUDA graph,并消除按 draft tokens 倍增的冗余显存分配。

## 动机

MTP 此前与 `--enable-dp-attention` 不兼容:DP 分数下的 gather/scatter、global_num_tokens 语义都是按单 token/req 设计的,叠加上投机解码的 `num_draft_tokens` 倍增后就错位;相关 bug 另见 #4783、#4847。此外 MTP on DP-attention 下 target 与 draft 模型都无法用 CUDA graph,decode 延迟受损;`gathered_buffer` 在 MTP 时按 `--speculative-num-draft-tokens` 倍数分配,造成 target/draft 两边的冗余 GPU 显存占用,加剧 DP 并发下的显存争抢与 all_reduce 通信开销。

## 优化点

三块机制改动:(1) DP-attention 语义适配 MTP——scheduler 新增 `coordinate_spec_dp_attn_batch`:DP 下只要任意 rank 有新 prefill batch,全体都走 prefill-first 分支,保证投机路径的 DP 批次对齐;decode 时 `num_tokens` 不再乘 `speculative_num_draft_tokens`,改为在 `ForwardBatch.init_new` 中把 `global_num_tokens` 显式乘上 `spec_num_draft_tokens`,并据此分配 `gathered_buffer`(每次只按真实需要分配,消除固定倍数冗余);`deepseek_nextn.py` 删除 DP 下 ReplicatedLinear lm_head 的特殊分支,统一走 `ParallelLMHead(use_attn_tp_group=enable_dp_lm_head)`。(2) CUDA graph 接口统一——所有 attention backend 的 `init_cuda_graph_state` 签名统一为 `(max_bs, max_num_tokens)`,kv_indices/custom_mask 等缓冲从按 `max_bs` 尺寸改为按 `max_num_tokens` 尺寸分配;target 与 eagle draft 的 cuda graph runner 都加 DP-attention 分支(cap/replay 时计算 `total_batch_size`、填充 `global_num_tokens_gpu`、复用常驻 `gathered_buffer`);replay 时 metadata 切片只取前 `bs` 行。(3) 修正非零 DP rank 的 MTP 接收率——`deepseek_v2.py` 在 DP attention + EAGLE 时对 layer 输出 `hidden_states.clone()`(注释指向 PR 讨论 r2147452251,解决 non-zero DP rank 的 MTP reception rate 退化)。

## 关键改动

```text
python/sglang/srt/managers/scheduler.py: 新增 coordinate_spec_dp_attn_batch,DP 投机下 prefill-first 对齐
python/sglang/srt/model_executor/forward_batch_info.py: global_num_tokens 乘 spec_num_draft_tokens,gathered_buffer 按需分配
python/sglang/srt/model_executor/cuda_graph_runner.py + eagle_draft_cuda_graph_runner.py: target/draft CUDA graph 的 DP-attention cap/replay 支持
python/sglang/srt/models/deepseek_v2.py: DP+EAGLE 下 hidden_states.clone() 修 MTP 接收率
python/sglang/srt/layers/attention/*(多个 backend): init_cuda_graph_state 统一 (max_bs, max_num_tokens) 签名
```

draft 图内 DP 缓冲预分配:

```python
if self.enable_dp_attention or self.enable_sp_layernorm:
    self.gathered_buffer = torch.zeros(
        (self.max_num_token, self.model_runner.model_config.hidden_size),
        dtype=self.model_runner.dtype,
    )
    self.global_num_tokens_gpu = torch.zeros((self.dp_size,), dtype=torch.int32)
```

## 性能声明

PR 未给出吞吐/延迟 benchmark 数字。显存方面只有定性声明:消除按 `--speculative-num-draft-tokens` 倍增的 `gathered_buffer` 冗余分配,减少 target/draft 显存占用、降低 DP 并发下的显存争抢与 all_reduce 通信开销,未附具体数值。

## 验证与回归风险

精度:MMLU benchmark(`bench_sglang.py --data_dir data --nsub 20`),DeepSeek-V3-0324,两节点 TP=16。Baseline(非投机)平均 accuracy 0.887;MTP + DP-attention(`--speculative-algo NEXTN --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --dp-size 8 --disable-cuda-graph` 下测)平均 accuracy 0.887,两者一致。风险面:`init_cuda_graph_state` 签名变更是跨所有 attention backend 的接口改动(aiter/flashinfer/fa3/flashmla/triton/cutlass_mla/tbo 均被触及);DP CUDA graph 的 bs 语义在投机下从 token 数改回 batch 数,`--disable-cuda-graph-padding` 与 padding 两条路径都需按新语义走;非 CUDA graph 的 MTP+DP 用户行为不变。

## 关联

修复相关 bug:#4783、#4847。MTP 接收率修正的讨论见本 PR 的 review comment(r2147452251)。同区域:MTP/NextN 投机解码与 DP-attention 集成的其他 PR 编号未在正文注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
