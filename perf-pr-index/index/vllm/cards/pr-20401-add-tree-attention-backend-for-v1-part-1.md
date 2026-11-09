# PR #20401: Add tree attention backend for v1 (part 1)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @TheEpicDolphin |
| 合并 | 2025-08-04 |
| 分类 | Speculative Decoding(tier: major) |
| Label | speculative-decoding, ready, v1, llama |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 中 4 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/20401 |

## 一句话结论

为 v1 EAGLE speculative decoding 新增 TreeAttentionBackend(part 1):triton unified attention 支持 query-on-query 树 mask,当前仅支持链式 draft。

## 动机

EAGLE speculative decoding 中 target model 校验 draft token 时,树形 draft 的 token 只 attend 祖先 token,需要 attention bias 屏蔽非后代 token 之间的注意力;v1 此前没有能校验 draft token 树的注意力后端。

## 优化点

triton `unified_attention` 新增 `qq_bias` 参数(2D (q_len, q_len)),仅非 None 时启用(默认关闭):2d/3d 内核中对 key 相对位置落在 query 区间内的列加载 bias 加到 S。新增 TreeAttentionBackend(底层是 triton attention + 自定义树 bias):`TreeAttentionMetadataBuilder` 由 `speculative_token_tree` 构造 (tree_len+1, tree_len+1) bias(-inf 填充、对角线与根置 0、祖先置 0),`build_for_drafting` 按 draft step 切片 bias;`SpeculativeConfig` 未给树时生成链式树并按 BFS 排序;注册 `_Backend.TREE_ATTN` 与 V1 backend 白名单。本 PR 不改变现有 v1 EAGLE 行为;tree scoring 尚未实现,故仅链式 draft。TODO:树分支选中时的 paged KV 修复、draft 模型 attention backend 可配置(当前强制 FlashAttention)。

## 关键改动

```text
vllm/attention/ops/triton_unified_attention.py: 2d/3d 内核新增 qq_bias_ptr / USE_QQ_BIAS(仅 query 区间加载)
vllm/v1/attention/backends/tree_attn.py(新): TreeAttentionBackend/Impl/MetadataBuilder 与树 bias 构造
vllm/config.py: SpeculativeConfig 生成/排序 speculative_token_tree
tests/v1/spec_decode/test_tree_attention.py(新): 树注意力输出 vs 逐分支 FA 输出正确性
```

## 性能声明

Llama-3.1-8B-Instruct + EAGLE-LLaMA3.1-Instruct-8B(num_speculative_tokens 3),V1,random 1000/300、并发 64、128 prompts(PR 未注明 GPU 型号):输出吞吐 FA baseline 3248.45 / FA after 3311.34 / Tree Attention 3000.36 tok/s;Mean TPOT 13.60 / 13.52 / 15.77 ms。作者结论:该基准验证本 PR 未回归现有 v1 spec decode(链式);树注意力本身当前略慢,预期树 draft 提升的平均接受长度会补偿。内核微基准(与 TritonAttentionBackend 对比,batch 1/16/64、seq 16/1024/2048、q 1/13/40 us 级):树 bias 带来轻微开销,如 batch 64、seq 2048、q 40:752.93 → 760.29 us。

## 验证与回归风险

tests/v1/attention/test_attention_backends.py 加入 TREE_ATTN(树 mask 与逐分支 FA allclose atol 7.81e-3,覆盖 batch/heads/seq 位置/链与树两种 mask);test_eagle propose 用例通过;手动 chat 输出与 FlashAttention 一致。风险:triton unified_attention 签名变化影响所有 Triton attention 用户(默认关闭);仅支持 DECODER,不支持 block-sparse 与 fused output quantization;head size 限定 32–256。

## 关联

同分类 Speculative Decoding:#43091、#38174、#50272、#45635。后续(作者 TODO):tree scoring、paged KV 树分支修复。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
