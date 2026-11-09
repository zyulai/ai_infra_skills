# PR #1138: Optimize MLA/GQA/MQA Triton decoding

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ispobock |
| 合并 | 2024-08-19 |
| 分类 | Attention / Kernel (tier: major) |
| Label | high priority, performance |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/1138 |

## 一句话结论

为 Triton decode attention 新增 GQA/MQA/MLA 分组 kernel:一个 block 处理共享同一 k/v head 的 BLOCK_H 个 q head,改善共享 KV 场景下的访存效率。

## 动机

优化 MLA/GQA/MQA decoding 的访存。原 Triton decode kernel 按 (batch, q_head, kv 块) 展开网格,每个 q head 独立加载同一 k/v head 的数据;在 GQA/MQA/MLA 下多个 q head 共享一个 k/v head,KV 被重复读取。思路来自 lmdeploy 的 PR #1649(每个 block 处理 BLOCK_H 个 q head、共享 k/v head)。

## 优化点

在 `decode_attention.py` 中按 `kv_group_num = q_head_num / kv_head_num` 分派两条路径:`kv_group_num == 1`(MHA)继续走原 `_fwd_kernel_stage1`/`_decode_softmax_reducev_fwd`;否则走新增的 grouped 两阶段 kernel。机制变化在网格与 head 维度组织:stage1(`_fwd_grouped_kernel_stage1`)网格为 (batch, cdiv(head_num, min(BLOCK_H, kv_group_num)), cdiv(max_len, 32)),每个 program 内 `cur_head = cur_kv_head * kv_group_num + arange(BLOCK_H)` 一次加载 BLOCK_H 个 q head 的向量,与共享的 K tile 做一次 `tl.dot(q, k)`(支持 MLA 的 576 = 512 + 64 拆分 `BLOCK_DMODEL`/`BLOCK_DPE` 双 dot),把 BLOCK_H 个 head 的 logits 一并写出;`BLOCK_H = max(16, next_power_of_2(kv_group_num))`。stage2(`_fwd_grouped_kernel_stage2`,BLOCK 128,num_warps 8)同样以 BLOCK_H 为行单位做 online softmax 累加并 `tl.dot(p, v)`,一次归约出整组 head 的输出。原 MHA-only kernel 中的 576 特判被移到 grouped kernel,原 kernel 收窄为仅支持 Lk in {16,32,64,128,256}。

## 关键改动

```text
python/sglang/srt/layers/decode_attention.py: 新增 _fwd_grouped_kernel_stage1/stage2 与两个 wrapper;decode_attention_fwd 按 kv_group_num 分派
```

关键片段(grouped stage1 的 head 分组与网格):

```python
cur_head = cur_kv_head * kv_group_num + tl.arange(0, BLOCK_H)
mask_h = cur_head < (cur_kv_head + 1) * kv_group_num
...
q = tl.load(Q + offs_q + start_mark, mask=mask_h[:, None])...
qk = tl.dot(q, k)
```

## 性能声明

PR 正文仅写"Optimize memory access for MLA/GQA/MQA decoding"与实现描述,未附任何 benchmark 数字、测试条件或对比表格——性能收益未注明。

## 验证与回归风险

PR 正文未注明新增测试。回归面:Triton decode attention 的 GQA/MQA/MLA 路径(DeepSeek MLA 576 维拆分、各 GQA/MQA 模型的 decode);MHA(kv_group_num==1)路径保持原 kernel 不变;行为差异点在 grouped kernel 的 BLOCK_H 对齐与 logits 中间布局(按 [head, token] 写出再由 stage2 读回);对 `logit_cap`、paged KV 索引(`Req_to_tokens`)的处理沿用了原 stage1 的语义。

## 关联

思路来源:InternLM/lmdeploy PR #1649(one block handles BLOCK_H q heads with shared k/v head)。同区域为 sglang Triton decode attention kernel 的后续演进(如 #4553 的动态 workload 优化);pack 未列出其他同分类 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
