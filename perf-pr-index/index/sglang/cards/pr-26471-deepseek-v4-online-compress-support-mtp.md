# PR #26471: DeepSeek-V4 Online Compress support MTP

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zhangxiaolei123456 |
| 合并 | 2026-06-16 |
| 分类 | Speculative Decoding(classify.py 给出,tier: major) |
| Label | documentation, quant, amd, dependencies, lora, deepseek, speculative-decoding, blackwell, npu, run-ci, piecewise-cuda-graph, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 片段约 5 个文件,另有 9 个文件因超长被裁剪 |
| 链接 | https://github.com/sgl-project/sglang/pull/26471 |

## 一句话结论

让 DeepSeek-V4 的 online c128 压缩与 MTP 共存:可处理 token 量从 2,046,720 提升到 5,692,160(约 +280%),decode 吞吐与 NO_ONLINE+MTP 仅差约 2%。

## 动机

DeepSeek-V4 的 online compress(c128 在线 KV 压缩)此前与 MTP(EAGLE 投机解码)组合不可用:`online_compress+MTP` 存在正确性问题。原因是 MTP 的 target-verify 阶段一次前向写入多个 token 的压缩状态,而 online 状态池只保留每请求一个 in-progress chunk,verify 被拒绝回滚时压缩 running state 会与实际序列长度脱节。本 PR 解决该组合问题,使 online compress(省 KV 显存)可以叠加 MTP(提 decode 吞吐)。

## 优化点

- 新增 `online_c128_mtp.cuh` JIT kernel 组:`write_prefix_states` 在 target-verify 时按 verify token 逐 step 前推 online running state(max/sum/kv 的 online softmax 累加),把每个前缀长度的中间状态写到 `slot + step*state_slot_stride` 的临时槽;`mark_pending` 在 verify 开始时快照各请求 seq_len;`commit_pending` 在下一轮按实际接受长度 clamp(`clamp_accept_len`)后把最终槽状态拷回主槽,完成提交/回滚。
- 修改 `c128_online_v2.cuh` 的 plan 语义:prefill write pass 从 `read_page_1` 读 prior state(缺失时 fallback `read_page_0`)并写 `read_page_0`,stage 1 kernel 用 `state_slot_offset` 区分主 bank 与 MTP 临时 bank 的槽位。
- decode/prefill plan kernel 增加 `state_slot_offset` 参数,由 `OnlineC128MTPController` 从 KV pool 读取;prefill plan 支持 `use_cuda_graph`,把 plan 数量 pad 到固定 N 并用 `CompressPlan::invalid()` 填充,cuda graph 下 kernel 内跳过 invalid plan,保证 capture 形状稳定。
- Python 侧 `OnlineC128MTPController` 协调 verify 生命周期:仅在 `SGLANG_OPT_USE_ONLINE_COMPRESS=1` 且 `SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1` 且 MTP 开启时激活;`prepare_forward` 在每轮 forward 前先 commit 上轮 pending、再对 target-verify 标记新 pending 并返回 state_slot_offset 供 compressor plan 使用。

## 关键改动

```text
python/sglang/jit_kernel/csrc/deepseek_v4/online_c128_mtp.cuh: 新增,write_prefix/mark_pending/commit_pending kernel
python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh: plan 语义改为双 bank(read_page_1 存 committed 槽)
python/sglang/jit_kernel/dsv4/compress.py: generate_online 增加 state_slot_offset/use_cuda_graph
python/sglang/jit_kernel/dsv4/online_c128_mtp.py: 新增,OnlineC128MTPController 生命周期管理
```

```python
def commit_pending(...):
    old_seq = params.pending_seq_lens[req]
    cur_seq = params.cur_seq_lens[bid]
    accept = clamp_accept_len(cur_seq - old_seq, params.num_verify_tokens)
    if accept <= 0: return
    final_seq = old_seq + accept
    if ((final_seq & 127) == 0) return  # 已闭合 chunk,主槽已正确
    # 把 slot + accept*stride 的前缀状态拷回主槽
    dst[d] = src[d]  (src = state + (slot + accept*state_slot_stride))
```

## 性能声明

H20,DeepSeek-V4-Flash,input 16000 / output 1500,chunked-prefill-size 8192,QPS 40,max_concurrency 128,num_promot 512(预填充/解码分离部署,TP8,decode 侧 DP8+DeepEP+EAGLE MTP steps=2/topk=1/draft=3):decode TPOT——online MTP c128 11.92 ms、online c128(无 MTP)23.03 ms、normal MTP 11.80 ms;decode throughput——8822.55 / 5091.45 / 8906.48 tok/s。即 online MTP 相比 online 无 MTP 提升约 73% 吞吐,与 normal MTP 差距约 1%。可处理 token 量从 2,046,720 增至 5,692,160,作者称 "roughly 280% improvement"。

## 验证与回归风险

精度(H20,MMLU/GSM8K/QPQA):online MTP c128 0.884/0.952/0.914,online c128 0.885/0.955/0.895,normal MTP 0.886/0.959/0.922,差距在小数第三位内。PR 未列新增单元测试,Checklist 各项未勾选;CI 快照 Base 失败、Extra 通过。风险:`c128_online_v2.cuh` 的 read/write page 语义变更是所有 online c128 用户的共享路径;cuda graph pad + invalid-plan 机制影响 prefill plan H2D 路径;MTP 回滚逻辑与 KV pool 的 state slot 布局强耦合(`get_online_c128_mtp_state_slot_offset` 等新接口)。

## 关联

同区域:DeepSeek-V4 MTP/CP 相关 PR(如 #24934 DeepSeek V4 MTP Support CP)、online compress 与 DSA indexer 优化(PR #32755 等);依赖环境开关 `SGLANG_OPT_USE_ONLINE_COMPRESS`、`SGLANG_EXPERIMENTAL_ONLINE_C128_MTP`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
