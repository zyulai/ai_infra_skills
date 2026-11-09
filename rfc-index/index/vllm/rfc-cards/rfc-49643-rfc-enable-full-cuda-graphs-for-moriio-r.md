# RFC #49643: [RFC]: Enable full CUDA graphs for MoRIIO READ mode via async KV-load gating

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49643](https://github.com/vllm-project/vllm/issues/49643) |
| 作者 | edwinlim0919 |
| 创建日期 | 2026-07-23 |
| 状态 | 🟢 open |
| labels | RFC, unstale |
| 热度 | 👍 0 · 💬 2 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
提议让 MoRIIO READ 模式采用与 WRITE/NIXL 相同的异步 KV 加载门控,把 KV 读取移出 forward 路径,恢复 decode 的 full CUDA graph。

## 提案动机
MoRIIO 支持 WRITE(producer 推 KV,decode 可跑 FULL CUDA graph)与 READ(decode 拉 KV,因 #48534 退化为 PIECEWISE)两种 KV 传输。READ 被迫 piecewise 的原因:它在 decode forward 开始时逐层发起 RDMA 读,且必须逐层阻塞等该层 KV 到位(wait_for_layer_load 是插在 attention 层之间的 Python 等待);FULL CUDA graph 把整个 forward 捕获为一张无 host 代码的图,回放时该 barrier 会被静默跳过,attention 抢跑未到位的 KV,产生随并发恶化的垃圾输出。#48534 让 barrier 生效(正确性)并把 decode 降级为 PIECEWISE,而 PIECEWISE 每步都付 per-op kernel launch 开销。对照:vLLM 的 NIXL pull 模式同为 READ 却能跑 full CUDA graph——靠异步 KV-load 门控,整份 KV 落地前请求停泊在 WAITING_FOR_REMOTE_KVS。

## 提出的方案
让 MoRIIO READ 采用 MoRIIO WRITE 与 NIXL PULL 已用的异步加载门控:
1. get_num_new_matched_tokens(READ)返回 async = True,请求停泊 WAITING_FOR_REMOTE_KVS
2. 把读取移出 forward 路径:start_load_kv 每步都跑(含 kv_connector_no_forward 路径),为停泊请求发读;经 get_finished → done_recving 上报 per-request 完成(_pop_done_transfers 已计算,但 READ 路径当前丢弃)
3. wait_for_layer_load 恢复为 no-op(KV 在 decode forward 前已全部到位,#48534 的 barrier 不再需要)
4. READ 的 requires_piecewise_for_cudagraph 置 False,恢复 full CUDA graph
权衡:用一次性延迟换每步节省——请求等整份 KV 落地才跑第一步 decode;start_load_kv 提前一步发读且每步运行,请求 N 的读与其他请求的 decode 计算重叠,稳态并发下 pipeline bubble 大体被隐藏。

## 优化点与收益
正文未给出量化数字;收益为消除 READ 模式每步的 per-op launch 开销,使 decode 性能对齐 WRITE 模式的 full CUDA graph。

## 现状与落地
🟢 open;作者建议的顺序是先合入 #48534(当下的正确性修复),再实现本 RFC 恢复 full-graph 性能;文中提到该话题在 AMD SIG 会议讨论过,并 @ tjtanaa、AndreasKaratzas、Rohan138 征求意见。

## 关联
- #48534(READ 模式 piecewise 降级与 wait_for_layer_load barrier 的来源)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
