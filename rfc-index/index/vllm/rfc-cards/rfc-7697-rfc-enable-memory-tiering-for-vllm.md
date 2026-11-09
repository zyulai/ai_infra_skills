# RFC #7697: [RFC]: Enable Memory Tiering for vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7697](https://github.com/vllm-project/vllm/issues/7697) |
| 作者 | PanJason |
| 创建日期 | 2024-08-20 |
| 状态 | ⚪ closed (关闭于 2024-12-21) |
| labels | RFC, stale |
| 热度 | 👍 12 · 💬 7 |
| 分类 | KV Cache / Memory |

## 一句话结论
提议为 vLLM 启用内存分层:prefix cache 淘汰时落到 DRAM 而非直接丢弃,让共享 prompt 免于 O(n^2) 重算。

## 提案动机
多轮对话、多模态、多 agent 等应用需要大量 KV cache,且多个请求共享 prompt;重算长度为 n 的共享 prompt 复杂度 O(n^2),而若把它存放在更慢但容量更大的 DRAM/Disk,加载 KV cache 并算下一个 token 只需 O(n)(假设全命中),超过某个点后换出重载比重算划算(论文 [1][2] 亦有此观察)。当前 vLLM 几乎不用二级存储(DRAM):只在 sequence 无法为新 token 分配块时换出(作者测的 sharedGPT、UltraChat、LooGle、Toolbench 上都很少发生),且 prefill 分配块时直接丢弃 evictor 中 GPU 块内容,而不是把它换到 CPU DRAM 供共享 prompt 的新请求免重算复用。

## 提出的方案
- 三步走:1) 实现与 DRAM 之间淘汰/提升的基本功能;2) KV cache 传输与模型执行重叠(逐层传输,而非传完再执行);3) 按 prompt 长度选择性淘汰到 DRAM
- Stage 1 原型已完成(prefix caching 开启时把块从 HBM 淘汰到 DRAM),改动不大:block_manager_v[1,2] 淘汰时分配 CPU 块并查询 cpu_allocator 的 block hash;scheduler 为 _scheduler_prefill 增加 blocks_to_swap_in/out;调整 worker.py 中 swap_out 与 swap_in 的顺序
- Stage 1 待办:更多测试;为 CPU 块加 free 时间戳(access_blocks_for_seqs 从不覆盖 CPU 块,淘汰策略有非预期行为);处理 oooooxxooooxx 模式下块同时从 CPU 与 GPU 掉落的情况(可能需改 kernel);支持 encoder-decoder 与 block_manager_v2;检查 chunked_prefill 兼容性
- Stage 3 需加 dry run 找临界点

## 优化点与收益
照抄数字:tp=1、pp=1、单卡 A100、togethercomputer/Llama-2-7B-32K-Instruct:recompute 时间二次增长,从 1024 tokens 起,其 latency 高于「prefill w cache + swapin」(未启用 swap-in 与执行重叠)。原型在 LooGLE benchmark 的 longdep_qa.json(平均 prompt 长度约 15k,request rate 1000)上测了不同 DRAM 大小下的 mean TTFT/TPOT(正文仅给图,未给具体数值;并观察到 DRAM 大于某阈值后性能反而下降,原因调查中)。

## 现状与落地
已关闭。作者已完成 Stage 1 原型并列出待办与后续阶段计划,落地情况需查关联 PR。

## 关联
- CachedAttention(USENIX ATC 24):Gao, Bin, et al. "{Cost-Efficient} Large Language Model Serving for Multi-turn Conversations with {CachedAttention}"
- FlexGen(ICML 2023):Sheng, Ying, et al.
- LooGLE:https://arxiv.org/abs/2311.04939

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
