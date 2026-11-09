# RFC #11382: [RFC]: Hybrid Memory Allocator

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#11382](https://github.com/vllm-project/vllm/issues/11382) |
| 作者 | heheda12345 |
| 创建日期 | 2024-12-20 |
| 状态 | ⚪ closed(关闭于 2025-06-23) |
| labels | RFC, stale |
| 热度 | 👍 18 · 💬 7 |
| 分类 | KV Cache / Memory |

## 一句话结论
为混合架构模型按层类型分配 KV cache,消除统一分配造成的内存碎片并支持 prefix caching。

## 提案动机
vLLM 中混合模型(一种以上层类型)越来越多,例如滑动窗口注意力+自注意力(Gemma2、Ministral)、交叉注意力+自注意力(mllama)、Mamba 层+自注意力(Jamba、Bamba)。这类模型中不同 token 的 KV cache 大小不再相同,但 vLLM 只能为所有 token 分配相同大小的 KV cache。正文的动机部分给出的内存浪费数字为:mllama 可浪费 79.6%,Gemma-2 为 25%,Ministral 为 56.25%。此外 vLLM 为 mamba 层设有专用 MambaCacheManager(`model_executor/models/mamba_cache.py`),但与 prefix caching 不兼容。

## 提出的方案
- **Milestone 1(按层分配)**:假设各层 kv_hidden_size 相同、仅 token 数不同,page size 设为 `[BLOCK_SIZE, kv_hidden_size]`(当前 vLLM 为 `[num_layer, BLOCK_SIZE, kv_hidden_size]`);每个请求调用内存分配器 num_layer 次,得到每层各自的 block table。
- **层分组**:同组层共享 block table,要求组内层 token 数相同、各组层数相同;如 mllama 11B(8 个交叉注意力层 + 32 个自注意力层)分为 5 组。
- **按层类型的内存管理器**:每层类型一个 manager(SelfAttentionManager、SlidingWindowManager、MambaManager、CrossAttentionManager 等);所有层的空闲块放入同一队列,用 LRU 统一淘汰。
- **可定制淘汰策略**:mamba 模型只需缓存最后若干 token(如请求 `[sys1 sys2 prompt1 prompt2 prompt3]` 可淘汰 sys1 prompt1 prompt2,仅缓存 sys2 与 prompt3);滑动窗口层只缓存窗口内 token。
- **Milestone 2(LCM pages)**:引入两级页表——大页分配器(page size 取各类型内存 page size 的最小公倍数)+ 小页分配器(每种类型一个 manager,把大页切分为该类型的小页),以覆盖 Jamba(mamba state 为 kv_hidden_state/token/layer 的 256 倍)、Llava 式多模态 VisionEncoderCache 等不同 page size 场景。
- **落地步骤**:每种层类型一个 manager class(定制 allocate/free/prefix caching 逻辑);`Runner.initialize_kv_cache` 改为分配一个大 tensor、每层用不同 offset;`AttentionMetadata` 改为 `Dict[Attention.layer_name, AttentionMetadata]`,无需模型侧改动;后续再扩展 MLA、mamba 层、KV cache sharing、交叉注意力层、embedding 模型及 LCM 分配。

## 优化点与收益
动机部分照抄的数字:内存浪费 mllama 为 79.6%、Gemma-2 为 25%、Ministral 为 56.25%(BLOCK_SIZE=1 的 mllama 示例)。作者称已在 v0 上实现原型并在 H100 上得到结果,加速来自「更少碎片 + 更好的 prefix caching」两方面;具体量化数字以图片(figure5)形式给出,正文文本未列出数值,无法照抄。

## 现状与落地
已关闭(2025-06-23),labels 含 stale;正文未写明是否已实现或被替代,落地情况需查关联 PR。

## 关联
- #9464(Gemma2/Ministral 滑动窗口支持)
- PR #10927(MLA,方案声明在其之后支持 MLA 层)
- #10043、#10783(KV cache sharing 的需求来源)
- #10646(sparse KV cache 提及处)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
