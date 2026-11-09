# RFC #42082: [RFC]: Standardize KV-cache Layouts

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42082](https://github.com/vllm-project/vllm/issues/42082) |
| 作者 | LucasWilkinson |
| 创建日期 | 2026-05-08 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 26 · 💬 10 |
| 分类 | Other |

## 一句话结论
统一各 attention 后端的 KV-cache 逻辑形状与 stride_order,让 KV-connector 变成后端无关的通用代码。

## 提案动机
目前不同 attention 后端的 KV-cache 在语义与物理上都可能有不同布局,导致 KV-connector 代码混乱(大量 `is_mamba`、`is_mla` 标志),connector 与 attention 后端紧耦合(如 flash_attn.py 中的后端分支),还引发过相关 bug(如 PR #41657 下的 issuecomment)。更多细节见正文链接的设计文档。

## 提出的方案
- **统一逻辑形状**:`[num_layers_slots, num_blocks, num_heads, num_states, <state_content>]`,所有后端(GQA、DeepSeek V4、MLA、Mamba2 Conv/SSM)都映射到它;`<state_content>` 后端特定、始终连续。
- **物理布局由 stride_order 控制**,定义四种:`LBHNC`(默认,头切片连续,异构 TP 友好)、`LBNHC`(state 切片连续,异构 block size 友好)、`BLHNC`(一个 block 跨所有层数据连续,connector 可单次 RDMA 读整块,homo-TP/block-size 下更大 P/D 传输)、`BHLNC`(异构 TP 更大 P/D 传输)。共享张量时要求每层每块 `[H, N, C]` 物理连续,只有 LBHNC 与 BLHNC 满足;若各层 H/N/C 尺寸一致则任一布局皆可。
- **KVCacheSpec 属性**:`num_heads`、`tokens_per_state`(-1 无限/1 标准/N 压缩)、`state_content_size`;`num_states` 由 `block_size / tokens_per_state` 在分配时推导。
- **Spec 自带 TP 切片**:各 spec 子类实现 `slice_for_tp_transfer`(签名带对端 spec/tp/rank;GQA 按头切片并处理两侧复制,Mamba Conv 按 x/B/C 三个切片),取代 connector 的字节计算。
- **Connector 变为通用**:TP 切片 setup 时算一次;`block_mapping` 天然处理异构 block size;`jointly_contiguous_chunks` 把每对块分解成双方物理布局都连续的区域做 RDMA 传输;目的端偏移来自 meta tensor(`device='meta'`,不占 GPU 内存)。

## 优化点与收益
正文未给出量化数字。定性收益:消除 `is_mamba`/`is_mla` 类标志与 connector-后端紧耦合;异构 TP、异构 block size 的 KV 传输成为自然支持;BLHNC/BHLNC 布局带来更大的 P/D 传输粒度(单块全层一次 RDMA 读)。

## 现状与落地
仍在推进/待实现。正文 Worktracking 列出:准备性 PR #42122、#42095;后续步骤 [1/N] PR #42374(标准化 attention backends 与 kv-cache 分配,移除 `get_kv_cache_stride_order`/`get_kv_cache_shape`)、[2/N] PR #43151、[3/N] 重构 kv-connector 使用 meta-tensors 与切片。

## 关联
- [设计文档(Google Doc)](https://docs.google.com/document/d/1-TnWZf8jI6nWgE-xaQkgsWy130CVUqqt1NYqY-mAOe4/edit?usp=sharing)
- [PR #41657 下的相关 bug](https://github.com/vllm-project/vllm/pull/41657#issuecomment-4400641394)
- [PR #42122](https://github.com/vllm-project/vllm/pull/42122)(preparatory)
- [PR #42374](https://github.com/vllm-project/vllm/pull/42374)([1/N] 标准化)
- [PR #43151](https://github.com/vllm-project/vllm/pull/43151)([2/N])

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
