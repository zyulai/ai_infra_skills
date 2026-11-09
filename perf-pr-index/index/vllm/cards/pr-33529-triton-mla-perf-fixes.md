# PR #33529: Triton MLA perf fixes

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @koush |
| 合并 | 2026-04-02 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready, v1, deepseek |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/33529 |

## 一句话结论

修复 Triton MLA 在 sm120 上 batch=1 长上下文的性能塌陷,Kimi 2.5 80K 上下文单并发峰值吞吐 +52.94%(34→52 tok/s)。

## 动机

Triton MLA 后端在 sm120 上 batch size 为 1 时,随上下文变长性能持续劣化,使 DeepSeek 与 Kimi k2/k2.5 几乎不可用。作者定位到两个主要问题:(1) 低 batch 时 KV split 数固定为 4,长序列下每个 split 的工作量与可用 SM 不匹配,SM 严重欠利用;(2) 内核里无条件加载 V 向量——MLA 只有一份 `c_kv`(K 与 V 共享),把它再当作 V 加载一遍纯属浪费。

## 优化点

两处内核层修复:(1) 动态 KV splitting——`forward_mqa` 不再固定 4 splits,改为按 `max_seq_len // min_work_per_split`(每 split 最少 512 token,硬件相关)算出理想 split 数,取 2 的幂(避免过多 kernel 实例化),再以 `SM 数 × occupancy_multiplier(2)` 为上限(每 SM 允许 2 个 block 以做延迟隐藏),`VLLM_BATCH_INVARIANT` 时仍保持 1 split 保证 batch 不变性;SM 数在 backend 初始化时经 `torch.cuda.get_device_properties(0).multi_processor_count` 缓存。(2) MLA 的 V 加载消除——stage1 内核新增 `IS_MLA` constexpr,MLA 路径不再把 `c_kv` 重复加载当 V,直接 `v = tl.trans(k)` 复用已加载的 k 做点积;同时循环体微调:`base_offs_k/v` 提出循环外只算一次,`range` 换 `tl.range`,Q 与 page table 加载用 `cache_modifier=".ca"`、K 加载用 `.cg` 以显式促进 load/compute 重叠。

## 关键改动

```text
vllm/v1/attention/backends/mla/triton_mla.py: 动态 num_kv_splits(按 max_seq_len 与 SM 数)+ is_mla=True
vllm/v1/attention/ops/triton_decode_attention.py: stage1 内核 IS_MLA 分支跳过 V 加载、cache modifier、循环外提
```

关键摘录(动态 split 决策):

```python
min_work_per_split = 512
ideal_splits = max(1, attn_metadata.max_seq_len // min_work_per_split)
ideal_splits = triton.next_power_of_2(ideal_splits)
occupancy_multiplier = 2
max_splits = self._sm_count * occupancy_multiplier
num_kv_splits = min(ideal_splits, max_splits)
```

## 性能声明

PR 正文测试条件:Kimi 2.5,`-dcp 8`,80K context。结果表:

| 并发 | 场景 | OLD 峰值吞吐 (tok/s) | NEW 峰值吞吐 (tok/s) | 提升 |
| --- | --- | --- | --- | --- |
| 1 | 128 tokens | 54.00 | 55.00 | +1.85% |
| 8 | 128 tokens | 272.00 | 272.00 | 0.00% |
| 1 | 80K tokens | 34.00 | 52.00 | +52.94% |
| 8 | 80K tokens | 176.00 | 184.00 | +4.55% |

即收益集中在长上下文低并发(batch=1、80K 时 +52.94%),短上下文与小 batch 场景基本持平或持平为 0。硬件为 sm120(正文以 sm120 表述问题,未给出具体 GPU 型号与精度)。

## 验证与回归风险

Test Plan:Kimi 2.5、DeepSeek v2、Qwen 235B 三个模型加载与生成正确;卡内未看到新增单元测试。回归风险:`triton_decode_attention` 的 stage1 是 MLA 与非 MLA(GQA/MQA)共用的 grouped decode 内核,`IS_MLA` 默认 False 保持旧 V 加载路径,非 MLA 调用方不受影响,但 cache modifier(`.ca`/`.cg`)与 `tl.range` 对所有走该内核的模型生效,需关注不同架构上的数值与性能;动态 split 数使 `attn_logits` 中间张量形状随 `max_seq_len` 变化(正文标注 TODO(lucas) Allocate ahead of time 的原分配逻辑仍在);`VLLM_BATCH_INVARIANT=1` 路径行为不变。

## 关联

同区域:Triton MLA 后端(`triton_mla.py`/`triton_decode_attention.py`)及 MLA 系列内核(FlashMLA、FlashInfer sparse MLA 等)。本卡所在批次素材中未列出其他同区域 PR 编号,暂无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
