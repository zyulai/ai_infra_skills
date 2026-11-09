# PR #46995: [Spec Decode] DSpark

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benchislett |
| 合并 | 2026-07-01 |
| 分类 | Speculative Decoding (tier: major) |
| Label | performance, new-model, speculative-decoding, ready, v1, qwen |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 内容可用,约 9 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/46995 |

## 一句话结论

新增 DSpark 半自回归投机解码方法:非因果滑窗稀疏 MLA + Markov 头注入块内依赖,DSV4-Pro 上平均接受长度较 MTP 最高 1.31x。

## 动机

DeepSeek 发布了 DSpark 投机解码权重(DeepSeek-V4-Pro-DSpark 与 DeepSeek 训练的 Qwen3-DSpark),vLLM 尚无对应方法。DSpark 的 draft 块内注意力是非因果的(块内 token 互相可见),若为它重写一套 MLA attention 会大动 KV cache 工具链;同时块内各位置的采样存在依赖,需要一种低成本方式注入。动态 drafting 与 confidence-based scheduling 明确留给 #45953 后续统一。

## 优化点

(1) 非因果滑窗注意力复用现有 SparseMLA 后端:把 topk 宽度从 127 个前序 token 扩到下一个 2 的幂(如 256),索引列表含滑动窗口上下文加全部块内 token(含未来位置),其余槽位 -1 padding——由于 FlashMLA(SM90/SM100)与 FlashInfer TRTLLM(SM100/SM120)的 sparse decode kernel 是索引驱动、无因果掩码,这样即得到对因果性精确正确的注意力,且几乎不改 KV cache 工具;(2) 模型结构:DFlash 式并行 backbone(context-KV 预计算 + 非因果 query-block 前向)+ 轻量 sequential Markov 头(低秩 V x r / r x V 转移偏置加到 base logits,由 speculator 从左到右采样),`DSparkMarkovHead` 与 DSV4 风格 DSpark 模型共享;(3) CUDA Graph 捕获整条 DFlash backbone 加自回归采样循环为单个 graph,降低 host 与 kernel launch 开销;(4) 方法接入:`dspark` 加入 speculative method 类型与自动检测(V4 权重随目标 checkpoint 发布,Qwen3 走独立 draft checkpoint),支持 `draft_sample_method: probabilistic`,允许 async scheduling,强制 V2 model runner,与 dflash 同属 parallel drafting。

## 关键改动

```text
vllm/model_executor/models/qwen3_dspark.py: 新增 Qwen3DSparkModel/DSparkMarkovHead(低秩转移偏置头)
vllm/config/speculative.py: 注册 dspark 方法、自动检测、parallel_drafting、probabilistic draft 采样
vllm/config/vllm.py: dspark 强制 V2 model runner、async scheduling 白名单
vllm/model_executor/models/qwen3_dflash.py: precompute_and_store_context_kv 支持每层独立 slot_mapping(dummy run 跳过 cache 写)
tests/v1/attention/test_dspark_noncausal_sparse_mla.py: 529 行非因果稀疏 MLA 正确性测试
```

关键代码(Markov 头):

```python
def bias(self, markov_embed: torch.Tensor, logits_processor) -> torch.Tensor:
    """Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V])."""
    return logits_processor(self.markov_w2, markov_embed)
```

## 性能声明

SPEED-Bench(vllm bench serve,thinking on,温度 1.0,probabilistic acceptance,max output 4096):DSV4-Pro-DSpark vs MTP 平均接受长度(DL1-DL6)为 1.82/1.86(0.98x)、2.43/2.36(1.03x)、2.86/2.55(1.12x)、3.16/2.59(1.22x)、3.34/2.60(1.28x)、3.42/2.60(1.31x);coding 类 DL6 达 1.42x。步时:8xB300 上 BS1、7 draft tokens 时验证约 11-13ms,DFlash backbone 0.6ms,自回归采样约 0.6ms,E2E 步时约 14ms,AL 约 5 时 >350 TPS(coding + no-thinking)。TP-only Pareto 图(BS32/BS8/BS1)随 PR 给出,作者注明仅 TP、无 EP/DP,且用按类别校准的强制 AL。GSM8K 高并发(有/无 prefix caching)分数正常。

## 验证与回归风险

新增 529 行 `test_dspark_noncausal_sparse_mla.py`(fp8_ds_mla round-trip、future-dominance 差分测试证明 kernel 真的在attend未来索引)、`tests/models/test_registry.py` 对 ROCm/XPU skip DSparkDraftModel;开发中修复了 probabilistic drafting 两个正确性 bug(提交 d0d2c85、54b454f)与 dense 模型 CUDA graph IMA(改用 FLASH_ATTN)。风险面:qwen3_dflash.py 共享 backbone 的 cache 写路径改动;speculative config 的方法白名单与 V2 runner 强制;DSparkDraftModel 仅 NVIDIA 实现(ROCm/XPU stub 为 None)。

## 关联

同区域:#45953(DSD 动态 drafting,后续统一);DFlash2(#52816,同 backbone 体系,基准对比对象)、Kimi K3 DSpark(#50000)、Gemma4-DSpark 明确留作后续。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
