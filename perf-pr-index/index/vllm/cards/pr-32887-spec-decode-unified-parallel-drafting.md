# PR #32887: [Spec Decode] Unified Parallel Drafting

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benchislett |
| 合并 | 2026-02-05 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, performance, speculative-decoding, ready, v1, llama, nvidia |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含约 15 个文件(7 个超长文件被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/32887 |

## 一句话结论

统一 EAGLE/draft-model/并行草稿的输入准备并支持并行投机(PARD、P-EAGLE),B200 上 PARD 1B 使 Llama 70B NVFP4 解码提速最高 3.10x。

## 动机

此前 v1 投机解码里 EAGLE 与 draft model 走各自独立的输入准备路径(draft_model.py 自带 merge_toks_kernel、extend_all_queries_by_1 等专用逻辑),且投机 token 只能自回归顺序生成;并行草稿类方法(AMD PARD 对微调外部 draft 模型做并行 propose、AWS P-EAGLE 对 EAGLE3 做并行 prediction)无法接入。本 PR 用单一输入准备 kernel 统一三条路径,并新增并行草稿支持(带与不带 target hidden states 两种形态)。

## 优化点

核心是重写 `SpecDecodeBaseProposer` 的输入准备:新增 `copy_and_expand_eagle_inputs_kernel`(Triton,每请求一个 program、按 max_query_len 分块),一次把 target token ids/positions、sampled next tokens 拷进预分配 drafting buffer,同时产出 rejected-token 掩码、parallel-padding 掩码、待采样位置索引与 hidden-state scatter 映射,替代原来"merge kernel + slot mapping 重算 + extend_all_queries_by_1"的多步管线;EAGLE 原路径(needs_extra_input_slots=False)保持纯 shift 逻辑不变。并行草稿配置 `parallel_drafting: bool`:每请求占用 `num_speculative_tokens` 个额外输入槽(顺序草稿只占 1),所有投机 token 一次前向并行产出(`num_speculative_tokens==1 or parallel_drafting` 时直接 argmax 返回);masked 槽填 `pard_token`/`ptd_token_id`(draft config 必须提供其一),带 hidden states 的 EAGLE 并行变体用 `parallel_drafting_hidden_state_tensor` 填充(masked 位置经 torch.where 避免 DtoH 同步),EAGLE3 并行 checkpoint 的 `mask_hidden` 权重接入加载,`llama_eagle3` 支持 attention_bias。周边适配:compile range 上界按 `num_speculative_tokens * max_num_seqs` 扩展;`reorder_batch_threshold` 上调;async scheduling 放开 draft_model;FlashInfer `get_cudagraph_support` 遍历 UniformTypeKVCacheSpecs 内全部 KV spec 判定 UNIFORM_BATCH;draft model 加载重构为 `_get_model` override(不共享 embedding/lm_head),multimodal/M-RoPE/padded-batch 校验上移到基类。

## 关键改动

```text
vllm/v1/spec_decode/eagle.py: 统一 propose/set_inputs_first_pass,copy_and_expand_eagle_inputs_kernel 接入,并行草稿参数与掩码 buffer
vllm/v1/spec_decode/utils.py: 新 kernel 与 extend_all_queries_by_N(文件在省略 diff)
vllm/v1/spec_decode/draft_model.py: 删除专用输入准备代码,改继承统一基类路径
vllm/config/speculative.py: parallel_drafting 开关
vllm/model_executor/models/llama_eagle3.py: attention_bias + mask_hidden 权重加载
```

## 性能声明

全部在 1xB200;AL 为 MTBench 各类平均接受长度,max 2048 输出 tokens。GPT-OSS 120B 2-layer P-EAGLE vs NVIDIA EAGLE3 short-context 基线:BS=1 最优 P-EAGLE K=7 约 560 输出 TPS,较基线 1.52x、较 EAGLE3 最优配置 1.12x;BS=8 P-EAGLE K=3 2388.8 TPS,较基线 1.34x、较 EAGLE3 最优 1.07x。Llama 3.3 70B-NVFP4 + AMD PARD Llama 3.2 1B,基线为自回归 drafter:BS=1 最优 PARD K=11 约 254 TPS,较基线 3.10x、较 vanilla draft-model 1.61x;BS=8 PARD K=7 1915.9 TPS,较基线 2.87x、较 vanilla draft-model 1.61x。明细表另含 EAGLE3/P-EAGLE K=2-7、PARD K=3-15 的逐点 AL/迭代时间/TPS。

## 验证与回归风险

测试:并行草稿 e2e 与输入准备逻辑单测本地通过;draft model 与 EAGLE3 既有 e2e 测试仍通过(PR 正文声明,未列具体用例)。风险面:输入准备统一改动触及所有 EAGLE/draft-model 投机器的 batch 布局与 slot mapping;并行草稿要求 checkpoint 训练时支持(缺 pard_token/ptd_token_id 启动报错,mask_hidden 缺失也报错);draft model 不再共享 target embedding/lm_head(`_maybe_share_embeddings/_maybe_share_lm_head` 显式 no-op),对曾依赖共享行为的配置是行为变化;multimodal/M-RoPE/非 padded drafter batch 在 needs_extra_input_slots 路径硬性 NotImplementError。

## 关联

同区域:引入的两个并行草稿方法 —— AMD PARD(外部微调 draft 模型并行 propose)、AWS P-EAGLE(EAGLE3 并行 prediction);EAGLE3/llama_eagle3 与 DraftModelSpeculator 基础设施。pack 素材未列出明确的前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
