# RFC #46707: [RFC]: Support for Domino speculative decoding in vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46707](https://github.com/vllm-project/vllm/issues/46707) |
| 作者 | GIREESH7963 |
| 创建日期 | 2026-06-25 |
| 状态 | ⚪ closed(关闭于 2026-10-31) |
| labels | RFC, speculative-decoding, stale |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Speculative Decoding |

## 一句话结论
提议在 vLLM 支持 Domino 推测解码:在既有 DFlash 并行草稿器上加约 56M 参数的 GRU+低秩 MLP 校正头,于 logit 空间注入因果信息。

## 提案动机
vLLM 已支持 EAGLE/EAGLE-3、MTP、PARD、DFlash、draft-model 等推测解码方法,但没有 Domino;公开的 Domino draft checkpoint(如 `Huang2020/Qwen3-8B-Domino-b16`)与作者参考实现均已存在,但参考实现按论文说法"mainly adapted to SGLang",vLLM 下无法使用。Domino(Huang et al., arXiv 2605.29707)针对推测解码的 quality–cost 权衡:自回归 drafter(EAGLE)建模草稿 token 间因果依赖但付出逐 token 串行成本;并行 drafter(DFlash)一次 forward 生成整块却丢失块内因果结构。Domino 保持并行 drafting——并行 backbone 一次产出初步块 logits,再由轻量 Domino head 以低秩残差校正的形式在 logit 空间注入因果信息,避免重复执行 draft model 或完整 LM head。关键框架:公开 checkpoint 的 `architectures` 就是 `["DFlashDraftModel"]`,Domino head 声明为 `dflash_config.projector_type = "domino"`——Domino 是 vLLM 已有 DFlash 之上的一个 head,而非新架构,这使集成是可行的。

## 提出的方案
- 不从零写 drafter:在 vLLM 现有 `DFlashProposer` 路径上加约 56M 参数的 Domino 校正头(GRU + 低秩 MLP);verifier 不改动。
- 新增 `DominoProposer`(镜像 `DFlashProposer`):继承 DFlash 的 cross-attention 输入布局与 hidden states stash(`pass_hidden_states_to_model=True`);由于并行 drafter 单次采样调用即持有全部 K 个位置的 base logits(request-major,reshape 为 `[N, K, V]`),Domino 校正在被覆盖的采样步内完成:位置 < `pure_draft_prefix_len` 直接 argmax 不校正;GRU 以 `[bonus, prefix_drafts]` embeddings 预热;循环内 `Δ_i = embed_proj(cat([H_i, S_{i-1}]))`、`logits_i = base_i + Δ_i`,argmax 后推进 GRU。
- 新 draft-model 类 `DominoForCausalLM` 载入头权重(已对公开 b16 checkpoint 的 safetensors manifest 逐键验证):`prefix_gru.weight_ih_l0 [3072, 4096]`、`weight_hh_l0 [3072, 1024]`、`embed_proj.0.weight [256, 5120]`(5120 = 4096 + 1024 的拼接输入)、`embed_proj.2.weight [151936, 256]`(低秩 r=256 → 全词表 151936);整个头无任何 bias 张量(须以 `bias=False` 构造);无 `d2t` remap,校正输出全词表 logit bias。
- 配置/注册:注册 `method="domino"` 路由到 `DominoProposer`;draft config 须在 `dflash_config` 内携带 `parallel_drafting` 与 `mask_token_id`(b16 为 151669),否则 `_init_parallel_drafting_params` 会报错。
- 方案取舍:否决 Alternative A(重写 `propose()` 的 K 步循环贯穿 GRU 状态,复制面大、易漂移),选择 B(保留 DFlash 单 pass 布局,顺序 GRU 校正放进单个采样步覆盖)。
- 分四个离散 PR 落地:模型类+权重加载 → proposer+注册 → 正确性(与参考 legacy cuDNN-GRU 路径做贪心解码 token-parity,目标为逐位置精确对齐)→ benchmark+docs。
- 开放问题:`DominoProposer` 是否直接继承 `DFlashProposer`;`llm_base_proposer.py` 中 head-offset 特判(`method != "dflash"` 会取 1)是否适用;新并行 drafting 方法的 draft-config schema;确认走 block verification 而非 tree decoding。

## 优化点与收益
正文照抄作者的数字(论文结果,测于 A100,作者技术栈为 SGLang/Transformers,**尚未在 vLLM 下复现**):
- Qwen3 上最高 5.49× 端到端加速(Transformers backend)、最高约 ~5.8× 吞吐(SGLang)。
- 对比 DFlash:参数仅 +56M(+5.3%),draft-then-verify 延迟只 +2.8%,但 average acceptance length +16.6%、端到端加速 +12.3%。
- Head 消融(同模型,correction off→on):avg acceptance length 3.49 → 4.19,avg speedup 2.84× → 3.31×。
RFC 明确告诫:不应声称论文加速会在 vLLM 下复现;vLLM 初步验证在 T4 上且以正确性为目标(T4 不代表 Domino 目标硬件, fused Triton kernels、CUDA Graphs 等效率工程不在 v1 范围),实测正是本工作的目标。

## 现状与落地
已关闭(关闭于 2026-10-31),labels 含 stale。正文给出四阶段 PR 计划与多个待维护者定夺的开放问题(subclassing、head-offset、config schema、block vs tree),属设计评审阶段;未列出实现 PR,也未写明是否已实现;落地情况需查关联 PR。

## 关联
- Domino 论文: https://arxiv.org/abs/2605.29707
- DFlash 论文(Chen et al.): https://arxiv.org/abs/2602.06036
- 公开 checkpoint:`Huang2020/Qwen3-8B-Domino-b16`(Hugging Face Hub)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
