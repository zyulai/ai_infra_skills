# PR #47808: [Spec Decode] DSpark confidence-scheduled verification

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LucasWilkinson |
| 合并 | 2026-08-12 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, performance, speculative-decoding, ready, ci/build, v1, qwen, nvidia, mrv2 |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含约 28 个文件(22 个超长文件被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/47808 |

## 一句话结论

按 per-request 置信度自适应裁剪 DSpark 投机验证预算,高并发下保住 spec-decode 收益(fixed-k 在 c=256 反而比不投机慢 33%)。

## 动机

fixed-k 投机解码在高并发下崩塌:GPU 饱和后,draft token 与真实 token 争夺同一份算力,每请求固定验证 7 个 draft 的算力开销超过被接受 token 带来的收益,吞吐降到低于非投机解码(c=256 时 fixed 7-token 比 no-spec 慢 33%)。per-position 接受率随位置快速衰减,GPU 未饱和时空槽近乎免费,饱和后则是真实开销;交叉点随负载与工作负载接受率移动,因此任何静态 `num_speculative_tokens` 都无法覆盖全部并发区间。

## 优化点

每步决定验证多少 draft 而非全量验证:(1) Triton kernel 按 survival probability(per-position confidence 的 cumprod)对每个 (request, position) draft 槽位打分,在全局预算内贪心录取前缀,最大化预估 accepted tokens/ms——槽位跨请求竞争,自信请求的位置 5 可以胜过迟疑请求的位置 1,一个请求可保整块而另一个可能只留 1-2 个 token;(2) 步代价模型在启动时用 dummy steps 对 captured graphs 建档,在/低于 cudagraph capture limit 时取阶跃值,高于则线性插值(可用 `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` 调节建档 KV 上下文,默认 8192);(3) batch 级预算在 CPU 侧用双缓冲的 stale confidences 计算(无 GPU 同步),per-request 分配在 GPU 侧用 live values;(4) per-request confidence 用 EMA 平滑(`adaptive_verification_ema_alpha` 默认 0.8,实测为各 batch size 下的 bias-zero crossing)。为此 decode cudagraph 改为 varlen:按 token grid 捕获、经 `max_query_len` 分发、每个捕获槽非空,要求 `AttentionCGSupport.ALWAYS`(DSV4 backends 在 SM100 上满足);DSpark confidence head 正式接入推理(qwen3_dspark 与 DSV4 dspark 模型加载 confidence_head 权重,原先直接丢弃)。

## 关键改动

```text
vllm/v1/spec_decode(adaptive verification 管理器,Triton 槽位打分与预算分配;diff 被省略)
vllm/config/speculative.py: enable_adaptive_verification 开关,仅允许 method=dspark
vllm/models/deepseek_v4/sparse_mla.py + nvidia/flashmla.py: 拆出 DeepseekV4SparseMLABackend 基类,FlashMLA builder 声明 AttentionCGSupport.ALWAYS(varlen decode graph)
vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py: decode max_q_len 改取 swa_metadata.max_decode_query_len,移除 CPU 侧 .item() 同步
vllm/model_executor/models/qwen3_dspark.py: DSparkConfidenceHead 从"权重丢弃"变为接入推理(compute_confidence)
```

## 性能声明

DeepSeek-V4-Flash-DSpark、TP=4、SM100 (B300)、`speed_bench` 256 prompts / 512 output tokens(吞吐在 commit 32881e 测量;输出 tok/s 曲线为图片,正文未给具体数值):Adaptive 在 c≤64 时与 fixed-k 差距 ±3% 以内,高并发下保住投机收益;c=256 时 fixed 7-token 比 no-spec 慢 33%,而 adaptive 保持 spec-decode 优势;接受长度在 c=256 从约 3.9 降到 3.5(预算主动裁剪)。注意:no-spec 组曾与节点另一半的 co-tenant 任务共享机器,数字"方向正确但略被压低";adaptive/fixed 组在空闲节点测得。精度:GSM8K 0.945 (c=16) / 0.951 (c=64),0 invalid(阈值 0.84);MTBench 80/80 完成,人工检查无重复/退化。

## 验证与回归风险

测试:`pytest tests/v1/spec_decode/test_adaptive_verification.py tests/v1/spec_decode/test_dynamic_sd_cug.py tests/v1/cudagraph/ -q` 全过,pre-commit 全绿;新增 GSM8K eval 配置 `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-confidence-TP4.yaml`;`test_gpu_batch_ordering.py` 增加 adaptive verification 仅对含 draft 的 batch 参与尺寸计算的测试。已知限制与风险:输出 logprobs 与该模式互斥(验证在 forward 后压缩 logits,启动/请求校验拒绝);非 SM100 回退 PIECEWISE decode graphs(正确但无 varlen FULL-graph 吞吐收益);不支持 LoRA、pipeline parallelism、`--enforce-eager`(启动期硬校验);sparse MLA backend 类层级重构(FlashMLA/ROCm Aiter/FlashInfer/XPU 子类改继承新基类)影响所有 DeepSeek-V4 attention 后端。

## 关联

同区域:DSpark 投机解码方法族(qwen3_dspark/dspark draft、probabilistic draft sampling);文档新增 `docs/features/speculative_decoding/adaptive_verification.md`,与 Dynamic Speculative Decoding 并列。pack 素材未列出明确的前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
