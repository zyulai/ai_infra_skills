# PR #64368: [Spec Decode][ROCm] draft_lm_head_quantization for DSpark drafters

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @AMD-yanfeiwang |
| 合并 | 2026-10-31 |
| 分类 | Speculative Decoding |
| Label | rocm, speculative-decoding, qwen, quantization, mrv2, verified, dflash |
| Diff | diffstat 缺失(pack 中为 +None/-None, None files);diff 含 7 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/64368 |

## 一句话结论

让 DeepSeek-V4/V4.1 DSpark drafter 也能用 `draft_lm_head_quantization`:draft 用共享 target lm_head 的 FP8 私有副本算 base logits,MI355X 上 c1 ITL P50 降约 2.4–4.0%。

## 动机

#59973 引入的 `draft_lm_head_quantization` 此前只覆盖 EAGLE 和 MTP:drafter 若共享 target 的 lm_head,只能用 BF16 大 GEMM 算 draft base logits。在 ROCm MI355X 上,c1(M=5)时 BF16 dispatch 选中 `wvSplitK` kernel,比 M=10 时慢约 3.5 倍,draft-head GEMM 单步高达约 348 µs,是小并发 DSpark 解码的明显瓶颈。该能力是 sglang 对应改动(sgl-project/sglang#39622)的 vLLM 版。

## 优化点

不改 kernel 机制,是能力面的扩展与重构:把 EAGLE 与 DSpark 共有的"安装共享 lm_head 或其量化副本"逻辑抽成 `eagle/utils.py` 中的 `share_target_lm_head` helper(内部复用 `_should_share` 与 `quantized_lm_head_copy`,请求量化但 drafter 无法共享时报错),EAGLE/MTP 路径与 `load_dspark_model` 都改为调用它;`load_dspark_model` 把 vocab-size 检查作为 `can_share` 传入。`DSparkSpeculatorConfig.supported_options` 加入 `draft_lm_head_quantization`,仅对 `DSparkDraftModel`/`DSparkV41DraftModel` 放行(其他 DSpark drafter 如 Muse Spark 会直接读 `lm_head.weight`,在 config 校验阶段即拒绝);EAGLE 原有的 `head_dtype` 校验抽为共享的 `check_draft_lm_head_quantization`。ROCm 上 `"fp8"` 解析为 `RowWiseTorchFP8ScaledMMLinearKernel`(hipBLASLt FP8 `_scaled_mm`),per-row 权重 + per-token 激活 scale;target 仍用自己的 BF16 head 验证。默认关闭。与 #47584 不重复:后者是 NVIDIA 专用的 env var + 新 module,本 PR 复用既有 config 与 kernel 选择,任何有 FP8 linear kernel 的平台可用。

## 关键改动

```text
vllm/v1/worker/gpu/spec_decode/eagle/utils.py: 新增 share_target_lm_head helper,统一 EAGLE/DSpark 头共享与量化副本安装
vllm/v1/worker/gpu/spec_decode/eagle/config.py: 抽出 check_draft_lm_head_quantization 共享校验
vllm/v1/worker/gpu/spec_decode/dspark/config.py: supported_options 加入 draft_lm_head_quantization 并限制 drafter 架构
vllm/v1/worker/gpu/spec_decode/dspark/utils.py: load_dspark_model 改用 share_target_lm_head
vllm/config/speculative.py: 文档更新(EAGLE, MTP → EAGLE, MTP and DeepSeek-V4/V4.1 DSpark)
```

helper 核心:

```python
head = target_lm_head
if quantization is not None:
    head = quantized_lm_head_copy(target_lm_head, quantization)
if draft_lm_head is not None:
    del draft_model.lm_head
draft_model.lm_head = head
```

## 性能声明

测试条件:MI355X,DeepSeek-V4.1-Flash,InferenceX recipe `dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml`(DSpark k=5,probabilistic drafting,`--monokernel`),双臂仅差 `"draft_lm_head_quantization": "fp8"`。

- AgentX ITL P50(BF16→FP8 head):TP2 c1 1.89→1.81 ms(−4.0%),c2 −4.9%,c4 −4.5%,c8 起 0.7% 内中性;TP4 c1 1.56→1.52 ms(−2.4%),c2 −2.9%。作者注明 c2/c4 时 trace 只能解释约 0.5%(c4 每步 44 µs),其余为 run-to-run 方差。
- torch profiler trace(TP2 rank 0):draft-head GEMM c1(M=5)`wvSplitK_hf_sml` 347.8 µs → `Cijk_..._F8BS` 54.6 µs + 每步 5.0 µs per-token 量化;c4(M=20)104.5 → 55.8 + 5.0 µs;target verify head(约 104 µs)不受影响。
- 单独 GEMM microbench(CUDA-graph replay):TP2 64640×5120 M=5 BF16 352.4 → FP8 57.0 µs;M=320 185.0 → 171.6 µs;TP4 M=5 183.9 → 33.7 µs。
- 精度:acceptance length 3.416→3.400(c8,约 112k drafts);GSM8K 5-shot 两次运行 0.909/0.913 → 0.904/0.914,均在 run-to-run 噪声内。内存代价:每 rank 多一份 FP8 head 副本,约 0.33 GB(TP2)/0.17 GB(TP4)。

## 验证与回归风险

测试:MI355X 上 `test_qwen3_omni.py`、`test_gpu_draft_lm_head.py`、`test_custom_speculator.py` 共 54 passed、4 skipped(NVFP4 case 仅 CUDA);EAGLE/MTP 重构路径由 `test_mtp_draft_lm_head_quantization`(fp8)覆盖并通过;pre-commit 含 mypy 通过。CI 缺少 DSpark checkpoint 的对应测试,DSpark 路径为上述 e2e 验证。风险面:量化副本只影响 draft logits(可能影响 acceptance length,实测未变),target 验证路径不变;weight reload/transfer 经 `refresh_quantized_lm_heads`/`share_source_lm_heads` 已覆盖副本刷新;`"nvfp4"` 走同一路径但本 PR 未测量。DSpark k=5 时 adaptive verification 不依赖 draft logits(`adaptive_verification_uses_draft_logits = False`)。

## 关联

前序:#59973(引入 `draft_lm_head_quantization`,限 EAGLE/MTP,本 PR 扩展到 DSpark)。相关但不重复:#47584(NVIDIA DeepSeek-V4 DSpark FP8 draft head,env var 机制)。后续可能:PR 提到 `load_dflash_model` 可在后续以 `quantization=None` 采用该 helper。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
