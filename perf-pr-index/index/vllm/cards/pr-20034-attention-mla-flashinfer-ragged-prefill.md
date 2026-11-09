# PR #20034: [Attention] MLA - Flashinfer Ragged Prefill

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @alexm-redhat |
| 合并 | 2025-07-11 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, performance, frontend, ready, v1, deepseek |
| Diff | diffstat 缺失(pack 未记录);diff 覆盖 8 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/20034 |

## 一句话结论

在 B200 上为 MLA prefill 接入 FlashInfer ragged prefill kernel,TTFT 提升 20-25%、TPOT 提升 15-20%。

## 动机

MLA 公共后端(`mla/common.py`)的 prefill 此前只走 FlashAttention(flash_attn_varlen)路径,在 Blackwell(B200)上不是最优选择;FlashInfer 的 ragged prefill 在该硬件上更快,但需要 NVIDIA 侧的一个 FlashInfer 修复(flashinfer-ai/flashinfer#1198)才能正确工作。此外 FlashInfer 后端用到的 `PerLayerParameters` 等超参推导逻辑当时私有于 flashinfer.py,MLA 路径无法复用。

## 优化点

在 MLA 公共后端内新增 FlashInfer prefill 路径:metadata builder 按 `use_flashinfer_prefill()`(Blackwell,compute capability ≥ 100 且 flashinfer 可用时默认启用)选择 `FlashInferPrefillMetadata`,持有一个主 `BatchPrefillWithRaggedKVCacheWrapper` 与按需扩展的 chunk wrapper 列表(共享 394MB workspace buffer,取值与 sglang 一致)。plan 阶段以 `qo_indptr`/`kv_indptr` 规划:main run 为 causal,chunked prefill 的 context chunk 为非 causal,MLA 下 non-latent 的 num_qo_heads == num_kv_heads,`head_dim_qk = qk_nope_head_dim + qk_rope_head_dim`。forward 侧通过 `_run_prefill_new_tokens_fi`/`_run_prefill_context_chunk_fi` 分派,FlashAttention 路径原样保留为 `_fa` 变体;FI 路径下 `_pad_v = False`。同时把 `get_per_layer_parameters`/`infer_global_hyperparameters` 提升到 `backends/utils.py` 并参数化 impl 类型供两条后端共用。

## 关键改动

```text
vllm/v1/attention/backends/mla/common.py: FlashInferPrefillMetadata、
  use_flashinfer_prefill、_build_fi_prefill_wrappers、FI/FA prefill 分派
vllm/v1/attention/backends/utils.py: 收纳 PerLayerParameters、
  get_per_layer_parameters(按 impl 类参数化)、infer_global_hyperparameters
vllm/v1/attention/backends/flashinfer.py: 改用共享的超参推导工具
vllm/attention/utils/kv_sharing_utils.py: validate_kv_sharing_target 迁移至公共位置
```

关键机制摘录(common.py,B200 默认启用):

```python
def use_flashinfer_prefill() -> bool:
    if flashinfer_available:
        # For blackwell default to flashinfer prefill if its available since
        #  its faster than FA2.
        return current_platform.has_device_capability(100)
    return False
```

## 性能声明

deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,单卡 B200,10000/100 prompt/output(输入/输出长度),多种 batch size:

- TTFT 提升 20-25%;
- TPOT 提升 15-20%。

具体逐 batch size 曲线见 PR 截图(数值未收入 pack 素材)。

## 验证与回归风险

依赖 FlashInfer 版本包含 flashinfer#1198 修复,未装或非 Blackwell 自动回退 FA 路径(`flashinfer_available` 检测 + capability 判断)。风险点:394MB 额外 workspace 显存占用;chunked prefill 需要 per-chunk wrapper,`num_chunks` 增长时按需新建;全局超参推导要求所有层 `window_left`/`logits_soft_cap`/`sm_scale` 一致,否则 assert 失败(两条后端同样受约束)。`can_run_in_cudagraph` 行为未变(仅 max_query_len == 1)。本 PR 还顺带迁移了 `validate_kv_sharing_target` 与 `test_multi_connector.py` 的测试工具类,属结构性整理。

## 关联

- 依赖:flashinfer-ai/flashinfer#1198(NVIDIA 侧修复)
- 同区域:vLLM 索引中的其他 MLA 后端优化,如 PR #20769(SM100 CUTLASS MLA decode)、PR #21078(FlashInfer TRTLLM-gen MLA decode)

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
