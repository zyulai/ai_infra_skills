# PR #42175: [Core][Model] Gemma4: Unified FA4 for all layers + FlashAttention mm_prefix support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lucianommartins |
| 合并 | 2026-06-10 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, speculative-decoding, ready, ci/build, v1, kv-connector, nvidia |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 内容可用,14 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/42175 |

## 一句话结论

Gemma4 在 Hopper 上全层统一 FA4(替代混合 FA3+FA4 或 TRITON_ATTN),并给 FlashAttention 后端补上 mm_prefix 双向注意力支持:prefill/长上下文吞吐 +25-70%,TTFT -22~-31%。

## 动机

Gemma4 各层 head_dim 异构:滑窗层 head_dim=256,全注意力层 global_head_dim=512。此前 `Gemma4Config` 检测到不匹配即强制 TRITON_ATTN(唯一能同时处理两种尺寸的后端);手动选 FLASH_ATTN 时会逐层混用 FA3(滑窗)与 FA4(全注意力),基准显示混合执行比全 FA4 慢约 8%。另外多模态 token 的双向注意力(mm_prefix/PrefixLM,Gemma4/Gemma3/PaliGemma/Molmo2 等都用)`FlashAttentionBackend.supports_mm_prefix()` 返回 False,这些模型被锁死在 Triton/FlexAttention,长上下文(多模态校验激活处)用不上 FA4。

## 优化点

(1) 后端选择:Gemma4Config 在 FA4 可用且 `max_head_dim <= 512` 时为所有层设 `flash_attn_version=4`(统一 kernel 路径,尊重用户显式覆盖),FA4 不可用才回落 TRITON_ATTN;(2) mm_prefix 元数据标准化:`CommonAttentionMetadata` 新增 `mm_req_doc_ranges`(请求索引 → 双向 range 列表),PrefixLM 双向范围走标准 metadata build 路径而非 build 后 monkey-patch;gpu_model_runner 把 `req_doc_ranges` 计算前移并删除 `_set_mm_prefix_range_for_metadata` 等方法,Triton/FlexAttention 及各 MLA 后端的 builder 同步从 common metadata 取值;(3) FlashAttention 后端 `supports_mm_prefix()` 在 FA4 可用时返回 True,实现两段调用分解:`_apply_mm_prefix_correction()` 先跑主因果 `flash_attn_varlen_func`(文本 token 正确),再对 mm_prefix range 跑非因果调用(修正多模态 token),用 `merge_attn_states` 按 LSE 重标定合并;`_precompute_mm_prefix_indices()` 在 build 阶段于 CPU 预计算修正索引,避免 forward 中 GPU tensor 的 `.item()` 同步;纯文本 batch 修正被 `mm_prefix_range_tensor is not None` 门控,零开销。

## 关键改动

```text
vllm/model_executor/models/config.py: Gemma4Config 改为 FA4 全层统一/回落 Triton 的选择逻辑
vllm/v1/attention/backend.py: CommonAttentionMetadata.mm_req_doc_ranges 字段 + supports_combination 增加 use_mm_prefix 维度
vllm/v1/attention/backends/flash_attn.py: supports_mm_prefix、mm_prefix metadata、_apply_mm_prefix_correction 两段调用 + LSE 合并
vllm/v1/attention/backends/triton_attn.py, flex_attention.py, mla/*.py: builder 从 common metadata 接 mm_prefix
vllm/v1/worker/gpu_model_runner.py: req_doc_ranges 计算前移,删除 post-build 修补方法
```

关键代码(后端选择):

```python
if is_fa_version_supported(4) and max_head_dim <= 512:
    if (vllm_config.attention_config.flash_attn_version is None
            and vllm_config.attention_config.backend
            in (None, AttentionBackendEnum.FLASH_ATTN)):
        vllm_config.attention_config.flash_attn_version = 4
```

## 性能声明

8x H100 SXM 80GB,TP=2(31B)/TP=1(E2B、E4B),全 Gemma4 尺寸,FA4 vs Triton:吞吐——Prefill 4K 输入 E2B +40%/E4B +24%/26B-A4B(MoE) +25%/31B +33%;长上下文 8K +41%/+28%/+27%/+28%;超长 15-16K +70%/+47%/+36%;mixed(1K/1K)与高 batch(256 in/out)约 +1~5%。P50 延迟——Prefill TTFT -31%/-22%/-24%/-22%,8K 长上下文 decode -27%/-19%/-20%/-16%,decode b=1/b=8 中性(约 +/-4% 内)。总结:FA4 赢下全部 4 个模型尺寸的所有吞吐场景,prefill/长上下文 +25-70% 吞吐、TTFT 快 22-31%;短上下文 decode 中性(权重加载主导);uniform FA4 比混合 FA3+FA4 快约 8%;mm_prefix 两段分解对纯文本零开销、多模态 batch 约 3%(kernel launch 开销)。功能测试 H100 SXM 80GB 全 PASS。

## 验证与回归风险

Test plan 含 FA4 自动选择断言脚本与 Gemma4 图像 Triton 路径冒烟;功能测试表(E2B 文本/图像 FA4、E2B 图像 Triton)全 PASS。风险面:`supports_combination` 签名加 `use_mm_prefix`,所有 attention backend(含 5 个 MLA 后端)同步改动;Gemma4Config 只影响 Gemma4 架构,但 mm_prefix 走 common metadata 影响 PrefixLM 模型家族(Gemma3、PaliGemma、Molmo2、Moondream3、Bagel 等);两段调用分解的数值合并依赖 merge_attn_states 的 LSE 重标定正确性;mm_prefix 深度优化(range-id 查表)见 PR #50294。

## 关联

同区域:PR #50294(FA4 mm_prefix range lookup 优化,本 PR 引入两段调用后的后续)、PR #65761(encoder cudagraph 修复,Gemma4 多模态相关)、PR #57250(DiffusionGemma 后端选择注释关联);Gemma4-DSpark 留待后续(#46995 提及)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
