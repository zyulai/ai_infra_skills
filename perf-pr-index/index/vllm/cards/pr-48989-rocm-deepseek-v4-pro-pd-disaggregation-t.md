# PR #48989: [ROCm] DeepSeek-V4-Pro PD Disaggregation through MORI IO KV Connector on AMD GPUs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @itej89 |
| 合并 | 2026-10-27 |
| 分类 | Overlap / Scheduler (tier: major) |
| Label | documentation, performance, new-model, rocm, kv-connector, deepseek, DSv4, mrv2 等多个 |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际改动 4 个文件(2 个 buildkite CI 配置 + 2 个 moriio connector 源文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/48989 |

## 一句话结论

通过 MORI IO KV Connector 在 AMD MI355X 上打通 DeepSeek-V4-Pro 的 PD 分离(TP/EP 模式),gsm8k 准确率 0.953。

## 动机

此前 vLLM 的 PD(prefill/decode)分离部署在 AMD GPU 上未覆盖 DeepSeek-V4-Pro:MoRIIO connector 对该模型的 attention backend 检测、KV cache 布局(尤其是 SlidingWindowMLASpec 类型的 MLA latent slots)与 block notify 逻辑均不兼容,且 CI 中缺少该模型的 disagg 测试流水线。本 PR 补齐这条路径,使 DeepSeek-V4-Pro 可在 MI355X-AINIC 集群上以 prefill/decode 分离方式服务。

## 优化点

机制上不是新 kernel,而是 KV connector 的适配:(1) `moriio_connector.py` 中 attention backend 检测从 `get_attn_backend(head_size, dtype, cache_dtype, use_mla)` 的单后端推断改为 `get_current_attn_backends(vllm_config)` 直接取当前实际后端;(2) WRITE 模式的 NotImplementedError 检查改为仅针对真正的 `SlidingWindowSpec`(排除 `SlidingWindowMLASpec`,其 slots 是 MLA latent,由 WRITE 路径处理);(3) `update_state_after_alloc` 的 block notify 条件从 `self._has_mamba` 扩展为 `self._has_mamba or self._is_hma_required`,让 HMA(混合 KV cache)场景也走 clamp 到 prompt blocks 的通知列表;(4) `moriio_layout.py` 的 `get_layer_transfer_geometry` 中 block_len 计算从 `num_states * slot_size_bytes` 改为按 stride 展开的精确跨度 `1 + sum((s-1)*st)`,正确处理非连续布局。另在 `.buildkite/amd-disagg` 增加 DeepSeek-V4-Pro 的 1P1D-TP8、1P1D-EP8/DP8-WideEP、2P2D-EP16/DP16 三条 MoRIIO+vllm-router CI 流水线。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py: backend 检测改用 get_current_attn_backends;SlidingWindowMLASpec 豁免;HMA 场景启用 block notify
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py: block_len 按 stride 精确计算,兼容非连续 KV 布局
.buildkite/amd-disagg/models.yaml / pipeline-disagg.yaml: DeepSeek-V4-Pro 的 TP/EP/WideEP disagg CI 用例
```

关键代码摘录(节选,<=15 行):

```python
backends = get_current_attn_backends(self.vllm_config)
self.backend_name = backends[0].get_name()
...
if self.mode == MoRIIOMode.WRITE and any(
    isinstance(g.kv_cache_spec, SlidingWindowSpec)
    and not isinstance(g.kv_cache_spec, SlidingWindowMLASpec)
    for g in kv_cache_config.transfer_groups
):
    raise NotImplementedError(...)
```

## 性能声明

- pack 未检测到吞吐/时延类性能数字;该 PR 属功能打通(适配)性质。
- 准确性验证:PD 分离部署下 lm_eval gsm8k(5-shot),DeepSeek-V4-Pro: flexible-extract exact_match 0.9530(±0.0058),strict-match 0.9522(±0.0059)。
- 部署条件(测试环境):MI355X-AINIC,TP=8,fp8 KV cache,`VLLM_ROCM_USE_AITER=1`,MORI SHMEM heap 16G,vllm-router PD disaggregation。

## 验证与回归风险

- 验证依赖新增的 Buildkite AMD disagg 流水线(1P1D TP8 / 1P1D EP8/DP8 WideEP / 2P2D EP16/DP16,queue 为 amd_mi350_ainic)与 lm_eval gsm8k;`MORIIO_READ_MODE=0`。
- 回归风险面:MoRIIO connector 的其他模型(GLM-5.2、MiniMax-M3 等同流水线模型)共享 block_len 计算与 notify 逻辑——block_len 公式变化对连续布局结果等价,但对任何依赖旧公式的非连续 spec 是行为变化;backend 检测从推断改为实际后端列表,若出现多后端仅取第一个。`--disable-hybrid-kv-cache-manager` 的 fallback 提示保留。

## 关联

- 同区域:索引库已有 pr-41751(MORI inter-node kernel)、pr-42296(simple connector DSV4)、pr-34328(flexkv connector)等 KV connector/PD 分离卡片。
- 关联外部 PR:vllm-project/router#192(router 侧 moriio TP size 修复)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
