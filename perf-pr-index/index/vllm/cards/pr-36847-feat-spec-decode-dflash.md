# PR #36847: [Feat][Spec Decode] DFlash

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benchislett |
| 合并 | 2026-03-30 |
| 分类 | Speculative Decoding |
| Label | new-model, speculative-decoding, ready, v1, qwen, nvidia |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/36847 |

## 一句话结论

引入 DFlash 投机解码 drafter:query tokens 对 target hidden states 做双向注意力,1 并发低时延场景最高 4.6x 输出 tok/s。

## 动机

DFlash 与 P-EAGLE(#32887)同族,但架构改为:drafter 的注意力发生在"最后采样 token + 一批占位 mask token"与"context states(target 模型 prefill 或已接受 token 的 hidden states)"之间,是双向非因果注意力。这要求把 context states 直接写入 KV cache 的额外操作,而不能复用标准因果注意力的执行路径,因此需要专门的 proposer 实现与调度支持。

## 优化点

核心机制是 `precompute_and_store_context_kv`:在主模型 forward 之外,把 context states 一次性投影为所有层的 K/V——先 `ops.rms_norm`,再用拼接的大权重做一个 fused GEMM 完成全部层的 KV 投影,permute 成 layer-major 连续布局后逐层 K-norm,最后用 `ops.rotary_embedding` 对 L*num_ctx 的大 batch 一次做 RoPE,再逐层写入 KV cache。该段不进 torch.compile/CUDA graph,但写满 cache 后用 `seq_lens_gpu` 忽略被拒绝的 token,主 forward 只跑 query tokens,从而兼容 async scheduling 与 (piecewise) CUDA graph(初版担心的 `disable_padded_drafter_batch`/CUDA graph 问题在 PR 内标注已解决)。`speculative_config` 将 dflash 设为 `parallel_drafting=True`,并复用 EAGLE 配置加载路径;attention selector 新增 `use_non_causal` 校验。

## 关键改动

```text
vllm/v1/spec_decode/dflash.py: DFlashProposer(投机逻辑,覆盖 eagle.py 的两个 hook)
vllm/model_executor/models/qwen3_dflash.py: DFlashQwen3 drafter 模型(fused KV 投影 + cache 直写)
vllm/v1/spec_decode/eagle.py: 抽出 build_model_inputs_first_pass / build_per_layer_attn_metadata 供 override
vllm/v1/attention/backend.py, flash_attn.py, selector.py: supports_non_causal 校验(仅 FA 返回 True)
```

## 性能声明

1xB200、Qwen3-8B(drafter `z-lab/Qwen3-8B-DFlash-b16`,`num_speculative_tokens=15`,attention backend flash_attn),按 DFlash 官方方法论在 Alpaca/GSM8k/HumanEval/Math500/MBPP/MT-Bench 上测输出 tok/s,conc 1/8/16/32:speedup 范围 Alpaca 1.179–2.819x、GSM8k 1.603–3.509x、HumanEval 2.050–4.633x、Math500 1.867–4.019x、MBPP 1.619–3.866x、MT-Bench 1.301–2.982x;并发越低加速越高(如 HumanEval conc=1:209.16→969.07 tok/s)。

## 验证与回归风险

新增 GSM8k 正确性与 acceptance rate 单元测试,复现 DFlash 论文的 acceptance rate 数值;Qwen3-8B 与 Qwen3.5-9B 在 FA2/FA4 下通过测试套件。兼容性风险:非因果注意力仅 Flash Attention 支持(Triton Attention、FlashInfer TRTLLM 均不支持),attention selector 会在不满足时给出 invalid reason;DFlash drafter 的 torch.compile 初版未打通。另在 `VllmConfig` 中对 spec decode 下 `max_num_scheduled_tokens` 过小的情况加了报错与性能告警。

## 关联

同区域:P-EAGLE #32887(前序架构)、#24322(eagle.py 复杂度重构先例)、EAGLE/MTP 系列投机解码 PR;`Qwen3NextModel` 改用 `EagleModelMixin` 统一 aux hidden states 采集。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
