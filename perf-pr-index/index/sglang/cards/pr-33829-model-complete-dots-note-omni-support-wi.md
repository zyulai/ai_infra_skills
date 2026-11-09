# PR #33829: [Model] Complete dots.note.omni support with native encoders, video preprocessing, and MTP decoding

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jianfei-wangg |
| 合并 | 2026-08-22 |
| 分类 | Speculative Decoding(classify.py 给出,tier: major) |
| Label | documentation, high priority, Multi-modal, deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight, memory-pool |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 片段约 9 个文件,另有 46 个文件因超长被裁剪(大型模型集成 PR) |
| 链接 | https://github.com/sgl-project/sglang/pull/33829 |

## 一句话结论

完整接入 dots.note.omni 多模态模型:原生视觉/音频 encoder、训练一致的视频预处理,以及全共享 MTP/NextN 投机解码与混合 SWA 注意力支持。

## 动机

dots.note.omni 不满足现有多模态与投机解码路径的多项假设:自定义 MoE ViT + Whisper 系音频 encoder、需要与训练一致的联合采样/交织视频 pipeline(通用均匀抽帧处理器会造成推理/训练不匹配);语言模型混合 MLA + full-attention/sliding-window 双几何 + 可选 DSA + attention gate,full-attn 与 SWA 层不能共用同一注意力路径或 KV-cache 记账;其 MTP 层是 SWA 形状 + MLA 基底的普通 SWA 层,现有 hybrid pool 配置器把所有 EAGLE/NextN draft 层按 full-attention 记账,会导致分配错误。

## 优化点

- 模型接入:新增 `Dots3Config`(model_type `dots3_note`,is_hybrid_swa)与 dots MoE ViT、Whisper 系音频 encoder 的原生实现,使用 SGLang FA3 与 fused MoE kernel,移除对 mm_encoder_server 的运行时依赖;支持 `--language-only` 做 encoder/LLM 分离部署。
- 全共享 MTP/NextN:`configure_draft_model()` 在派生 attention 形状前重写 draft 配置,暴露恰好一个递归共享 MTP 层并改用 dots3 SWA 几何,目标 LM head 与 draft 共享;target verify/draft extend 走 paged absorbed SWA-MLA 路径。
- 共享基础设施 1(hybrid SWA draft KV 记账):speculative 辅助配置记录 SWA draft 层数,hybrid pool 配置器区分 full-attention draft 层、按 target SWA capacity 记账的普通 SWA draft 层与 full-capacity SWA 形状 draft 层,避免 under-allocation 与 target KV 容量损失。
- 共享基础设施 2(DP dummy 行归一化):DP attention + overlap 调度下,speculative 元数据可能在 DP padding 前已规划;本 PR 在 draft buffer 分配前归一化 dummy 行,保持 CPU/GPU 元数据对齐,并提供 backend hook 在 DP padding 后刷新行相关元数据(默认 no-op,只有可安全重建的 backend opt-in,不触碰预规划稀疏/DSA 调度)。
- 共享基础设施 3(FA3 SWA-MLA fallback):FA3 新增 absorbed SWA-MLA fallback,直接消费 SWA page table 与 latent KV cache,作为 backend capability 由模型按需选择;另加 page table 与 DP 归一化 batch 对齐的共享 helper。
- 请求链路:`video_config` 请求级控制(seq/audio_cap/audio_sr/k_mode)贯穿 ChatCompletionRequest → GenerateReqInput → dots processor;新增 dots function-call 格式 detector。

## 关键改动

```text
python/sglang/srt/configs/dots3.py: 新增 Dots3Config,含 configure_draft_model()/backend wrap 钩子
python/sglang/srt/layers/attention/dots_hybrid_backend.py: 新增,层级 DSA/SWA attention dispatch
python/sglang/srt/configs/model_config.py: draft 配置改写、SWA 几何派生、hybrid layer ids
python/sglang/srt/layers/attention/attention_registry.py: dots draft decode/extend backend 包装接入
python/sglang/srt/function_call/dots_detector.py: 新增,dots 工具调用格式解析
```

```python
def configure_draft_model(self) -> str:
    """Configure the recursively shared MTP layer with SWA geometry."""
    self.num_nextn_predict_layers = 1
    self.layer_types = ["sliding_attention"]
    self.attention_gate_type = self.swa_attention_gate_type
    self.kv_lora_rank = self.swa_kv_lora_rank
    self.qk_nope_head_dim = self.swa_qk_nope_head_dim
    self.num_attention_heads = self.swa_num_attention_heads
    ...
    return "Dots3NoteForCausalLMNextN"
```

## 性能声明

PR 正文未提供 benchmark 或性能对比数据(Accuracy/Speed 章节为空模板)。性能相关收益定性为:full-sharing MTP/NextN 投机解码可用、DP/TP/EP(含 overlap 调度)可执行、hybrid SWA draft KV 记账避免 KV 容量损失;均无数值佐证。

## 验证与回归风险

新增共享正确性测试:DSA K-only rotary 路径不 alias query/key、draft-extend dummy 请求 CPU/GPU 长度对齐、idle speculative batch 保留归一化 dummy logits/hidden states、hybrid-SWA draft 层按 SWA 几何与容量做 KV sizing;另附 EP8/TP8/DP8 + DeepEP + NEXTN + FA3 draft 的启动脚本。CI 快照 Base 通过、Extra 失败、AMD ROCm 7.2 排队中。风险面广:触碰 hybrid KV pool 配置器、DP dummy 行归一化、attention registry 与 FA3 backend 等 shared 路径,影响所有 EAGLE/NextN + hybrid-SWA 模型;新增 backend hook 默认 no-op,仅 opt-in backend 重建元数据。

## 关联

同区域:EAGLE/NextN 投机解码、hybrid SWA KV pool(Inkling 案例)、DSA/DeepSeek 系模型支持(Dots3NoteForCausalLM 被加入 DSA 架构清单);与 DeepSeek/DSA 分支 overrides(_deepseek_family_overrides)共享注册路径。PR 正文未列出具体关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
