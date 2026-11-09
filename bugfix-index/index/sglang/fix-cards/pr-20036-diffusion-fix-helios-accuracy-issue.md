# PR #20036: diffusion: fix helios accuracy issue

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20036](https://github.com/sgl-project/sglang/pull/20036) |
| 作者 | yhyang201 |
| 合并日期 | 2026-03-15 |
| labels | amd, run-ci, diffusion |
| 热度 | 👍 0 · 💬 19 |
| 分类 | Other |
| bug 类型 | correctness |

> 注:本 PR 正文为空模板(各节均为占位注释),未披露现象与根因;下文机制描述均来自 diff。素材中的 diff 亦在 40000 字符处截断(文件粒度),截断后的改动未包含在素材中。

## 一句话结论

正文为空;由 diff 看,Helios 文生视频管线存在多处与 diffusers 参考实现不一致之处(文本序列长度、序列并行分片、VAE 分块解码、SP 噪声种子),逐一修正以恢复精度。

## 问题现象

正文未描述具体现象(正文全部为模板占位注释)。从标题与 diff 推断,症状是 Helios(multimodal_gen 扩散管线)生成结果精度/正确性问题:输出与 diffusers 参考实现不一致——文本编码截断到 226 长度、未启用 sequence parallelism 的实现路径、整段 latents 一次性 VAE 解码产生 chunk 边界伪影、以及 SP rank 间噪声不同步。(由 diff 推断)

## 根因

正文未披露根因。由 diff 可描述的机制(均由 diff 推断):

1. `HELIOS_MAX_SEQUENCE_LENGTH` 为 226,而 diffusers `HeliosPipeline.__call__` 默认 `max_sequence_length=512`,UMT5 文本编码长度不足。
2. `enable_sequence_shard` 默认开启条件只匹配 "wan",Helios 不在其列;且 Helios DiT forward 未做 SP 分片/pad/all-gather,attention 未传入 `num_replicated_prefix`,跨注意力层用了 `is_cross_attention=True` 而非 `skip_sequence_parallel=True`。
3. chunked denoising 后将整段 latents 交给标准 DecodingStage 一次性解码;Wan VAE 的 causal 3D 卷积带 feature caching,整段解码时每 latent 帧产出的输出帧数与逐 chunk 解码不同,造成时间维错位与 chunk 边界可见接缝(新解码 stage 的 docstring 明确写了该机制)。
4. 噪声用 `torch.randn(..., device=device)` 直接生成,SP rank 间噪声不一致;`sample_block_noise` 的协方差/采样在 GPU 上做 cholesky,ROCm/CUDA 需要 MAGMA。

## 修复方案

- 配置:`HELIOS_MAX_SEQUENCE_LENGTH` 226→512;`enable_sequence_shard` 默认开启条件加入 "helios"。
- DiT(`runtime/models/dits/helios.py`):forward 中按 SP rank 分片 hidden_states 与 rotary_emb(不足 sp_size 时零 pad),以 `effective_context_length` 替代 `original_context_length` 参与 history/temb 计算,attention 调用改为 `self.attn(q, k, v, num_replicated_prefix=history_seq_len)`,输出前 `sequence_model_parallel_all_gather` 汇聚并去 pad;跨注意力层 `is_cross_attention=True` 改为 `skip_sequence_parallel=True`。
- 解码:新增 `HeliosDecodingStage`,逐 chunk 解码 latents 并在像素空间拼接(匹配 diffusers 行为),替换 `add_standard_decoding_stage()`;denoising 循环将每 chunk latents 存入 `batch.latent_chunks`。
- 噪声与性能:`torch.randn` 改用 `batch.generator`(SP rank 间噪声一致);`sample_block_noise` 的协方差矩阵与采样显式放 CPU(避免 ROCm/CUDA 上的 MAGMA cholesky);denoising 步骤包 `StageProfiler` 并用 `global_step_offset` 跨 chunk 计步;`set_forward_context` 传入 `batch` 而非 None。
- 调度器:`HeliosDMDScheduler = HeliosScheduler` 别名,`EntryClass` 同时注册两者(Helios-Distilled 的 scheduler_config.json 引用 HeliosDMDScheduler)。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:按 diff,全部改动位于 `python/sglang/multimodal_gen/` 下 Helios 相关配置、DiT、调度器、pipeline 与解码 stage;改变 Helios 的 SP 分片执行路径、文本编码长度、解码流程与噪声生成;非 SP 或单 chunk 场景分别回退到原路径。其他 diffusion 模型不受影响(仅 "helios" 加入条件分支)。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
