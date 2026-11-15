# PR #65453: [Bugfix][Multimodal] Repeat-last default padding for `cu_seqlens` in multimodal CG

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#65453](https://github.com/vllm-project/vllm/pull/65453) |
| 作者 | cjackal |
| 合并日期 | 2026-11-14 |
| labels | bug, documentation, ready, multi-modality, qwen, deepseek, nvidia, mistral, kimi, glm, minimax, cohere |
| 热度 | 👍 0 · 💬 26 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
encoder CUDA graph replay 序列数少于 capture 时 cu_seqlens 尾部被零填成负长度序列,FlashAttn 等 ViT 后端输出错误;默认改为 repeat-last。

## 问题现象
#64984 的 follow-up:encoder CUDA graph replay 的序列数少于 capture 时,manager 的默认拷贝把 `cu_seqlens` 尾部零填充;真实总偏移之后出现 0,使 padding 序列变成负长度,FlashAttn 与其他非 FlashInfer ViT 后端对真实条目产生错误 encoder 输出。受新默认修复的模型:GLM-4.1V/4.5V 及继承其 config 的模型(GLM-OCR、GLM-5.3-Flash),以及设置 `encoder_cudagraph_max_vision_items_per_batch` 时的 Qwen2-VL——两种场景都是 video-enabled capture 把 `cu_seqlens` pad 到 `max_frames_per_batch`,而 image replay 只 pad 到 `max_batch_size`。FlashInfer packed 两段布局仍需专门 padding 的:Qwen2.5-VL、Qwen3-VL(此前自行注册)、GLM-4.1V(video enabled 时 image replay 用了更小的 batch bucket)、Kimi-K2.5(capture pad 到 `max_batch_size + 1` 而 replay 不 pad,`max_batch_size >= 8` 时 bucket 不同)。GPU 验证(@Pb314314,GLM-4.1V-9B-Thinking,GB200,bf16):graph 输出对 eager 的相对 L2 误差,336x336 图 main 106.8% → 本 PR 2.70%,448x448 169.1% → 2.36%,672x672 169.4% → 0.18%(224x224 两者均 0.20%)。`pytest tests/v1/cudagraph/test_encoder_cudagraph.py -v`:101 passed, 0 failed;旧的零填充默认下 FlashAttn 用例失败。

## 根因
正文披露:replay 序列数少于 capture 时,manager 的默认拷贝对 `cu_seqlens` 尾部零填充,真实累计偏移之后的 0 使 padding 序列长度为负,FlashAttn 等后端据此对真实条目算出错误输出。

## 修复方案
(1) `vllm/v1/worker/encoder_cudagraph.py`:新增 `_copy_cumulative_buffer`(尾部重复最后一个值)与 `_get_padding_logic`;`_CUMULATIVE_BUFFER_KEYS = frozenset({"cu_seqlens", "cu_window_seqlens"})` 命中时默认 repeat-last,其余 buffer 仍零填充;(2) FlashInfer 的 `_pad_flashinfer_cu_seqlens_buffer` 从 qwen2_5_vl.py 移入 `MMEncoderAttention.get_cu_seqlens_padding_logics`(非 FLASHINFER 返回 `{}`),与 `maybe_recompute_cu_seqlens` 相同的 OOT dispatch,OOT attention 可覆写 padding;(3) 各模型统一接线:glm4_1v、qwen2_5_vl、qwen3_vl、kimi_k25、pixtral、cohere_compass、minimax_m3 经 `get_cu_seqlens_padding_logics` 提供 padding;删除与新默认等价的 per-model 注册(Qwen2.5-VL/Qwen3-VL/LFM2-VL 的 `_pad_cumulative_seqlens_buffer`,Kimi-K2.5/DeepSeek-V4.1/MiniMax-M3 的 `pad_cu_seqlens`)。Pixtral 与 Cohere Compass(纯图像)capture 与 replay 的 `cu_seqlens` 等长,FlashInfer helper 退化为普通拷贝。

## 影响与回归风险
- 影响窗口:自 #64984 引入 encoder CUDA graph manager 默认零填充拷贝起(正文称本 PR 为其 follow-up)。
- 回归面:所有 encoder-CG 模型的 `cu_seqlens`/`cu_window_seqlens` padding 默认值变化;被删注册与新默认等价(正文论证 Kimi-K2.5/DeepSeek-V4.1 的 `min(src, dst)` 截断分支永不运行:每图一条序列,replay 至多 `max_batch_size + 1` 个 offset 而 capture 为 `max_batch_size + 2`);未接线的 DeepSeek-V4.1/Kimi-K3/Qwen2-VL 自建 `cu_seqlens`,依赖 manager 默认;vllm-ascend OOT 覆写 `_run_budget_graph` 沿用旧 lookup,新默认不适用(其 FIA 路径丢弃非递增 endpoint,零填充与 repeat-last 结果相同);MiniMax-M3 拒绝 FlashInfer,得到 `{}` 与默认,与之前一致。

## 关联
- [#64984](https://github.com/vllm-project/vllm/pull/64984)(follow-up 目标)
- [#66067](https://github.com/vllm-project/vllm/pull/66067)(@Pb314314 的纯重构,不碰 cu_seqlens padding)
- GPU 验证 comment:https://github.com/vllm-project/vllm/pull/65453#issuecomment-6645668754

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
