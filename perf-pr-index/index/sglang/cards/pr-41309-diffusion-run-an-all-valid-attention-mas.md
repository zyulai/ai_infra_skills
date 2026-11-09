# PR #41309: [diffusion] Run an all-valid attention mask on the backend's unmasked kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-09-27 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(素材未提供统计);diff 实际涉及 3 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/41309 |

## 一句话结论

全有效的 attention mask 改走后端无 mask kernel,Ideogram4 在 B200 上单步 denoising 277→158 ms。

## 动机

sm100 上默认 attention 后端解析为 `DYNAMIC_CUDNN_SDPA`,但 `USPAttention` masked local 路径的 varlen fast path 只对 FA 后端开放;其他后端一律生成 dense additive mask 喂给 `torch.nn.functional.scaled_dot_product_attention`。任何 mask 都会把 SDPA 从 flash/cuDNN 挤到 PyTorch 的 `fmha_cutlassF_*_sm80` kernel——即使 mask 什么都没掩。Ideogram4 在 batch 1 时两个 guidance 分支都传这种 mask(text 不 padding,`segment_ids > 0` 全真),B200 上每次 attention 都走 cutlassF:denoising loop 中 attention 占 54% GPU 时间(1×B200)/47%(TP2);单次调用([1,18,4126,256] bf16)带 mask 2.19 ms、不带 0.81 ms。旁证:Ideogram4 在 B200 上仅比 H200 快约 9%,而其他模型快 1.7-2.6x——Hopper 默认 FA,varlen 路径本就把全 valid mask 当 dense。

## 优化点

`USPAttention.forward` masked local 分支入口增加判定:当 `attn_mask_meta` 表明每个 key 都有效(indices 数量 == mask 元素总数,host 侧即可知、无 device sync)、mask 是 2D 非浮点张量时,直接调后端自身的无 mask kernel(`self.attn_impl.forward`),跳过 dense SDPA mask。分段前缀输入(Qwen-Image)先拼接 q/k/v 再分发,前缀得以保留;带真实 padding 的 mask 与无 indices 的 padding-gap meta 仍走 masked 路径;FA 后端与其 varlen 分支不受影响,sequence-parallel 路径不变。另将 cuDNN→FlashAttention 回退日志改为 `warning_once`(否则每次 attention layer 打一条,启动打 68 次)。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/attention/layer.py: all-valid 判定 + 直通后端无 mask kernel
python/sglang/multimodal_gen/runtime/layers/attention/backends/sdpa.py: 回退日志 warning_once
python/sglang/multimodal_gen/test/unit/test_usp_all_valid_key_mask.py: 新增 4 个单测
```

```python
if (
    attn_mask_meta is not None
    and "indices" in attn_mask_meta
    and attn_mask.dim() == 2
    and not torch.is_floating_point(attn_mask)
    and attn_mask_meta["indices"].shape[0] == attn_mask.numel()
):
    return self.attn_impl.forward(q, k, v, ctx_attn_metadata)
```

## 性能声明

4×B200 验证,Ideogram4:

- Serving(TP2 默认命令,warmup 后 5 请求,main/PR 交错):main 3.424/3.427 s → 本 PR 2.192/2.205 s(约快 36%)。
- 稳态单步 denoising(`sglang generate`,无 warmup):1 GPU 277.0 → 158.2 ms;TP2 160.8 → 126.1 ms。
- kernel(torch profile,TP2):每步 attention 78.0 → 8.7 ms;cuDNN 9.20(torch 2.13 附带)在 sm100 无 head_dim-256 kernel,`DYNAMIC_CUDNN_SDPA` 走 FA4 fallback(`sm100_hd256_2cta_fmha_forward`,128 µs/次 对 cutlassF 1104 µs/次)。

## 验证与回归风险

新增 `test_usp_all_valid_key_mask.py`:all-valid mask 走后端 kernel 一次且结果匹配无 mask attention;真实 padding mask、无 indices 的 padding-gap meta 保持 masked 路径;分段前缀先拼接再分发。其中两个后端 kernel 测试在 main 上失败。周边 attention/mask 单测(ring tail-pad、K/V gather、varlen meta、replicated prefix、Ideogram4、BCG padding)通过。输出质量:PR vs main 的 PSNR 26.2 dB(TP2)/26.5 dB(1 GPU),而 main 自身 TP2 与 1 GPU 之间为 22.7 dB——all-valid additive mask 全零,数学不变,guidance 放大 FP 重排序属预期。Qwen-Image-2512 与 Z-Image-Turbo 输出 bitwise 一致(batch 1 不进该分支)。回归风险:判定依赖 `attn_mask_meta` 的 indices 语义,若上游 meta 构造变化可能误判;仅覆盖 2D bool/int mask。

## 关联

无(正文未提及其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
