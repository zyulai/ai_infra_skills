# PR #15205: [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-12-18 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | performance, Multi-modal, ready-for-review, run-ci, vlm |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及 4 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/15205 |

## 一句话结论

Qwen3-VL 与 GLM-4.1V 视觉 RoPE 改用预计算 cos/sin 缓存,RoPE 路径 490us→186us,端到端 TTFT 约 +2%。

## 动机

Qwen3-VL 与 GLM-4.1V 的视觉编码器此前每次前向都重算 RoPE 频率并反复调用 `cos()`/`sin()`:各模型自持一套 `VisionRotaryEmbedding`(如 GLM-4.1V 的 `Glm4vVisionRotaryEmbedding`、Qwen3-VL 复用 transformers 的 `Qwen2_5_VisionRotaryEmbedding`),2D grid RoPE 与 text RoPE 的实现相互独立。对高分辨率/多图输入,这段重复计算直接加在 TTFT 上。思路受 vllm#28798、#28962 启发:预计算并缓存 1D cos/sin 表,2D grid RoPE 通过索引缓存复用。

## 优化点

在共享的 `RotaryEmbedding` 上新增 `get_cos_sin(seqlen)` 接口,直接从预计算的 `cos_sin_cache` 切片并 chunk 出 cos/sin,不再做 `outer` 重算与 `emb.cos()/emb.sin()`。GLM-4.1V 删除自定义 `Glm4vVisionRotaryEmbedding`,改用 `get_rope(head_size, rotary_dim=head_dim//2, max_position=8192, base=10000.0, is_neox_style=True)` 构建共享模块;`rot_pos_emb` 改为返回按 pos_ids 索引缓存后的 `cos_combined/sin_combined`(Qwen3-VL 同样接入,并把 max_grid_size 改为在 Python 侧计算以避免同步)。`vision.py` 的通用 attention forward 新增可选 `rotary_pos_emb_cos/sin` 参数(原 `position_embeddings` 路径保留),并在 cos 宽度为 head_size 一半时自动 cat 复制以适配 neox 旋转。顺带把 pos_ids/cu_seqlens 改为 `non_blocking` 搬运到 device。

## 关键改动

```text
python/sglang/srt/layers/rotary_embedding.py: 新增 get_cos_sin(seqlen),从 cos_sin_cache 切片
python/sglang/srt/models/qwen3_vl.py: 视觉 RoPE 改用共享 cos/sin 缓存并减少同步
python/sglang/srt/models/glm4v.py: 删除 Glm4vVisionRotaryEmbedding,改用 get_rope
python/sglang/srt/layers/attention/vision.py: attention forward 直接接收 cos/sin 张量
```

```python
def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_sin = self.cos_sin_cache[:seqlen]
    cos, sin = cos_sin.chunk(2, dim=-1)
    return cos, sin
```

## 性能声明

① RoPE 计算路径:改前 490us → 改后 186us(正文截图数据,未注明硬件与输入形状;服务启动命令含 `--mm-attention-backend fa3`、TP2)。② 端到端:正文标注 8xH20(server 命令显示 `--tp-size 1`),Qwen3-VL-8B-Instruct,image dataset,256 prompts,random-input-len 128(另有 83k vision tokens)、output 1、concurrency 32、560x560 jpeg、fa3 后端:Mean TTFT 2475.22→2429.13 ms(约 2% 提升),Input token throughput 5094.98→5139.84 tok/s。

## 验证与回归风险

accuracy:lmms_eval mmmu_val(Qwen3-VL-8B-Instruct,batch_size 16)0.5122 → 0.5156(单次运行,无 stderr)。diff 未新增测试,checklist 各项均未勾选。回归面:Qwen3-VL/GLM-4.1V 视觉路径与 `vision.py` 通用入口(新参数可选,旧路径保留);cos_sin_cache 按 `max_position=8192` 预计算,2D grid 超出该范围的行为素材未说明;Qwen3-VL 对 transformers `Qwen2_5_VisionRotaryEmbedding` 的依赖被移除。

## 关联

受 vllm-project/vllm#28798、#28962 启发(正文引用)。同批无其他 VLM RoPE 优化卡。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
