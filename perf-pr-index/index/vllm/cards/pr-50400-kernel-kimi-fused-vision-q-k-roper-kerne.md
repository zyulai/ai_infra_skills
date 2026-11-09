# PR #50400: [Kernel][Kimi] fused vision q/k roper kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lengrongfu |
| 合并 | 2026-08-20 |
| 分类 | Attention / Kernel \| tier: major |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, qwen, deepseek, kv-connector, nvidia, quantization, rust, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/50400 |

## 一句话结论

用 Triton 融合 vision Q/K RoPE，kernel 较 PyTorch 实现快约 5x，端到端 TTFT 改善 1%-19%。

## 动机

PR 正文 Purpose 仅写 "By using Triton to fuse Q/K RoPE operations, kernel efficiency is improved by 5x compared to the existing PyTorch implementation."，未展开此前瓶颈。从 diff 看，kimi_k25_vit.py 里原有手写的 apply_rope 依赖 torch.view_as_complex / view_as_real 与 float() 转换，本 PR 将其替换为统一的 ApplyRotaryEmb。

## 优化点

正文称用 Triton 融合 Q/K RoPE 操作。diff 层面：删除 kimi_k25_vit.py 内手写的 apply_rope（complex view 实现），在 Kimi vision attention 构造 ApplyRotaryEmb（enforce_enable=True、is_neox_style=False、enable_fp32_compute=True）；attention_qkvpacked 里把 rope_freqs_cis 拆成 real/imag 的 contiguous cos/sin，先做 _apply_rope_input_validation，再分别对 xq、xk 调用 apply_rotary_emb；并把 rope_freqs_cis 参数从 Optional 改为必需（attention_qkvpacked 与 forward 签名）。

## 关键改动

```text
vllm/model_executor/models/kimi_k25_vit.py: 删除手写 apply_rope，改用 ApplyRotaryEmb 对 Q/K 施加 RoPE（pack 的 diff 仅含此 1 个源文件）
```

```python
        _apply_rope_input_validation(xq, rope_freqs_cis)
        _apply_rope_input_validation(xk, rope_freqs_cis)
        rope_cos = rope_freqs_cis.real.contiguous()
        rope_sin = rope_freqs_cis.imag.contiguous()
        xq = self.apply_rotary_emb(xq, rope_cos, rope_sin)
        xk = self.apply_rotary_emb(xk, rope_cos, rope_sin)
```

## 性能声明

正文称 "kernel efficiency is improved by 5x compared to the existing PyTorch implementation"。微基准（benchmark_kimi_vision_rope.py --dtype bfloat16 --cudagraph，head_dim 128，单位 us）：256 tokens/3 heads 13.409 → 1.570（8.542x）、1024/6 18.178 → 2.884（6.303x）、4096/12 112.455 → 17.820（6.311x）、8192/12 212.181 → 36.951（5.742x），表中 speedup 介于 5.742x 与 8.542x 之间。端到端：vllm serve /mnt/model/Kimi/Kimi-K2.6-NVFP4/（--kv-cache-dtype fp8 --tensor-parallel-size 8 等），vllm bench serve --dataset-name random-mm，覆盖 224×224 / 448×448 / 720×720 / 1080×1080 与图片数 1-8，正文称 "TTFT improvements ranging from 1% to 19%"，且图片数越多改善越明显。测试条件：微基准 bfloat16 + cudagraph；serving 模型 Kimi-K2.6-NVFP4、精度 NVFP4 + fp8 KV、TP8、random-mm 多图 workload；硬件 PR 正文未注明（正文提到仅单 GPU 时大数量 720 分辨率图片会 OOM）。

## 验证与回归风险

单元测试 `python -m pytest tests/model_executor/layers/test_vision_rope.py`：8 passed、16 warnings in 8.41s（Python 3.12.13、pytest-9.1.1）；正文另给出端到端 random-mm 的 TTFT 对比（1%-19%）。正文 Test Plan 还给出 benchmark_kimi_vision_rope.py（--dtype bfloat16 --cudagraph）的微基准命令与 12 组 tokens/heads 组合，Test Result 注明 Unit test PASS，并给出 random-mm 多分辨率测试矩阵（224×224 至 1080×1080、图片数 1-8）与原始结果压缩包链接。回归风险：正文注明因只有单 GPU，720 分辨率、图片数较多时出现 OOM、服务崩溃并产生失败请求（表中 "×"）；rope_freqs_cis 由 Optional 改为必需，影响 Kimi vision 模型的调用方；改动集中在 Kimi vision 的 Q/K RoPE 与多模态路径。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
