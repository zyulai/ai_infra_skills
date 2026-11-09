# PR #27063: [AMD] Optimize gpt-oss-120B performance

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kkHuang-amd |
| 合并 | 2026-06-08 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | amd, run-ci |
| Diff | diffstat 缺失(素材未提供 +N/-N);pack diff 片段含 4 个文件,另有 8 个文件因体积未收录 |
| 链接 | https://github.com/sgl-project/sglang/pull/27063 |

## 一句话结论

AITER 后端接入 Atom pa_decode_gluon 与 SHUFFLE 5D KV layout,gpt-oss-120b decode 吞吐 +15%。

## 动机

把 Atom 的 `pa_decode_gluon` paged-attention decode kernel 集成进 SGLang 的 AITER 后端,加速 gpt-oss-120b 在 ROCm(MI300 / MI355)上的 decode。该 kernel 要求 SHUFFLE 5D 向量化 KV cache 布局(aiter 的 `mha_batch_prefill_func` 也原生消费该布局),因此新增 opt-in 池格式 `SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`,让 AITER 后端在 prefill 和 decode 都走该布局;默认 `"nhd"` 时所有代码路径与 main 完全一致(bit-for-bit)。

## 优化点

`mem_cache` 中 `MHATokenToKVPool` 按需分配 5D 布局(K: `(num_blocks, H_kv, head_dim/X, page_size, X)`,V: `(num_blocks, H_kv, page_size/X, head_dim, X)`,X = 16/itemsize),`SWAKVPool` 把布局传播到 full-attn 与 SWA 子池;`utils.py` 新增 Triton writer `reshape_and_cache_shuffle_5d` 与位精确逆变换 `gather_shuffle_5d_to_linear`;新文件 `aiter_utils.py` 提供 `forward_extend_vectorized_5d`(fresh-prompt shortcut 免池读,否则 gather-and-linearize 后走 LINEAR prefill,fp8 K/V 原生支持)与 `forward_decode_vectorized_5d`(full-attn 与 SWA 层都走 `pa_decode_gluon`,fp8 KV 传 key/value scale);重新启用 HIP `fused_qk_rope_reshape_and_cache` 融合 RoPE + KV 写;`server_args` 在 5D 布局下默认 `page_size=64`。

## 关键改动

```text
python/sglang/srt/layers/attention/aiter_utils.py: 5D 专属 prefill/decode 路径(新文件)
python/sglang/srt/layers/attention/utils.py: SHUFFLE 5D Triton writer/gather kernel
python/sglang/srt/layers/attention/aiter_backend.py: 5D 池检测与分派
python/sglang/srt/environ.py: SGLANG_AITER_KV_CACHE_LAYOUT 等环境变量
```

## 性能声明

gpt-oss-120b @ MI300、TP=1、page=64、8K/1K decode(256 prompts):SHUFFLE 5D vs NHD baseline,c=4 TPOT ~4.75 → ~4.17ms(-12%)、吞吐 780 → 920 tok/s(+18%);c=8 ~6.06 → ~4.88ms(-19%)、1220 → 1530(+26%);c=128 ~7.05 → ~6.06ms(-14%)、2007 → 2307(+15%)。正文总结 "+14% TPOT and +15% output throughput";`pa_decode_gluon` 本身比 `unified_attention` 快约 46%。GSM8K(3-shot, chat template)在种子噪声内:bf16 0.8855 → 0.8870,fp8 0.8893 → 0.8848;fp8+8K c=64 与 ATOM 吞吐差 ±6% 内,低并发反超 18-26%。

## 验证与回归风险

GSM8K 覆盖 bf16/fp8 KV、chunked-prefill gather 路径与禁用 fused write 的独立 writer 路径。TP>1 的 GSM8K 失败在 pre-Plan-A main 同样存在(疑似上游 MoE FlyDSL kernel 分派问题,非本 PR 引入);`--max-running-requests < 256` 会触发上游 AITER MoE bf16-activation 路径产出错误 logits(独立问题,可用 `GPTOSS_SWIGLU_MXFP4_BF16_BOUND=128` 规避);默认 nhd 布局完全不受影响(全 opt-in)。

## 关联

无(PR 正文未列出前序/后续 PR;依赖外部 aiter 的 `pa_decode_gluon` kernel)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
