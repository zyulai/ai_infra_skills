# PR #53175: [Kernel] Resubmit PR 48666 - Gemma4 FP8 KV FA4 head dim 512 backend selection

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jhaotingc |
| 合并 | 2026-09-24 |
| 分类 | Quantization(tier: major) |
| Label | rocm, speculative-decoding, ready, ci/build, deepseek, cpu, nvidia, mrv2, DSv4, minimax, DSv4.1 |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 中 5 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/53175 |

## 一句话结论

修正 Gemma4 FP8 KV 在 SM90 的注意力后端选择:head dim 512 causal 用 FA4、head dim 256 用 FA3、mm-prefix 用 Triton,输出吞吐提升 1.45x。

## 动机

前序 #48666 曾把 Gemma4 FP8 KV 全层强制升级到 FA4,但 FA4 FP8-KV 内核不支持 bidirectional attention(mm-prefix),模型会输出 special token,CI 报 perf 回归后被 #52987 revert。#56305 之后 `TRITON_FLASH_ATTN` composite 可让 causal 文本按层用 FA3/FA4、mm-prefix 请求用 Triton,`--language-model-only` 不再必需;#42175 统一 FA4 后还存在静默升级问题。

## 优化点

三处机制:`get_supported_kernel_block_sizes` 全线接受 `kv_cache_spec` 参数(约 30 个 attention backend 签名补齐);`FlashAttentionBackend` 新增 `_get_sm90_fa4_fp8_kv_block_size`,SM90 + FP8 KV + head dim 512 时声明 64-token TMA tile 页(SM90 FP8-KV-dequant 内核的页大小);`models/config.py` 的 Gemma4 异构 head dim 配置在 SM90+FP8 KV 时改为 per-layer FA3/FA4 选择而非强制 FA4;`supports_combination` 显式拒绝 SM90 FP8 KV + mm_prefix 组合(需走 Triton 复合后端);FA4 SM90 FP8-KV dequant 路径消费原生 FP16/BF16 Q,禁用 quant query input。

## 关键改动

```text
vllm/v1/attention/backends/flash_attn.py: SM90 FA4 FP8-KV 64-token 页声明与 supports_combination 规则
vllm/model_executor/models/config.py: Gemma4 异构 head dim 的 per-layer FA3/FA4 分支(其余配置仍强制 FA4)
vllm/v1/attention/backends/fa_utils.py: SM90 FP8 KV 允许 FA3 或 head_size=512 的 FA4
vllm/model_executor/layers/attention/attention.py: _largest_kernel_block_within 透传 kv_cache_spec
```

## 性能声明

H200,RedHatAI/gemma-4-31B-it-FP8-dynamic,`--kv-cache-dtype fp8`,TP1,vllm bench serve random 8192/1024、512 prompts、并发 128(CI 跑 3 次,报告各取一次):TRITON_ATTN(before)→ TRITON_FLASH_ATTN(after):输出吞吐 546.84 → 791.60 tok/s(1.45x);总吞吐 4921.59 → 7124.39 tok/s;Mean TPOT 195.10 → 129.67 ms;Median TTFT 16375.95 → 14289.54 ms;P99 ITL 1172.52 → 672.11 ms。SM90 上 head-dim 256 R-SWA 层即使无 mm-prefix 也用 FA3:强制 FA4+FP8 KV 会输出损坏。

## 验证与回归风险

tests/kernels/attention/test_attention_selector.py 复合后端期望更新(Hopper 因果路由 FA3 或 FA4 即选 composite);test_backend_per_kind 适配新签名。风险:大量 backend 签名变更为机械性但波及全部 attention backend;SM90 FA4 hd512 的 64-token 页改变 KV 页布局/容量计算;mm-prefix + FP8 KV 组合现被显式拒绝,必须用 Triton composite。

## 关联

前序:#48666(被 #52987 revert)、#56305、#42175;同分类 Quantization:#47972。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
