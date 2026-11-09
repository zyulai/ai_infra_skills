# PR #36559: MoE: small-batch sorting path with fused mxfp8 quantisation

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zcnrex |
| 合并 | 2026-09-24 |
| 分类 | MoE(major) |
| Label | quant, amd, run-ci, jit-kernel, bypass-fail-fast |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 含新 Triton sorting 模块(557 行)及 mxfp8/aiter/triton runner/unquant 接线 |
| 链接 | https://github.com/sgl-project/sglang/pull/36559 |

## 一句话结论

decode 小批(P=M×topk≤64)时把 aiter MoE 排序 + stage1 mxfp8 激活量化合并为单次 Triton launch,省约 7+4us/层 launch 开销,低 batch 下 output 吞吐 +2.7-4.2%。

## 动机

aiter 的 opus MoE sorting kernel 按数千 token 规模设计,decode batch 下每层仅 launch/ramp 开销就约 7us;独立的 `fused_dynamic_mxfp8_quant_moe_sort` stage1 量化还要再加约 4us。当路由对数 P=M×topk 很小时,整个排序任务(stable sort-by-expert、每 expert 块 padding、expert-id 表、num_valid、moe_buf 清零)足以塞进单个小 Triton program。

## 优化点

核心是 P×P rank-compare 排序:每个 pair 对所有 pair 计数"同 expert 且序号在前"得到稳定 rank,"更小 expert 的 leader 块数"给出块前缀,dest = blocks_before×BLOCK_SIZE + rank,一次算出全部 scatter 目标;量化 CTA 不做跨 CTA 同步,而是重算这份(廉价的)pair 数学。输出布局与 aiter kernel bit-for-bit 一致(sorted_ids 编码 (slot<<24)|token,e8m0 RoundUp scale 按 aiter 的 mx_scale_shuffle_idx 地址 swizzle)。compact 变体(P≤64 且 ≤2×block_size)单 CTA 完成排序;distributed 变体(64 宽 tile,one writer per byte)覆盖 P≤256。安装方式是对 `aiter.fused_moe` 模块三个命名点(`fused_moe`、`_moe_sorting_impl`、`fused_dynamic_mxfp8_quant_moe_sort`)做 runtime patch,不适用时逐点回落原 aiter 函数;由 `AiterRunnerCore.__init__` 惰性、幂等地应用。配套:`sanitize_topk_ids` 参数让 native gfx950 MXFP8 MoE 在无 EP 时跳过防御性 topk_ids 拷贝(alias 语义经论证安全);`aiter_swiglu_oai_limit` 识别 aiter `ActivationType.Swiglu` 烧死的 SwiGLU-OAI 配置(alpha=1.702、up+1.0),使 bf16 aiter runner 用对 kernel 而非静默跑 plain SiLU;aiter CK bf16 无 SwiGLU-OAI codegen 时 `UnquantizedFusedMoEMethod` 报告原因并回落 Triton runner。

## 关键改动

```text
python/sglang/kernels/ops/moe/moe_sorting_small.py: 单 launch 排序+量化 Triton kernel(两个变体)与 aiter 模块 patch
python/sglang/kernels/ops/moe/mxfp8_moe_amd_gfx95.py: 新增 sanitize_topk_ids 参数(默认 True 保持现状)
python/sglang/srt/layers/moe/moe_runner/aiter.py: AiterRunnerCore 应用 patch;SwiGLU-OAI 识别;gate_mode 按 gfx95/布局选择
python/sglang/srt/layers/moe/moe_runner/triton.py: 传 sanitize_topk_ids=_topk_ids_may_be_nonlocal(config)
python/sglang/srt/layers/quantization/unquant.py: aiter CK 支持性检查给出明确原因,SwiGLU-OAI 回落 Triton
```

## 性能声明

PR 注明条件:8x MI350X(gfx950),MiniMax-M3-MXFP8,TP8,80k input / 600 output,cache hit rate 90%,每次 boot 两轮取 warm 轮(两轮差 <0.1%),噪声底 ±0.4%,baseline 为 `upstream/main`:

- output throughput:BS=1 55.35→57.68(+4.2%),BS=2 +3.2%,BS=4 204.62→212.93(+4.1%),BS=8 370.13→383.45,BS=16 616.28→636.27,BS=32 934.92→960.01(+2.7%);区间 +2.68%..+4.21%。
- input throughput:+0.31%..+0.95%。
- 重要说明(README 节):`--moe-runner-backend aiter` 在当时 main 上不被 mxfp8 采纳、回退 triton,因此 A/B 两侧实际都跑 Triton MoE runner,对比为 apples-to-apples;`AITER_CONFIG_FMOE` 在该配置下无效,aiter MoE 路径本身未被测到。收益集中在低 batch(fast path 门控 M×topk≤64),BS=32(+2.7%)时 fast path 未激活,证明 wrapper 判定在 fallback 路径零开销。

## 验证与回归风险

GSM8K 512 例(max_tokens 2048,temperature 0,seed 0,同一次 server boot):baseline 97.07% vs PR 96.88%(-0.19pp,1 题),stop_rate/truncated_rate 与 baseline 完全一致、error_rate 0,单题翻转是真实数值差异而非截断伪影。正确性论证:`sanitize_topk_ids` 仅在 rank 拥有全部 expert(`storage_ep_size==1`,即 `num_experts==num_local_experts`)时跳过,与 main 在 TritonRunnerCore 已用的 `filter_expert` 概念一致;skip 路径 alias 依赖 `moe_align_block_size` 对 topk_ids 只读。风险:aiter 模块 monkey-patch 式安装依赖 aiter 版本的函数签名;distributed 变体注释提到 static_range 内 runtime if 存在 mis-lowering 陷阱;CI Base/AMD 尚在运行、Extra 失败。

## 关联

同区域:aiter MoE runner、native gfx950 MXFP8 MoE(`fused_moe_mxfp8_native`)、SmallMMoe 路径。同分类其他 AMD MoE PR 见 #32577(aiter fused mHC)。前序/后续无其他收录,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
