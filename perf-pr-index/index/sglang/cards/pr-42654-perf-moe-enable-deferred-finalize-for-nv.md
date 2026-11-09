# PR #42654: [Perf] moe: Enable deferred finalize for NVFP4 on the routed TRT-LLM runner

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @vincentzed |
| 合并 | 2026-11-02 |
| 分类 | MoE |
| Label | amd, deepseek, run-ci |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/42654 |

## 一句话结论

NVFP4 routed TRT-LLM MoE 启用 deferred finalize:DeepSeek-V4.1-Flash 4×B300 上 ITL p50 约 -3% 到 -9.9%,finalize+shared-expert+all-reduce 融合为单 kernel。

## 动机

Deferred finalize 指 MoE kernel 跳过最后一步(对选中 experts 的加权和),由后续 fused kernel 把该步与 shared-expert add 一起做。NVFP4 experts 在 `flashinfer_trtllm_routed` runner 上从未能用:混合 FP8+NVFP4 的 DeepSeek V4/V4.1 checkpoint 固定走该 runner(kernel 内建 `RoutingMethodType` 无法描述其路由)。三个阻塞:`supports_deferred_finalize` 要求 `is_flashinfer_trtllm()`(routed 为 false);`DeepseekV2MoE` 只在 top-k 输出为 BYPASSED 时 defer;routed 调用恒传 `do_finalize=True`(#44666 已在 main 上解除该条,本 PR 复用)。

## 优化点

`FusedMoE` 新增 `supports_routed_deferred_finalize` 标志:`ModelOptNvFp4FusedMoEMethod` 在 `flashinfer_trtllm_routed` 上、无 A2A backend、routed scaling factor 已折入 top-k weights 时为 true;`supports_deferred_finalize` 不变(CuTe DSL layer-boundary fusion、Kimi、Qwen 路径维持原状)。`DeepseekV2MoE.forward_normal_dual_stream` 按 top-k 输出格式选检查的标志:BYPASSED 查旧标志、常规 top-k 查新标志。配合 #42245 后 V4 block 只要 fused mHC kernel 可用即请求 deferred 输出,NVFP4 现在由单个 fused kernel 完成 finalize + shared-expert add + all-reduce。工程 workaround:deferred routed 调用仍传一个 `[T, hidden]` 占位输出 buffer(kernel 在 deferred 模式不写它)——FlashInfer autotune cache key 含输出 shape,`output=None` 会 miss 调优条目落到 fallback tactic,FlashInfer 0.7.0.post1 上每次调用慢 9-14%(已报 flashinfer#6077,0.7.2 上 fallback 差距缩到 2% 内,buffer 待上游修复后移除)。`SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=0` 可关闭;#39199 的 padded hidden size NVFP4 权重规则保留(defer 仅在 hidden size 未 padded 时启用)。

## 关键改动

```text
python/sglang/srt/layers/moe/fused_moe_triton/layer.py: 新增 supports_routed_deferred_finalize 标志
python/sglang/srt/models/deepseek_v2.py: 按 top-k 输出格式选择 deferred 标志
python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py: routed deferred 调用保留占位 output buffer 的 workaround
test/registered/unit/models/test_deepseek_v4_amd_tp4.py: MoE stub 补新属性
```

## 性能声明

- 硬件/负载:4×B300,FlashInfer 0.7.0.post1(复测用 0.7.2),nvidia/DeepSeek-V4.1-Flash-NVFP4,TP4+EP4,合并 main `abed685de1`(含 #42245)后;DSPARK(block size 5)+ aiperf ISL 8192/OSL 1024。ITL p50:c1 1.005→1.004 ms(-0.1%)、c4 1.529→1.455 ms(-4.8%)、c8 2.042→1.945 ms(-4.7%);同构建两次启动差异可达 2.2%/cell(测量噪声,非加速数字)。
- 多轮 rerun(不同 main 合并点,ITL p50 delta):Plain server -5.8% 到 -7.1%(如 out tok/s 1432.4→1521.7 @c8、4666.9→4982.5 @c64);DSpark 行混合(-5.1% 到 +1.4%),c64 时每 decode step 320 行超过 `SGLANG_MOE_DEFERRED_FINALIZE_MAX_TOKENS`(192)故 defer 关闭;另一合并点 c4 达 -9.9%。
- 归因:deferring 本身只省 finalize 步(约 GPU 时间 1.5%,#39299),#42245 之前同改动在噪声内(-0.3% 到 -1.1%);收益主要来自 fused kernel 顺带做 all-reduce。
- 异常记录:某次 rerun 中 DSpark c8 行 TTFT p99 1462 ms vs 301 ms、wall 5.7s vs 4.8s,作者未找到原因;另一检查中 GSM8K wall time 变长,同样未解释。

## 验证与回归风险

Kernel 级:28 个用例(V4.1-Flash expert 形状 hidden 5120 / intermediate 2304 / 384 experts / top-6,T∈{1..384},全本地与 EP 96、per-tensor/per-token scale):最坏 max abs diff / max abs ref = 1.06e-3,cosine 相似度 0.99999964。E2E:V4.1-Flash-NVFP4 TP4+EP4、12 prompts × 48 greedy tokens:PR vs main 从第 246/504 token 起出现差异(2/12 prompts 全同)——`moe_finalize_fuse_shared` 以 FP32 相加、一次舍入到 BF16,旧路径 finalize 与 add 各舍一次,属预期数值路径变化;同构建多次启动输出一致。GSM8K 500 题 greedy:多轮 0.976-0.984 vs off 臂 0.974-0.984,均在 1 个标准误(0.006)内。单元测试:分支与未改 main 的 pass/fail 集完全一致(如 1527 passed / 342 failed / 40 skipped,同 76 个失败 id)。风险面:deferred 数值路径改变所有命中该 runner 的 NVFP4 MoE 输出舍入行为;依赖 FlashInfer autotune workaround(上游 #6077/#6549 未修);`supports_routed_deferred_finalize` 与 #46140 的 `packed_nvfp4_routed_deferred` 并存。PR CI:Base/Extra 失败(x),AMD 通过。

## 关联

前序:#39299(.deferred finalize 框架,本 PR 为其 NVFP4 case)、#44666(routed 调用支持 defer)、#42245(fused mHC 请求 deferred 输出)、#38798(MXFP4 已用 do_finalize=False)。后续:MXFP8 routed defer(#46387)、移除占位 buffer(待 flashinfer#6077/#6549)。同分类:PR #7278(CUTLASS FP8 Blockscale MoE)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
