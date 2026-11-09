# PR #22763: [Feature] Full Cuda Graph Support for Cutlass MLA and 6% E2E Throughput Improvement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2025-08-15 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | performance, ready, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),实际改动为 1 个 backend 源文件 + 1 个测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/22763 |

## 一句话结论

让 SM100 Cutlass MLA backend 支持 full CUDA Graph(decode-only capture),E2E 吞吐提升约 6%。

## 动机

Cutlass MLA 是 SM100(Blackwell)上的 MLA attention backend,但此前不支持 full CUDA Graph,只能走 piecewise 路径,decode 阶段每次 step 都有 graph 外的 CPU 调度开销。在 DP + EP 部署 DeepSeek 类 MLA 模型时,这部分开销在小 input/长 output 的 decode 密集负载上占比可观,值得补齐 full cudagraph 支持。

## 优化点

机制上改动很小但关键:为 Cutlass MLA 新增 `CutlassMLAMetadataBuilder`,并声明 `attn_cudagraph_support = AttentionCGSupport.PURE_DECODE_ONLY`,即向 V1 引擎声明该 backend 只在 pure decode capture 下兼容 full CUDA Graph。backend 再通过 `get_builder_cls()` 暴露该 builder。这样 full cudagraph capture(decode-only)即可覆盖 Cutlass MLA 路径,消除逐 step 的 eager 启动开销。测试中同时设置了 `FORCE_NUM_KV_SPLITS=1` 并注明是绕过 hang 问题的临时措施,说明 kv-split 路径与 full graph 的组合还有已知问题。

## 关键改动

```text
vllm/v1/attention/backends/mla/cutlass_mla.py: 新增 CutlassMLAMetadataBuilder,声明 PURE_DECODE_ONLY 的 cudagraph 支持,并接入 get_builder_cls()
tests/compile/piecewise/test_full_cudagraph.py: 新增 cutlass_mla_llm_pair fixture 与 TestFullCUDAGraphCutlassMLA,对比 full cudagraph 与 piecewise 输出一致性
```

关键声明代码:

```python
class CutlassMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    # enable full CUDA Graph support for decode-only capture
    attn_cudagraph_support: ClassVar[
        AttentionCGSupport] = AttentionCGSupport.PURE_DECODE_ONLY
```

## 性能声明

测试模型 DeepSeek-V2-Lite,`--enable-expert-parallel --data-parallel-size 2`,full_cuda_graph 开启,cudagraph-sizes 16/32/64/128/256/512。负载:`vllm bench serve`,1000 请求、max-concurrency 256、random-input-len 32、random-output-len 512:

- full cuda graph:11.73 req/s,output token 吞吐 5934.46 tok/s,总吞吐 6297.38 tok/s,Mean TTFT 655.21 ms,Mean TPOT 40.85 ms
- 无 full cuda graph:11.04 req/s,5578.42 tok/s,5920.04 tok/s,Mean TTFT 846.60 ms,Mean TPOT 43.14 ms

即 output 吞吐约 +6.4%(5934.46/5578.42),TTFT 均值下降约 191 ms。精度:gsm8k 5-shot,flexible-extract exact_match 0.3753(开) vs 0.3776(关),差异在 stderr(约 0.0133)范围内。PR 未注明具体 GPU 型号(测试仅标注需 SM100 capability (10,0))。

## 验证与回归风险

新增 `TestFullCUDAGraphCutlassMLA`:用 full cudagraph 与 piecewise 两种配置对 DeepSeek-V2-Lite 生成并逐 token 比对输出文本,仅在 Blackwell(capability (10,0))上运行,并依赖 `FORCE_NUM_KV_SPLITS=1` 规避未修复的 hang 问题——后续修复 kv-split hang 后该约束需移除。风险面:Cutlass MLA 在 prefix caching / spec decode 等非 pure-decode 场景下被声明为不支持 full graph,若 metadata builder 判定有误可能触发 fallback 或错误结果。

## 关联

同区域:SM100 Cutlass MLA 解码支持(正文致谢的"previous work of enabling cutlass MLA on SM100",见索引中 SM100 CUTLASS MLA 相关卡,如 pr-20769);full cudagraph 基础设施见 pr-20059(Allow full cudagraph with separate compilation)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
