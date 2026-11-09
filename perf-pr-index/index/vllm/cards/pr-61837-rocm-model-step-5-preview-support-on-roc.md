# PR #61837: [ROCm][Model] Step-5-Preview support on ROCm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @akii96 |
| 合并 | 2026-10-31 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, new-model, rocm, ready, ci/build, multi-modality, tool-calling, kv-connector, nvidia, mooncake, nixl |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含约 36 个文件(29 个超长文件被省略),amd/ 下大量新文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/61837 |

## 一句话结论

stepfun-ai/Step-5-Preview 上 ROCm(MI300X/MI325X/MI355X):CSA 稀疏注意力 + 3-token MTP,GSM8K 验证下 MTP 带来约 2.3x 提速。

## 动机

Step-5-Preview 的 NVIDIA 支持已在 #61727 落地(SM90/SM103),但 ROCm 用户无法运行该模型。该 PR 为其补齐 ROCm 实现,主支持与验证目标是 MI300X/MI325X (gfx942),MI355X (gfx950) 次级验证(由 @jamesETsmith 完成)。模型核心是 compressed sparse attention (CSA):23 个稀疏层在 prefill、chunked prefill、decode 与 MTP verification 上都要走 region scoring + top-k 选择 + split-K 稀疏注意力的专用路径。

## 优化点

代码重组为 `vllm/models/step5/{common,nvidia,amd}`:平台无关的 Step-5 稀疏注意力代码(sparse_indexer、summary_update、sparse_decode_attention 等)从 nvidia/ 上移到 common/,ROCm 实现自包含于 amd/(不 import nvidia/),`step5/__init__.py` 用 `__getattr__` 按平台分发模型类。ROCm 路径要点:主 paged K/V 由 `unified_kv_cache_update` 写入、直接消费 AITER unified paged 布局;稀疏状态为 FP8 summary cache + 投机 tail ring,挂在既有 KV-cache 生命周期上,无需 scheduler 改动或 GPU-CPU 同步;decode 与四行 MTP verification 共享 summary scoring、per-row top-k 与 split-K paged attention(含 head-wise attention gate);CSA 封装为 graph-splitting custom op `vllm::step5_csa_attention`(加入 CompilationConfig fused ops 列表,支持 piecewise CUDA graphs);scored-prefill 路径(histories 超过 topk x region_size)在 ROCm 分发到 `amd/warmup.py`,启动 warmup 覆盖该路径。稀疏层支持 BF16 K/V 与 FP8 E4M3 K/V(需 `--block-size 32`);P/D 侧 NIXL 与 Mooncake 保持压缩 summary page 完整,layer-outer 传输会拆页时启动期直接报错(`check_compressed_kernel_split`,适用于所有压缩缓存模型);indexer Q/W 按 provider group 用 ColumnParallelLinear 分片。限制:仅 piecewise CUDA graphs;`--moe-backend triton` 必需(AITER block-FP8 MoE 尚未实现每层 SwiGLU clamp at 7)。

## 关键改动

```text
vllm/models/step5/__init__.py: 按平台分发 Step5ForCausalLM/Step5ForConditionalGeneration/Step5MTP
vllm/models/step5/amd/: 新 ROCm 实现(model、csa_attention、csa_cache、ops/csa_paged、checks、warmup)
vllm/models/step5/common/: 平台无关稀疏算子自 nvidia/ 上移
vllm/model_executor/warmup/step5_dsa_warmup.py: ROCm 分发 + scored-prefill warmup 返回值重构
vllm/distributed/kv_transfer/kv_connector/utils.py: check_compressed_kernel_split(NIXL/Mooncake 接入)
```

## 性能声明

MTP:3 draft tokens 平均接受长度 3.1-3.3(较早版本),GSM8K 早期运行中 MTP 带来约 2.3x speedup。gfx942(8x MI300X,TP8,max-model-len 32768,ROCm nightly 3f1bcfd82 + 本分支):GSM8K 全量 1319 5-shot 97.2%(MTP 97.4%);最终版本 200 题抽样 MTP 96.0%;NIAH 4K/16K/30K 27/27;图像输入 3/3;prefix caching 共享约 22K 前缀命中 66%;26K-token prefill TTFT 2.7 s;`VLLM_GPU_SYNC_CHECK=error` 下长 chunked prefill + 15 并发 decode 无同步。gfx950(8x MI355X):GSM8K 96.97/97.27 strict/flexible(MTP 96.59/97.04),GPQA-Diamond 85.86%(MTP 85.35%),RULER 单/多 key NIAH 100%@32K/128K/256K。重组前后 TPOT(8x MI300X,3 次取中位):BF16 1K-in-1-req 38.97→39.42 ms、8K 86.67→87.01、1K-16 并发 45.10→44.80;scored-prefill warmup 把首个冷 9K-token prompt 从 3.46 s 降到 1.14 s。PR 未注明纯 ROCm vs NVIDIA 的端到端吞吐对比。

## 验证与回归风险

新增测试:test_amd_csa_metadata.py(metadata builder 位置回退、decode/MTP 不误判 prefill)、test_amd_csa_paged.py(乱序 block table 上 paged kernel vs PyTorch 参考,FP8 K/V 各变体)、test_amd_csa_warmup.py、test_amd_fp8_kv_init.py(FP8 KV block-32 初始化)、test_warmup_dispatch.py;test_strided_summary_cache.py 改为 is_cuda_alike 以覆盖 ROCm;Buildkite 新增 MI355 DPX Step5 步骤;`tests/models/test_initialization.py` 增加 ROCm MI3xx 门控。风险面:nvidia/ 侧改动仅 import 路径,但 common/ 上移影响两个平台;`check_compressed_kernel_split` 对所有压缩缓存模型的 P/D 路径生效(layer-outer 布局 + 不兼容 block size 会启动失败);warmup 重构改变 NVIDIA scored-prefill warmup 的调用顺序(先跑混合步再做 prefill scoring warmup)。

## 关联

直接前序:#61727(Step-5-Preview NVIDIA 支持,本 PR 为其 ROCm 对应物,共享 cache 契约与 Triton scoring/split-K decode)。同区域:Step-3.5 MTP proposer、AOITER unified attention 后端(rocm_aiter_unified_attn.py)、Mooncake/NIXL KV connector。后续限制项:CSA metadata 支持全量 graphs、AITER block-FP8 MoE 增加 SwiGLU clamp、融合 region selection。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
