# PR #56273: [Quantization] Support packed NVFP4 Qwen4Exp PLE embeddings

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @stecasta |
| 合并 | 2026-10-14 |
| 分类 | Speculative Decoding \| tier: major |
| Label | ready, ci/build, qwen, quantization |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/56273 |

## 一句话结论

让 26.822 GiB packed NVFP4 PLE 表常驻 CUDA，1x DGX Spark 无需 CPU / 磁盘 offload 加载该 checkpoint。

## 动机

正文目标是让 `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` 在 1x DGX Spark 上运行且不做 CPU / 磁盘 PLE offloading：packed NVFP4 ngram 表占 26.822 GiB 并常驻 CUDA；resident path 在单 GB10 上以 4,096-token 上下文、2 GiB KV cache、禁用 MTP 验证，加载报 97.47 GiB。PR 标注 Fixes #56272。上游目前为 ModelOpt 混合精度 checkpoint 里这种显式存储格式选中未量化表。

## 优化点

新增 `ple_embedding_dtype="nvfp4"`：该 checkpoint 每个逻辑 160 值行存为 80 个 packed E2M1 bytes、10 个 E4M3 block scales 与 1 个 FP32 global scale。改动在既有 device / pinned-host 存储接口上新增 embedding method，两路都保持 packed 权重与 block scales 原存储、只解码被请求的行；lookup 复用 E2M1 decoder 产出 activation-dtype 行再做 ETP reduction。loader 拷贝各 shard 与本地 ETP 词表范围的重叠，并在所有权 weight stream 结束后校验行覆盖、global scale、tensor shapes 与 storage dtypes。

## 关键改动

```text
vllm/models/qwen4_exp/nvidia/ngram_embedding.py: 实现 NVFP4 packed PLE embedding（device / pinned-host 两路解码）
vllm/models/qwen4_exp/common/ngram_embedding.py: 存储接口与 ple_embedding_dtype 分发
vllm/models/qwen4_exp/amd/ple_layer.py: shard 流式加载与 ETP 覆盖校验
vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py: 复用 E2M1 解码
tests/models/qwen4_exp/test_ple.py: 新增 NVFP4 流式 shard、缺失行、错误 scale、编译 lookup 等测试
```

独立参考解码片段（摘自 test_ple.py 的 diff）：

```diff
+def _reference_nvfp4_ple(codes, scales):
+    nibbles = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(8, 160)
+    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
```

## 性能声明

正文未给出本 PR 自身加速的定量数据；"Single-Spark reference measurements" 明确声明 "collected independently of this PR and do not measure its performance impact"，仅作单 DGX Spark 可容纳性参考，不能视为本 PR 收益。条件：一个 GB10、TP1/PP1、40,960-token 上下文、prefix caching、chunked prefill、MTP 四个 draft tokens；workload 32,768 shared prefix + 2,048 suffix + 256 output tokens，streaming chat、temperature 0、并发 1/4/8。表格（local-inference-lab / b12x 对比 NVIDIA / vLLM with disk PLE）：并发 1 为 11.27 s / TTFT 3.40 s / 32.41 tok/s / acceptance 2.40 对 14.30 s / 3.45 s / 23.51 tok/s / 2.64；并发 8 为 45.99 s / 6.75 s / 43.44 tok/s / 2.20 对 44.34 s / 7.12 s / 45.92 tok/s / 2.59。正文注明每格单次运行、未评估精度等价性、观察到 host swap 故不作 swap-free 声明。资源：26.822 GiB packed PLE 在 Spark 常驻 CUDA、在 Max-Q 位于 pinned host。

## 验证与回归风险

新增 NVFP4 用例到 H200 Qwen4Exp CI lane；Test Result：PLE / configuration suites 在 Max-Q 与 DGX Spark 各 76 passed，编译启动 canary 2 passed，CPU suites 43 passed / 33 skipped（无 CUDA），全模型启动各捕获 6 个 CUDA graph，GSM8K smoke 每 GPU 31/32（96.875%，正文注明为 small smoke evaluation 而非 full benchmark）。回归风险：正文明确 "Real multi-GPU collectives and MTP have not been validated with this PR"；checkpoint 用预览架构名需 `hf_overrides` 归一化；Spark 禁用 PLE offload、Max-Q 启用；自动 MXFP8 backend 拒绝 96-output GDN projection，故选 `linear_backend=marlin`。

## 关联

正文明确线索：Fixes #56272；#54882、#55334 为 FP8 PLE 表；#53899 含较早 NVFP4 PLE 但混合精度分派未选中该显式表；#42791 增加通用量化词表 embedding；#54371 合并本改动依赖的存储接口。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
