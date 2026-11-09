# PR #58069: [Dependency] Upgrade FlashInfer version to 0.7.0

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wzhao18 |
| 合并 | 2026-09-24 |
| 分类 | Attention / Kernel (tier: major) |
| Label | ready, ci/build, deepseek, nvidia, ready-run-all-tests, DSv4, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/58069 |

## 一句话结论

把 FlashInfer 依赖从 0.6.18.post1 升级到 0.7.0,并适配 SM90 稀疏 MLA 的 CSR 索引布局与 top-k 采样器的小 SM 门控(依赖升级 + 接口适配)。

## 动机

vLLM 全量依赖 FlashInfer 提供注意力/采样/分布式内核,升级到 0.7.0 才能获得该版本的新内核与修复;但 0.7.0 改变了若干接口契约(稀疏 MLA 索引布局、GDN 后端选择参数、top-k masking 的 SM 数量约束),需要同步修改 vLLM 侧调用代码。PR 正文极简,仅一句 "Upgrade FlashInfer version to 0.7.0",动机细节需从 diff 中的注释推断。

## 优化点

版本钉从 `flashinfer-python==0.6.18.post1` 改为 `0.7.0`(cuda.txt、rubin-prerelease.txt、Dockerfile、versions.json 同步)。代码适配三块:(1) SM90 FlashInfer 稀疏 MLA 后端——0.7 要求 packed CSR 偏移与每行精确长度(含零长 padding 行)一致,`plan()` 从"每行填满 topk_width"改为对 `_lens_cpu` 做 `cumsum` 生成 CSR `kv_indptr`,并新增 Triton 内核 `_pack_topk_indices` 把有效前缀按偏移打包进 `kv_indices`(替代原来整块 reshape+clamp 的拷贝),配合 CUDA graph replay 使用;(2) DeepSeek-V4/V4.1 稀疏 MLA launcher 声明 `sparse_indices_are_storage_offsets=True`,使 packed pages(#44577)下 vLLM 侧的重映射逻辑可保留;(3) FlashInfer 采样器新增 >16 SM 门槛——0.7+ 拒绝在 ≤16 SM 上做 multi-CTA top-k masking(跨 CTA 软件屏障无法保证前进性),vLLM 侧在支持性检查中提前拒绝并回退。另在 Qwen GDN 的 `chunk_gated_delta_rule` 调用上显式传 `backend="flashinfer"`。

## 关键改动

```text
requirements/cuda.txt + requirements/rubin-prerelease.txt + docker/Dockerfile + docker/versions.json: 版本钉 0.6.18.post1 -> 0.7.0
vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py: 新增 _pack_topk_indices Triton 内核与 pack_indices;plan() 改为 CSR 累积偏移
vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py + vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py: launcher 增加 sparse_indices_are_storage_offsets=True
vllm/v1/sample/ops/topk_topp_sampler.py: flashinfer_sampler_supported 增加 >16 SM 检查
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py: 显式传 backend="flashinfer"
```

关键摘录(plan 的 CSR 化):

```python
self._lens_cpu.zero_()
self._lens_cpu[:num_tokens] = kv_lens.to(torch.int32)
self._kv_cpu[0] = 0
torch.cumsum(self._lens_cpu, dim=0, out=self._kv_cpu[1:])
```

## 性能声明

无性能数字声明;PR 正文未附 benchmark,未注明测试条件。

## 验证与回归风险

PR 正文的 Test Plan / Test Result 为空。diff 中带测试:`tests/kernels/attention/test_flashmla_sparse.py` 新增 deepseek_v4/v41 稀疏 forward 读取 packed KV 行的参数化测试(需 SM100/SM103,否则 skip);`tests/v1/attention/test_flashmla_sparse_sm90.py` 扩展 wiring 测试并新增 CUDA `pack_indices` 的 graph replay 参数化测试(width 128/2048/2176);`tests/samplers/conftest.py` 新增 autouse fixture,在 ≤16 SM 且显式开启 FlashInfer 采样器时跳过采样器套件,`test_topk_topp_sampler.py` 的支持性判定同步。回归风险面广:所有经 FlashInfer 的路径(MLA 稀疏后端、DSv4/DSv4.1 稀疏注意力、top-k/top-p 采样、GDN prefill、MNNVL allreduce 等)随版本变化行为可能改变;≤16 SM 的 GPU 上 FlashInfer 采样器会静默回退非 FlashInfer 路径。

## 关联

与依赖 FlashInfer 接口的 PR 相关:PR #32846(GDN prefill 用 FlashInfer 内核)、PR #50932(MNNVL allreduce buffer 容量,依赖 flashinfer-python==0.6.15.post1 时代的接口)。本卡所在批次素材中未提供更完整的同区域 PR 列表,暂无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
