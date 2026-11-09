# PR #48642: [Bugfix] Sparse MLA: enable fp8_ds_mla dense prefill

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-07-16 |
| 分类 | Quantization (tier: major) |
| Label | bug, performance, ready, v1 |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 可用但被截断(2 个文件未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/48642 |

## 一句话结论

修复 sparse MLA 在 `fp8_ds_mla` 混合批次下 dense-MHA prefill 路由的越界写与 FP8 cache gather 两个 bug,使 decode/prefill 混合批次可正确运行。

## 动机

前序 PR #47327 为 sparse MLA 引入了 dense-MHA prefill 路由(prefill 走 dense MHA、decode 走 sparse MQA),但暴露了两个 `fp8_ds_mla` 混合批次下的 bug:其一,`req_id_per_token` 可能长于 top-k tensor,导致索引转换 kernel 越界写内存;其二,dense MHA 无法 gather 打包的 656 字节 FP8 KV cache,且 FlashMLA 假定自己的 metadata 覆盖整个 batch,而实际只有 decode 子集。该 PR 取代了 #48612。

## 优化点

机制上做了三件事:第一,给 `triton_convert_req_index_to_global_index` 的调用方加上长度校验,`_forward_bf16_kv` 只把 MQA 子集的 `req_id_per_token` 切片传入,防止越界写。第二,改造 `cp_gather_and_upconvert_fp8_kv_cache` CUDA kernel:删除不用的 `seq_lens` 参数,新增可选 `seq_starts`(每请求的 cache 起始偏移),使 packed FP8 cache 的 gather/upconvert 可以从任意序列位置开始,chunked prefill 的多轮迭代得以正确对齐;同时 workspace dtype 在 `fp8_ds_mla` 下固定为 bf16。第三,metadata builder 引入 `require_uniform_decodes` 约束并复用公共的 `split_decodes_and_prefills`,FP8 分离 prefill/decode 路径改为从公共 metadata 取 decode/prefill 计数,剔除 0-token decode padding,保证 FlashMLA 的 MQA metadata 只覆盖 decode 子集而 dense MHA 处理 prefill。

## 关键改动

```text
csrc/libtorch_stable/cache_kernels.cu: cp_gather_and_upconvert_fp8_kv_cache kernel 增加 seq_starts 偏移支持,签名去掉 seq_lens
vllm/model_executor/layers/attention/mla_attention.py: fp8_ds_mla 下 chunked prefill context(含 DCP 路径)改用 cp_gather_and_upconvert_fp8_kv_cache 并传 seq_starts
vllm/v1/attention/backends/mla/flashmla_sparse.py: metadata builder 增加 require_uniform_decodes,FP8 分离路径复用公共 split 结果并支持 decode 子集
tests/v1/attention/test_sparse_mla_backends.py: 新增越界回归、bf16 kv 切片、FP8 路径 decode 子集等测试
```

kernel 侧核心改动:

```cuda
int token_offset = out_token_id - workspace_starts[req_id];
if (seq_starts != nullptr) token_offset += seq_starts[req_id];
```

## 性能声明

PR 正文无吞吐 benchmark 数字(明确说明"吞吐未比较,因为 scheduler batch sizes 不同")。给出的是 GSM8K 精度验证:`RedHatAI/DeepSeek-V3.2-NVFP4-FP8-BLOCK`,4×B200,TP2/DP2/EP,`FLASHMLA_SPARSE` 后端 + `fp8_ds_mla`,官方 5-shot `lm_eval` 全部 1,319 样本:dense-MHA 路由 Flexible/Strict 89.69%/89.54%,强制 sparse MQA 89.92%/89.69%,差异低于 ~0.84 点标准误,即 dense-MHA 路由无可测精度回归。

## 验证与回归风险

新增/依赖测试:`tests/kernels/test_cp_gather_fp8.py`(FP8 gather,16 passed,含任意 seq_starts 用例)、`tests/v1/attention/test_sparse_mla_backends.py`(回归集 16 passed,515 deselected,含 mixed decode/dense-prefill 正确性用例)、`tests/v1/attention/test_mla_backends.py` 的 kv_cache_spec dtype 测试。风险面:sparse MLA(FlashMLA Sparse)路径、`fp8_ds_mla` cache dtype 别名(`fp8`/`fp8_e4m3` 被规范化到 `fp8_ds_mla`)、chunked prefill 与 DCP(context parallel)场景、CUDA graph UNIFORM_BATCH 约束下的 decode/prefill 切分;对非 MLA 模型无影响。

## 关联

前序 PR #47327(引入 dense-MHA prefill 路由,本 PR 修复其遗留 bug);本 PR 取代 #48612;同属 Quantization/MLA FP8 cache 方向的后续 PR 见索引同分类其他卡。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
