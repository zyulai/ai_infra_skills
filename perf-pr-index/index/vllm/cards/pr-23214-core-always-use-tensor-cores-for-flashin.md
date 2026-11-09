# PR #23214: [Core] Always use tensor cores for Flashinfer Decode Wrapper

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @pavanimajety |
| 合并 | 2025-08-21 |
| 分类 | Attention / Kernel(major) |
| Label | performance, ready, v1 |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 与文件数);diff 内容可见 5 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/23214 |

## 一句话结论

Flashinfer 分页 KV decode wrapper 对所有 head group ratio 恒开 tensor cores,不再按 GQA 比例 >4 的启发式切换。

## 动机

此前 `BatchDecodeWithPagedKVCacheWrapper` 仅在 `num_qo_heads // num_kv_heads > 4` 时启用 tensor cores(因为不这么做小比例下会显著掉速),1-4 比例走 CUDA core 路径。作者论点:tensor core 对 MMA 的性能"总是不差于 CUDA core",可以在所有比例下启用,并顺带删除按架构判断的启发式与对应环境变量。

## 优化点

`vllm/v1/attention/backends/flashinfer.py`:decode wrapper 构造固定 `use_tensor_cores=True`(代码注释:最新 GPU 上 tensor core 对所有 attention op 至少与 CUDA core 一样快);`fast_plan_decode` 删除 standard(非 tensor core)plan 分支,统一走 tensor core 版 15 参 `plan` 调用(需要 qo_indptr)。`vllm/envs.py` 删除 `VLLM_FLASHINFER_FORCE_TENSOR_CORES` 环境变量及其 config hash 参与。测试与 benchmark 文件中的启发式同步改为 `True`。

## 关键改动

```text
vllm/v1/attention/backends/flashinfer.py: decode wrapper 恒用 tensor cores;fast_plan_decode 只保留 tensor core plan
vllm/envs.py: 移除 VLLM_FLASHINFER_FORCE_TENSOR_CORES
tests/kernels/attention/test_flashinfer.py 等: 启发式写法改为 use_tensor_cores=True
```

## 性能声明

PR 正文未附 benchmark 数字(仅标注 "CI / simple change");依据为定性判断:tensor cores 对 MMA "always as good as cuda cores"。

## 验证与回归风险

依赖现有 flashinfer 测试(decode with paged KV、FP8 KV、trtllm baseline decode)。回归面:head group ratio 1-4(MHA/MQA 等)模型的 decode 全部切到 tensor core kernel;`fast_plan_decode` 参数布局固定为 tensor core 版 15 参,对 flashinfer 版本兼容性敏感;删除的 `VLLM_FLASHINFER_FORCE_TENSOR_CORES` 不再可用。

## 关联

素材未列出其他同区域 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
