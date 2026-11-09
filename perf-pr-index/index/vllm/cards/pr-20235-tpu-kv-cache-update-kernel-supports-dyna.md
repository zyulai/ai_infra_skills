# PR #20235: [TPU] kv cache update kernel supports dynamic grid

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yaochengji |
| 合并 | 2025-07-02 |
| 分类 | Attention / Kernel(tier: major) |
| Label | tpu, ready, v1 |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动 4 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/20235 |

## 一句话结论

TPU `kv_cache_update` Pallas 内核改用动态 grid,消除为防重编译而 pad 出的大量平凡 DMA 拷贝,e2e 吞吐通常提升 1% 以上。

## 动机

TPU/XLA 下为避免按 batch 形态重编译,`kv_cache_update` 的输入 `slices`(每条为 kv_cache_start/new_kv_start/slice_len)按编译期上界 pad 到固定长度。但每个调度步实际要写的 slice 数远小于上界,pad 出来的大量条目让内核白白执行平凡的 DMA 拷贝,浪费 TPU 内存带宽与周期。

## 优化点

机制是利用 Pallas 内核的 dynamic grid 特性:grid 大小不再取静态的 `slices.shape[1] // num_slices_per_block`,而是 `cdiv(num_kv_update_slices[0], num_slices_per_block)`,其中 `num_kv_update_slices` 以设备端 tensor 形式传入(保持形状静态、值动态,XLA 不会重编译)。为此把 `num_kv_update_slices` 贯穿整条链路:`PallasMetadata` 新增该字段,`tpu_model_runner._prepare_inputs` 由 slot_mapping_metadata 形状取实际 slice 数,`_dummy_run` 用 pad 值,`kv_cache_update_op` 的 XLA op 签名新增该 tensor 参数,`write_to_kv_cache`/`kv_cache_update_op_impl` 透传给 `xb.call_jax`。内核本体逻辑不变,仅 grid 维度动态化,pad buffer 仍保留但多余块不再执行。

## 关键改动

```text
vllm/attention/ops/pallas_kv_cache_update.py: grid 由 num_kv_update_slices 张量动态计算
vllm/v1/worker/tpu_model_runner.py: metadata 携带实际 num_kv_update_slices,dummy run 填 pad 值
vllm/v1/attention/backends/pallas.py: PallasMetadata/write_to_kv_cache 透传该值
tests/v1/tpu/test_kv_cache_update_kernel.py: 传入动态 slice 数的新 op 签名
```

```python
grid=(cdiv(num_kv_update_slices[0], num_slices_per_block), ),
```

## 性能声明

PR 正文仅给出定性结论:通常提供 **超过 1% 的 e2e 吞吐收益**("more than 1% e2e throughput gain usually"),未附具体 benchmark 表格,模型/硬件/workload 条件未注明。

## 验证与回归风险

`pytest -s -v tests/v1/tpu/test_kv_cache_update_kernel.py` 通过(含非整页 slice、page_size、combined_kv_head_num 参数矩阵)。风险点:XLA op 签名变更要求 runner 与 op 同步更新(dummy 路径必须填 pad 值以保持 shape 稳定,否则触发重编译);动态 grid 上界来自设备端值,若 metadata 与实际 slot_mapping 不一致会出现漏写/越界;仅影响 TPU(Pallas/XLA)路径。

## 关联

同区域(TPU attention/缓存 kernel):TPU 侧 kv cache 与 paged attention 相关优化(如 #12348 ROCM paged attention 同类问题的 CUDA/ROCm 对应实现);pack 未提供同 PR 依赖/后续编号。
