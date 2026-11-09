# PR #32683: [Quantization][Deprecation] Remove BitBlas

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @robertgshaw2-redhat |
| 合并 | 2026-01-28 |
| 分类 | Quantization(major) |
| Label | documentation, performance, ready |
| Diff | diffstat 缺失(pack 未记录 +N/-N);diff 内容可用(以删除为主,另有 5 个文件因篇幅未收入) |
| 链接 | https://github.com/vllm-project/vllm/pull/32683 |

## 一句话结论

按弃用计划(0.14 已预告,0.15 执行)整体移除 BitBLAS / GPTQ-BitBLAS 量化后端及其文档、测试与 benchmark。

## 动机

PR 正文只有一句:"now that 0.14 is out, deprecate bitblas for 0.15"。素材未给出更多背景(如使用率数据);从被删除代码可见其维护成本信号:BitBLAS 权重需在线 hardware-aware tuning(`hardware_aware_finetune(topk=20)` 写入数据库),对应测试长期 `skipif(True, reason="BitBLAS takes too much time for tuning.")`,且其内部用锁同步带来轻微非确定性、测试需 reruns=2。

## 优化点

纯删除,不引入新机制。移除:后端实现 `bitblas.py`(BitBLASConfig/BitBLASLinearMethod,约 502 行,含 GPTQ checkpoint 的 create_weights_gptq/apply_gptq 与 operator 缓存)与 `gptq_bitblas.py`(约 482 行);`linear.py` 中的 `adjust_bitblas_shard` 分片调整;`vllm/config/model.py` quantization override 优先级列表中的 `bitblas`/`gptq_bitblas`;quantization 注册表与导出;`docs/features/quantization/bitblas.md` 与 README 表格两行;`benchmarks/kernels/benchmark_bitblas.py`;`tests/models/quantization/test_bitblas.py`、`test_gptq_bitblas.py`。附带把 `gptq.py` 中 4-bit gptq_gemm kernel bug 提示从"switch to gptq_marlin or gptq_bitblas"改为"switch to gptq_marlin"。对性能索引的意义:收缩维护面、减少 load 时 tuning 分支,属清理而非加速。

## 关键改动

```text
vllm/model_executor/layers/quantization/bitblas.py: 删除(BitBLAS 线性后端)
vllm/model_executor/layers/quantization/gptq_bitblas.py: 删除(GPTQ checkpoint 走 BitBLAS kernel)
vllm/model_executor/layers/linear.py、vllm/config/model.py、quantization/__init__.py: 清除 shard 调整、override 列表与注册
docs/features/quantization/bitblas.md、benchmarks/kernels/benchmark_bitblas.py、tests/models/quantization/test_bitblas.py、test_gptq_bitblas.py: 删除文档/benchmark/测试
```

## 性能声明

无性能数字,PR 未声明任何加速;效果是移除一个量化后端。使用 `quantization="bitblas"`(含 GPTQ-BitBLAS 格式 checkpoint)的模型在 0.15 起将无法加载;素材 diff 未包含迁移指引。

## 验证与回归风险

删除的两个模型级测试原本已 `skipif(True)` 长期不跑。回归风险面向用户:依赖 BitBLAS 的量化模型与 `gptq_bitblas` checkpoint 失效;gptq.py 中 4-bit kernel bug 警告的替代路径只剩 gptq_marlin;linear.py 的 weight_loader 少一次 `adjust_bitblas_shard`(对无 bitblas_tile_size 的参数是 no-op)。不使用 BitBLAS 的路径无行为影响。

## 关联

素材未给出关联 PR 编号;属 Quantization 分类的后端清理(同分类 kernel 优化另见 #20841、#63915 等卡)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
