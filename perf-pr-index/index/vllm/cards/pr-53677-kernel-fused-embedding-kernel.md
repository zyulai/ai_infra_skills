# PR #53677: [kernel] Fused embedding kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jeejeelee |
| 合并 | 2026-08-31 |
| 分类 | Attention / Kernel（tier: major） |
| Label | performance, ready, ci/build |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/53677 |

## 一句话结论

新增 fused vocab-parallel embedding CUDA kernel，把 TP>1 的 mask、id shift、gather、masked_fill 合为一次 pass。

## 动机

PR 正文未用文字说明动机：Purpose 仅给出 before/after 两张截图，Test Plan 仅 "Perf comparsion" 一张截图，Test Result 为空。从 diff 头部注释可见瓶颈所在——TP > 1 时每个 rank 只持有词表的一个切片，eager 路径需要依次做 range mask、id 偏移、int64 转换、gather，以及对 gathered rows 的 `masked_fill_`；本 kernel 把这一串操作合并为一次遍历完成。正文除此之外的动机描述属于「PR 未注明」。

## 优化点

`csrc/libtorch_stable/vocab_parallel_embedding_kernels.cu` 新增 `vocab_parallel_embedding_kernel`：属于本 rank 的 token 直接按行 gather 权重，属于其它 rank shard 的 token 写出零行，从而保持紧随其后的 all-reduce 结果精确。由于 gather 只是纯行拷贝，kernel 以 byte-vector 类型（`vec_t`）而非 value type 模板化，可适用于任意 embedding dtype；配合 `kBlockThreads=256`、`kTokenWaves=4`，token 循环按 `gridDim.y` 跨步覆盖剩余 token。`vllm/_custom_ops.py` 新增 `vocab_parallel_embedding()` 包装并调用 `torch.ops._C.vocab_parallel_embedding`。`vllm/model_executor/layers/vocab_parallel_embedding.py` 在 `__init__` 计算 `use_fused_embedding = tp_size > 1 and current_platform.is_cuda() and isinstance(quant_method, UnquantizedEmbeddingMethod)`（注释说明 fused kernel 假定 plain row-major shard，故只用于未量化权重）；`forward()` 中 tp_size==1 直接走原路径，命中 fused 时调用自定义 op 并在多维输入下 reshape 视图，否则回退原 `get_masked_input_and_mask` 路径，末尾统一执行 `tensor_model_parallel_all_reduce`。CMakeLists.txt、ops.h、torch_bindings.cpp 负责把新 kernel/op 纳入构建与注册。

## 关键改动

```text
csrc/libtorch_stable/vocab_parallel_embedding_kernels.cu: 新增 fused kernel，一次 pass 完成 mask/shift/gather/置零
vllm/model_executor/layers/vocab_parallel_embedding.py: 新增 use_fused_embedding 判定与 forward 分支改写
vllm/_custom_ops.py: 新增 vocab_parallel_embedding() 包装调用自定义 op
csrc/libtorch_stable/ops.h、torch_bindings.cpp、CMakeLists.txt: op 注册与构建接入
benchmarks/kernels/benchmark_vocab_parallel_embedding.py、tests/kernels/core/test_vocab_parallel_embedding.py: 新增 benchmark 与正确性测试

# 机制摘录 (vocab_parallel_embedding_kernels.cu)
    if (in_org || in_added) {
      const int64_t row =
          in_org ? id - org_vocab_start_index : id + added_row_offset;
      const vec_t* weight_row = weight + row * vecs_per_row;
      for (int64_t v = vec_begin; v < vecs_per_row; v += vec_stride) {
        out_row[v] = weight_row[v];
      }
    } else {
      const vec_t zero = vec_t();
      for (int64_t v = vec_begin; v < vecs_per_row; v += vec_stride) {
        out_row[v] = zero;
      }
    }
```

## 性能声明

PR 正文未给出定量性能数据。Purpose 的 before/after 与 Test Plan 的 "Perf comparsion" 均为 `<img>` 图片链接（截图），正文中无可读取的数值；Test Result 段为空，PR 未注明测试条件。因此无可照抄的 benchmark 数字与模型/硬件/精度/workload 条件。

## 验证与回归风险

PR 正文 Test Plan 只列出 "Perf comparsion" 图片、Test Result 为空，未给出文字测试结论。diff 新增 `benchmarks/kernels/benchmark_vocab_parallel_embedding.py` 与 `tests/kernels/core/test_vocab_parallel_embedding.py`，可视为随 PR 引入的 benchmark 与正确性测试。回归范围由 `use_fused_embedding` 判定限定：仅 `tp_size > 1`、CUDA 平台、且权重为 `UnquantizedEmbeddingMethod` 时启用；tp_size==1、非 CUDA、量化 embedding 仍走原有路径。kernel 假定 plain row-major shard，注释声明因按 byte-vector 模板化故支持任意 embedding dtype。正文未给出精度/数值等价性回归结论，也未注明测试硬件与模型。

## 关联

无。PR 正文与 diff 未给出同区域其它 PR 编号或前序/后续线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
