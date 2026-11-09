# PR #6699: [EP] Add cuda kernel for moe_ep_pre_reorder

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-06-02 |
| 分类 | Attention / Kernel(major) |
| Label | 无 |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/6699 |

## 一句话结论

为 EP MoE 的 pre_reorder 步骤新增 CUDA kernel 替代 Triton 实现,作者声称快 10-20%。

## 动机

`moe_pre_reorder` 是 EP MoE 前向的关键 kernel 之一,负责把每个 token 按 topk 专家分配复写到对应的 gateup 输入槽位(含动态量化 scale 缩放)。此前该步骤用的是 Triton kernel(`pre_reorder_triton_kernel`),作者认为 CUDA 实现可进一步压榨该访存密集步骤的性能。

## 优化点

新增 CUDA kernel `ep_pre_reorder_cuda_kernel`:grid 为 token 数,每个 block 负责一个 token,顺序遍历该 token 的 topk 个专家,专家不在本 rank 的 `[start_expert_id, end_expert_id]` 区间时跳过(EP 局部性过滤);对每条 (token, expert) 按 `src2dst` 映射写入 `gateup_input`,数据搬运用 flashinfer 的 `vec_t<float, 4>` 做 16 字节向量化 load/store,循环内直接乘上 scale。scale 逻辑与 Triton 版一致:动态量化且 per-token 时取 `1/a1_scales[token_idx]`,否则取 `1/a1_scales[expert_id - start_expert_id]`。block 固定 512 线程。同时注册 `sgl_kernel.ep_moe_pre_reorder` torch op 并暴露 Python 包装,附一个与 `bench_moe_fused_gate` 风格一致的 Triton benchmark 脚本对比 CUDA vs Triton。

## 关键改动

```text
sgl-kernel/csrc/moe/ep_moe_reorder_kernel.cu: 新增文件;CUDA kernel 实现(hidden 4 维向量化搬运 + scale 缩放)
sgl-kernel/python/sgl_kernel/moe.py / __init__.py / csrc/common_extension.cc: 注册 ep_moe_pre_reorder op
sgl-kernel/benchmark/bench_moe_ep_pre_reorder.py: 新增文件;CUDA vs Triton 对比 benchmark
sgl-kernel/CMakeLists.txt / include/sgl_kernel_ops.h: 构建与声明
```

```cpp
for (int k = 0; k < topk; ++k) {
  int expert_id = token_topk_ids[k];
  if (expert_id < start_expert_id || expert_id > end_expert_id) continue;
  ...
  constexpr uint32_t vec_size = 16 / sizeof(float);
  using vec_t = flashinfer::vec_t<float, vec_size>;
  for (int idx = tid; idx < hidden_size / vec_size; idx += blockDim.x) {
    vec_t input_vec, output_vec;
    input_vec.cast_load(src_ptr + idx * vec_size);
    for (uint32_t i = 0; i < vec_size; ++i)
      output_vec[i] = static_cast<float>(input_vec[i]) * scale;
    output_vec.cast_store(dst_ptr + idx * vec_size);
  }
}
```

## 性能声明

PR 正文只有一句:"The new kernel gains 10-20% performance improvement",未注明模型、硬件、精度或 batch 配置等测试条件。repo 内新增的 benchmark 脚本覆盖 batch 64-4096、hidden_size 4096、topk 8、256 个专家、fp32、block_size 512,与 Triton kernel 用 `triton.testing.do_bench` 对比,但 PR 正文未给出具体数值结果。

## 验证与回归风险

PR checklist 中格式、单元测试、文档、benchmark 结果各项均未勾选;正文未给数值精度对比。仓库内 benchmark 脚本可复跑验证吞吐声明。回归风险:kernel 仅支持 fp32 输入指针(`data_ptr<float>`)与固定 512 线程,若上游调用侧改 dtype/hidden 不对齐 16B 需另行适配;per-token 与 per-expert 两种 scale 分支逻辑必须与 Triton 版严格一致,否则影响量化 MoE 精度;此外该 op 只新增未替换调用点(pack diff 中未改 `ep_moe` 侧调用),实际生效与否取决于后续接线。

## 关联

无(pack 素材未提及其他同分类或前序/后续 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
