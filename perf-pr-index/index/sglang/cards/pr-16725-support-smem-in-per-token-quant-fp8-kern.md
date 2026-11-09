# PR #16725: support smem in per_token_quant_fp8 kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zhangxin81 |
| 合并 | 2026-02-02 |
| 分类 | Quantization (tier: major) |
| Label | quant, sgl-kernel, run-ci |
| Diff | pack 未提供 diffstat(diff 完整可见,仅改 sgl-kernel/csrc/gemm/per_token_quant_fp8.cu) |
| 链接 | https://github.com/sgl-project/sglang/pull/16725 |

## 一句话结论

per_token_quant_fp8 在 hidden_dim<2048 时启用带 bank-conflict padding 的 shared memory,免去第二次全局内存读取,hidden=1368 最高提速 17.4%。

## 动机

per-token FP8 量化 kernel 原实现不使用 shared memory:warp-local 路径的 Pass-1 求 token 最大值、Pass-2 量化写回各读一次输入,输入要从 global memory 加载两次。性能分析划分出三个区域:hidden_dim<2048 是 sweet spot,shared memory 优化通过更好的存储层级利用与减少 global 流量带来显著收益;hidden_dim>=2048 时 shared memory 因 bank conflict、寄存器压力与不利的计算访存比成为瓶颈;hidden_dim>=4096 时两者接近且 baseline 更稳。因此策略定为 hidden_dim<2048 默认用 smem,否则保持原实现。

## 优化点

kernel 加模板参数 `USE_SMEM`:Pass-1 将向量加载的数据同时写入 dynamic shared memory 并求 max,`__syncwarp()` 后 Pass-2 直接从 shared memory 读数据量化,省掉第二次 global 读取;`USE_SMEM=false` 时走原逻辑。host 侧按维度决策:`use_smem = (hidden_dim < 2048)`,并保留 48KB 安全检查(超过 `DEFAULT_SHARED_MEM_THRESHOLD_KB` 则关闭 smem,避免 launch 失败)。为避免 bank conflict,每个 warp 的 shared memory 区域做 32 字节对齐 padding(`warp_smem_stride = ceil(hidden_dim*sizeof(T)/32)*32`),kernel 与 host 两侧一致计算,dynamicSmemSz 按 padded stride 计算;每个 warp 的 padding 开销依 hidden_dim 约 0-16 字节。launch 侧抽出 `launch_per_token_quant_fp8_warp_kernel` 按 vec16/vec8/vec4 与 USE_SMEM 分发。

## 关键改动

```text
sgl-kernel/csrc/gemm/per_token_quant_fp8.cu: per_token_quant_fp8_kernel 加 USE_SMEM 模板参数与 warp 级 smem 读写
```

```cuda
if constexpr (USE_SMEM) {
#pragma unroll
  for (uint32_t j = 0; j < kVecSize; ++j) {
    shared_input[i * kVecSize + j] = input_vec[j];
  }
}
max_value = fmaxf(max_value, fabsf(static_cast<float>(input_vec[j])));
...
// Pass-2
if constexpr (USE_SMEM) {
  input_vec[j] = shared_input[i * kVecSize + j];   // 从 smem 读
} else {
  input_vec.cast_load(token_input + i * kVecSize); // 原逻辑:再读一次 global
}
```

## 性能声明

测试环境:H20,cuda 12.8,python 3.12,torch 2.9.1+cu128;PR 给出 batch_size x seq_len x hidden_dim 全量 baseline/after 表(SGL Kernel 列)。hidden_dim=1368 的一致性收益:

- seq_len=4096:最高 17.1% 提升(374.65 → 310.61,batch=16)
- seq_len=1024:最高 14.8% 提升(99.07 → 84.46,batch=16)
- seq_len=256:最高 17.4% 提升(27.07 → 22.34,batch=16)

其他 hidden_dim(2048/4096)保持不变;PR 未注明表中数值单位(正文按 ms 口头引用)与 GPU 计数。

## 验证与回归风险

Accuracy:与 Torch 参考的 scale/output 差异在改动前后基本一致(scale diff 约 0.00000048-0.00000052,output diff 约 0.099-0.117),vLLM 对比因环境不可用而跳过;Checklist 未勾选,diff 中无新增测试。回归面:所有走 warp-local 路径的 `sgl_per_token_quant_fp8` 调用方(hidden_dim<2048 时行为改变,数值上仅量化中间精度路径不同但结果一致);dynamic shared memory 用量随 hidden_dim x TOKENS_PER_CTA 增长,由 48KB 阈值兜底;hidden_dim>=2048 与 baseline 路径不变。

## 关联

同区域:sgl-kernel 的 FP8 per-token 量化 kernel(`per_token_quant_fp8_kernel`/`per_token_quant_fp8_small_batch_kernel`);pack 正文未列出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
