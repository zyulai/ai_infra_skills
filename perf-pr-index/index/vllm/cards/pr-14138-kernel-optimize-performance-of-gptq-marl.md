# PR #14138: [Kernel] optimize performance of gptq marlin kernel when n is small

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jinzhen-lin |
| 合并 | 2025-03-07 |
| 分类 | Quantization(tier: major) |
| Label | performance, ready |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 本身可用(6 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/14138 |

## 一句话结论

给 GPTQ/AWQ marlin GEMM 增加可选的 atomicAdd 输出规约路径,免去 slice 间的 barrier 同步;n 小 k 大时 kernel 耗时从 0.095 ms 降到 0.016 ms(约 6 倍)。

## 动机

gptq/awq marlin kernel 在 n(输出维度)小时性能很差,原因是 `barrier_acquire`/`barrier_release` 全局锁同步占用过多时间。n 小的场景常见于大 tensor parallelism 推理或特殊模型:例如 DeepSeek-V3 的 `kv_a_proj_with_mqa` 形状为 k=7168、n=576。

## 优化点

CUDA kernel 新增 `use_atomic_add` 参数:当 `slice_count > 1` 时,各 block 用 half2 粒度的 `atomicAdd` 直接把局部结果累加进全局输出 C,从而完全跳过 `barrier_acquire`/`barrier_release` 与全局 reduce 路径;scale 的 cp.async 加载与结果写出的条件从 `last` 放宽为 `last || use_atomic_add`,atomic 路径下 C 需要 `torch.zeros` 初始化(非 `empty`)。附带两处分配优化:`a_tmp` 仅在 has_act_order 时分配,`c_tmp` 仅在 use_fp32_reduce 时分配。默认关闭:atomicAdd 以 fp16 累加,作者不确定是否会重新引入 #6795 修复过的精度问题("need more test"),故经 `VLLM_MARLIN_USE_ATOMIC_ADD` 环境变量启用;`should_use_atomic_add_reduce` 进一步按形状判定——仅当 `max(m,64)*n < 64*2048 and k >= 2048`(m*n 小且 k 大)时启用,且 CC < 9.0 + bf16(无原生 fp16/bf16 atomicAdd 支持)显式排除,非 CUDA 设备一律关闭。

## 关键改动

```text
csrc/quantization/gptq_marlin/gptq_marlin.cu: kernel 增加 use_atomic_add;slice_count>1 时 half2 atomicAdd 规约并跳过 barrier/global reduce
vllm/model_executor/layers/quantization/utils/marlin_utils.py: should_use_atomic_add_reduce 按 m/n/k/dtype/device 判定;apply_gptq_marlin_linear / apply_awq_marlin_linear 接入
vllm/envs.py: 新增 VLLM_MARLIN_USE_ATOMIC_ADD(默认 False)
vllm/_custom_ops.py + csrc/torch_bindings.cpp: gptq_marlin_gemm 签名新增 use_atomic_add
tests/kernels/test_marlin_gemm.py: test_gptq_marlin_gemm 参数化 use_atomic_add
```

```cuda
if (use_atomic_add && slice_count > 1) {
  scalar_t2* C_half2 = reinterpret_cast<scalar_t2*>(&C[c_gl_wr]);
  scalar_t2* sh_red_half2 = reinterpret_cast<scalar_t2*>(&sh_red[c_sh_rd]);
  #pragma unroll
  for (int a = 0; a < 4; a++) {
    atomicAdd(&C_half2[a], sh_red_half2[a]);
  }
} else {
  C[c_gl_wr] = sh_red[c_sh_rd];
}
```

## 性能声明

PR 内含简单 benchmark(triton `do_bench`,fp16,固定 m=16、k=8192,n 从 128 到 32768;硬件未注明):marlin-atomic-add-false → marlin-atomic-add-true 耗时(ms)n=128:0.095163→0.0159413;n=256:0.0539876→0.0168648;n=512:0.0413818→0.0188273;n=1024:0.0320496→0.0213404;n=2048:0.0272889→0.024638;n≥4096 后 atomicAdd 不再有优势(如 n=4096:0.0300557 vs 0.0323643,略慢)。同表附 torch 与非 marlin gptq kernel 作参照。作者结论:atomicAdd 在 n ≤ 2048 时提升性能。

## 验证与回归风险

`tests/kernels/test_marlin_gemm.py` 的 `test_gptq_marlin_gemm` 新增 `use_atomic_add`(True/False)参数化,含 opcheck;默认关闭,行为不变,启用需 `VLLM_MARLIN_USE_ATOMIC_ADD=1`。风险:atomicAdd 规约以 fp16 累加,精度上可能复现 #6795 修复过的问题,作者标注需要更多测试;CC < 9.0 的 bf16 与非 CUDA 路径被排除,不受影响;签名变更触及 `_custom_ops`、torch bindings 与两个 apply 入口。

## 关联

#6795(此前 fp16 atomic reduce 精度问题的修复,作者担心复现)。同分类 Quantization 区域其他相关 PR 未在素材中列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
