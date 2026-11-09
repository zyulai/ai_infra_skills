# PR #45591: [Perf] moe: Add H200 block-FP8 Triton configurations

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Jzz1943 |
| 合并 | 2026-10-22 |
| 分类 | MoE(major) |
| Label | 无 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示新增 2 个 JSON 配置文件(各 227 行) |
| 链接 | https://github.com/sgl-project/sglang/pull/45591 |

## 一句话结论

为 H200 上 GLM-5.3-Flash(TP4/EP4)的 block-FP8 fused MoE 新增 Up/Down GEMM Triton tuned 配置,较默认启发式提速 1.054x–1.199x。

## 动机

H200 上 GLM-5.3-Flash 以 TP4/EP4 部署时的 fused MoE 形状为 E=72、N=2048、block shape [128,128],该形状此前没有针对 Triton 3.8.0 的 tuned 配置,只能使用默认启发式,存在可观的 kernel 级提速空间。

## 优化点

纯配置新增,零 kernel/runtime 改动:为 Triton 3.8.0 添加 Up 与 Down GEMM 两份配置文件,覆盖 M=1 到 M=8192 共 25 个 input token count,经既有 loader 自动加载。所有配置固定 `BLOCK_SIZE_K=128`,与量化 group 宽度一致;其余 tile 参数(BLOCK_SIZE_M 16/32/64、BLOCK_SIZE_N 64/128、GROUP_SIZE_M、num_warps=4、num_stages 3/4、USE_TMA)按 M bucket 选择——小 M 用 BLOCK_SIZE_M=16,大 M 用 64;Up 配置各 M 档普遍开 USE_TMA,Down 配置在 M<=1536 关 TMA、M>=2048 开 TMA。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/
  E=72,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json:
  Up GEMM 配置(25 个 M 档)
  E=72,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json:
  Down GEMM 配置(25 个 M 档)
```

关键摘录(M=1 的 Up 配置):

```json
"1": {
  "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
  "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 4, "USE_TMA": true
}
```

## 性能声明

H200,Triton 3.8.0,fused MoE CUDA Graph microbenchmark,BF16 inputs + block-FP8 weights:hidden size 4096、72 local experts、intermediate size 2048、top-k 8,不含分布式通信开销;相对默认启发式配置,三次配对试验取中位数:M=1 时 1.199x,M=16 时 1.108x,M=64 时 1.101x,M=128 时 1.074x,M=192 时 1.054x(最小),M=512 时 1.060x,M=768 时 1.064x,M=1024 时 1.073x,M=2048 时 1.102x,M=4096 时 1.110x,M=8192 时 1.085x。整体范围 1.054x–1.199x。PR 未注明端到端 serving 数字。

## 验证与回归风险

数值验证:全部 25 对 Up/Down 配置相对默认启发式通过,采用非均匀 activation 与 weight 量化 scale(rtol=0.03, atol=0.003),held-out routing 样本覆盖多层与全部 4 个 EP rank;JSON/grid 检查与 pre-commit 通过。回归风险极低:只新增 JSON 配置,匹配该 (E, N, device, dtype, block_shape) 组合的调用自动加载,其他形状/硬件不受影响;PR 的 CI 状态显示 base/extra/AMD ROCm 10 均为失败(pack 记录,正文未说明原因)。

## 关联

无(pack 内未提及同区域其他 PR;属 fused MoE Triton tuned config 系列)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
