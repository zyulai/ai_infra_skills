# PR #3922: Tuning Script for Feature DeepSeek V3/R1 INT8 Quantization (block-wise)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @laixinn |
| 合并 | 2025-02-27 |
| 分类 | Quantization(tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未记录);diff 可用,另有 7 个文件被截断(详见 PR) |
| 链接 | https://github.com/sgl-project/sglang/pull/3922 |

## 一句话结论

把 fused MoE 与 blockwise 量化 matmul 的 tuning 脚本扩展到 INT8 block-wise(w8a8),并附上 A100 上 DeepSeek V3/R1 的调优配置,吞吐提升 9%(qps=128)/50%(bs=1)。

## 动机

延续 #3730:此前的 tuning 脚本只支持 FP8 block-wise,不支持 block-wise INT8。DeepSeek V3/R1 的 INT8 量化路径要拿到与 FP8 相当的 kernel 性能,同样需要按 (N, K, device, dtype, block_shape) 粒度搜索 Triton config 并落盘 JSON,而当时脚本与配置库都缺这一档,导致 INT8 路径只能用默认 config。

## 优化点

这不是运行时优化,而是补齐 INT8 的调优基础设施并提交调优产物。两个脚本:(1) `tuning_fused_moe_triton.py` 在 `benchmark_config`/`benchmark`/`tune`/`save_configs` 全链路穿入 `use_int8_w8a8` 参数,`--dtype` choices 增加 `int8_w8a8`,权重 randint 生成与 scale 形状按 INT8 分支处理,FP8 独有的 `.to(float8_e4m3fn)` 转换改为仅 `use_fp8_w8a8` 时执行;(2) 原 `tuning_block_wise_fp8.py` 更名为 `tuning_block_wise_kernel.py`,`tune()` 增加 `input_type` 参数按 fp8/int8 生成测试张量,kernel 选择改为按 A 的 dtype 判定——FP8 dtype 走 `_w8a8_block_fp8_matmul(_unrolledx4)`,否则走 `_w8a8_block_int8_matmul`。产出物:两套 JSON 调优配置(E=256,N=128 的 fused MoE config,以及 N=1536,K=1536 / N=1536,K=7168 / N=2048,K=512 / N=2304,K=7168 / N=24576,K=7168 / N=256,K=7168 / N=32768,K=512 的 blockwise matmul config),dtype 均为 `int8_w8a8`,block_shape=[128,128],device 为 NVIDIA_A100-SXM4-80GB。

## 关键改动

```text
benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py: 全链路增加 use_int8_w8a8 分支与 --dtype int8_w8a8
benchmark/kernels/quantization/tuning_block_wise_fp8.py -> tuning_block_wise_kernel.py: 支持 fp8/int8 双输入类型,按 dtype 选 kernel
python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=128,...,dtype=int8_w8a8,block_shape=[128, 128].json: A100 MoE 调优配置(新增)
python/sglang/srt/layers/quantization/configs/N=*,K=*,...,dtype=int8_w8a8,block_shape=[128, 128].json: A100 blockwise matmul 调优配置(7 个,新增)
```

kernel 选择关键逻辑:

```python
if A.dtype == torch.float8_e4m3fnuz or A.dtype == torch.float8_e4m3fn:
    kernel = (
        _w8a8_block_fp8_matmul_unrolledx4
        if (is_hip_ == True and num_workgroups <= get_device_core_count())
        else _w8a8_block_fp8_matmul
    )
else:
    kernel = _w8a8_block_int8_matmul
```

## 性能声明

条件:A100x8x2nodes,DeepSeek V3/R1 INT8 block-wise 量化。tuned config 开启前后:output throughput(qps=128)2426.61 vs 2225.01(+9%);output throughput(bs=1)33.17 vs 22.09(+50%)(正文表格中两列均写作 "Output Throughputs",单位未注明)。

## 验证与回归风险

PR 未提供新的精度测试(正文 checklist 全勾但未附数字)。风险面:改动集中在 benchmark 脚本与新增 JSON 配置,不改运行时代码;新增的 `use_int8_w8a8` 参数按 keyword 传入,调用方不传则默认 None 不受影响。配置文件按 device/dtype/block_shape 命名精确匹配,只影响 A100 + int8_w8a8 + block_shape [128,128] 的组合;其他硬件仍需自行跑 tuning。

## 关联

前序:#3730(tuning 脚本尚不支持 block-wise INT8 的出处)。同区域:FP8 blockwise 调优脚本(即本 PR 更名与泛化的对象)、DeepSeek V3/R1 INT8 量化支持 PR(编号未在正文注明)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
