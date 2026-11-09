# PR #34331: [quantization] Add tuned Triton tile configs for channelwise FP8 GEMM on L40S

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @RunkaiTao |
| 合并 | 2026-08-14 |
| 分类 | Quantization(major) |
| Label | run-ci, jit-kernel, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示 4 个新 L40S JSON 配置 + fp8_kernel.py / environ.py / fp8_utils.py 改动 |
| 链接 | https://github.com/sgl-project/sglang/pull/34331 |

## 一句话结论

L40S 上为 per-token/per-channel FP8 GEMM 加入 tuned Triton tile 表,离线 sweep 中最高 1.8x(o_proj M=512),Qwen3-8B FP8 端到端输出吞吐 +3.0%..+6.1%。

## 动机

`apply_fp8_linear` 在权重形状 CUTLASS 兼容时分派到 CUTLASS `fp8_scaled_mm`,否则落到 Triton `scaled_mm_kernel`(粗粒度按 M 的 tile 启发式)。在 L40S/SM89 上对 Qwen3-8B 四个 FP8 linear 形状的离线 sweep 发现:tuned Triton tile 几乎在所有 token count 上胜过 CUTLASS 分派——小 M 时快 3–10%,M=512 时最高 1.8x——因为默认启发式只从四个 tile 里选,且从不调 `num_warps`/`num_stages`。

## 优化点

沿用既有 `get_w8a8_block_fp8_configs` 的约定(device 命名的 per-shape JSON + 最近 M 查找)为 channelwise FP8 GEMM 加 tuned 配置表:新增 `get_w8a8_channelwise_fp8_config(N, K, M)`,返回 tuned tile 或 None;`triton_scaled_mm` 增加透传 `num_warps`/`num_stages` 的能力(仅在调用方设置时作为 launch kwargs 传入,默认路径保持 Triton 自身默认);`apply_fp8_linear` 在 flag 开启且该形状/M 存在 tuned tile 时走带 tuned tile 的 Triton 路径,否则行为不变。配置只在 L40S 上生效(按 `device_name` 键控,其他 GPU/未 tune 形状查找返回 None,分派逐字节不变);CUTLASS 更快的 M 点在 JSON 里存 null 占位以防邻近 M 吸附过去,调用方回落 CUTLASS。默认开启,`SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=0` 为 kill switch。torch.compile 下查找返回 None(宿主侧 device-name/文件 I/O 不可 trace),compiled prefill 保持 CUTLASS 路径。

## 关键改动

```text
python/sglang/kernels/ops/quantization/fp8_kernel.py: 新增 get_w8a8_channelwise_fp8_
  configs/config(lru_cache 加载 JSON、最近 M 查找);triton_scaled_mm 支持 num_warps/
  num_stages 透传与 use_heuristic 关闭
python/sglang/srt/layers/quantization/fp8_utils.py: apply_fp8_linear 在原走 CUTLASS 的
  分支上接入 tuned Triton tile
python/sglang/srt/environ.py: 新增 SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE(默认 True)
configs/N=*,device_name=NVIDIA_L40S,dtype=fp8_w8a8_channelwise.json (x4): Qwen3-8B 的
  qkv/o/down/gate_up 四个 FP8 linear 的 tuned tiles
```

关键摘录(最近 M 查找):

```python
def get_w8a8_channelwise_fp8_config(N, K, M):
    configs = get_w8a8_channelwise_fp8_configs(N, K)
    if not configs:
        return None
    return configs[min(configs.keys(), key=lambda x: abs(x - M))]
```

## 性能声明

Kernel 级(L40S 离线 sweep,Qwen3-8B 四形状 vs 默认 CUTLASS-or-heuristic-Triton 分派):几乎每个 M 都胜出,典型 1.0–1.3x,最高 1.8x(o_proj,M=512);qkv_proj M=256 为 1.402x、M=192 为 1.389x;down_proj M=8192 为 1.461x;gate_up_proj 的 M=512/1024/2048 CUTLASS 胜出(回落,1.000x)。

端到端(L40S,Qwen3-8B FP8,random 1500-in/500-out 单 wave,每格 5 次):并发 1 输出吞吐 +5.1%(75.17→79.01 tok/s)、TPOT −4.9%;并发 8 输出吞吐 +6.1%(461.22→489.27 tok/s)、TPOT −5.7%;并发 32 输出吞吐 +3.0%(1055.87→1087.88 tok/s)、TTFT −8.8%。TTFT 在 c=1/8 噪声大(单 wave,run-to-run std 至 ±37ms),不计为收益。GSM8K 全 1319 例:0.9346 vs baseline 0.9193(+1.5pp,在 ±0.7pp 单次运行方差内,判定无精度退化);tuned run 同时端到端快 3.1%、输出吞吐高 2.4%。

## 验证与回归风险

Kernel 级精度:实机 L40S 上 4 个 tuned 形状与 baseline 比 cos 0.999994–1.000000;CUTLASS 胜出的 M 点回落原路径(maxdiff=0)。模型级 GSM8K 精度如上,确认 Triton tile 不同的累加顺序不跨层累积。回归风险:行为变化仅限 L40S 的四个 tuned 形状(默认走 Triton 路径,数值轻微偏移,PR 明确建议 L40S bit-exact/精度基线需重测或用 kill switch);配置按 device_name 键控不外溢;torch.compile 下 no-op;CI Extra 通过、Base 失败(pack 记录)。

## 关联

无(pack 内未提及具体关联 PR;方法上对齐既有 block-FP8 的 `get_w8a8_block_fp8_configs` 默认开启 JSON 表路径)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
