# PR #35981: [diffusion] Flatten Wan VAE RMSNorm row addressing

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-08-24 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, run-ci, diffusion, jit-kernel, bypass-fastfail, run-ci-extra, mergeable |
| Diff | diffstat 未提供(pack 素材无增删行数字) |
| 链接 | https://github.com/sgl-project/sglang/pull/35981 |

## 一句话结论

把 Wan VAE RMSNorm+SiLU Triton kernel 的行寻址扁平化为稠密 [pixel, channel] 行,单 kernel 提速约 37-40%。

## 动机

Nsight Compute 对代表性 FastWan2.1 形状(`[1, 96, 4, 480, 832]`,BF16 输入/FP32 affine)显示:SM throughput 已达 80.58%,但 DRAM throughput 仅 9.14%(occupancy 93.93%)——瓶颈不在访存而在索引计算。源码计数器把 196,793 个采样中的 120,771 归因于整数/索引指令,最热的 PC 是用倒数下发的 row→b/t/h/w div/mod 链(`I2F.RP`、`IABS`、`MUFU.RCP`)。而 dense channels-last-3d 本来就把每个像素存成一条连续 channel 行,恢复 b/t/h/w 坐标纯属多余。

## 优化点

`wan_rmsnorm_silu` Triton kernel 删除 per-program 的 b/t/h/w div/mod 链与全部 10 个 5D stride 参数,行地址直接写为 `row * channels + offsets`;归一化、dtype 边界、SiLU 路径、launch geometry 与输出 layout 均不变。入口 guard `can_use_wan_rmsnorm_silu` 显式增加 `x.stride(1) == 1`:size-one channel 张量可能满足 channels_last_3d 的 memory-format 谓词却保持 channel-first strides,扁平寻址 kernel 对其不成立,必须回退。同时新增注册在 CI 的 cold-L2 benchmark 与 kernel 文档表格更新。

## 关键改动

```text
python/sglang/kernels/ops/diffusion/norm/wan_rmsnorm_silu_triton.py: kernel 扁平行寻址 + stride(1)==1 guard
python/sglang/kernels/ops/diffusion/README.md: 文档标注 dense channels_last_3d (stride(C)==1) 前提
test/registered/kernels/benchmark/diffusion/bench_wan_rmsnorm_silu.py: 新增注册 benchmark(两个代表形状)
test/registered/kernels/ops/diffusion/test_norm.py: 新增非稠密行拒绝测试
```

核心改动(旧代码为 7 行 div/mod 链加双 base 重构,现为一行寻址):

```python
# Dense channels-last-3d stores each pixel as one contiguous channel row.
# Address it directly instead of recovering b/t/h/w with integer div/mod.
row_offsets = row * channels + offsets
x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
```

```python
# Size-one channel tensors can satisfy the memory-format predicate
# while retaining channel-first strides, so require dense rows too.
and x.stride(1) == 1
```

## 性能声明

单 NVIDIA H200,cold-L2 registered benchmark(main → PR):

- FastWan2.1 BF16 `[1,96,4,480,832]`:1550.848 → 972.576 us(**37.29%**);FastWan2.2 FP32 `[1,256,4,384,576]`:921.184 → 553.760 us(**39.89%**)。

端到端(ABBA,eager,resident DiT):

- FastWan2.1 832x480x61、3 steps、no compile:lossless -0.10%(control,kernel 未挂载);quality=high e2e **+3.84%**(mean denoise/e2e 0.8513/1.9575 → 0.8533/1.8823 s)。quality-high full-stage trace 中全部 415 个 Wan RMSNorm+SiLU launch 从 215.234 ms 降至 127.128 ms(**40.94%**),最大 C96 形状 171.438→101.697 ms(105 launches);peak reserved memory 31.402 GiB 不变。
- TurboWan2.1 T2V 1.3B,832x480x81、4 steps、quality=high:e2e **+3.76%**(2.4235→2.3325 s);denoise 慢 0.57%,作者以此确认收益来自 VAE 路径而非 DiT 计时漂移。
- TurboWan 14B 分辨率扩展检查(同样 545 launches):480P kernel time 289.647→169.135 ms(**-41.61%**)、e2e +1.20%;720P 669.365→389.957 ms(**-41.74%**)、e2e +0.96%。
- TurboWan2.2 I2V A14B,4x H200 CFG/Ulysses,1280x720x81:rank-0 GPU time 12,020.198→11,866.292 ms(-1.28%),但 synchronized e2e 回退 0.32%,作者定性为验证证据而非请求加速声明。
- 全部 ABBA 输出 byte-exact(FastWan 与 TurboWan 各自 SHA256 一致)。

## 验证与回归风险

验证:`pytest test_norm.py -k wan_rmsnorm_silu` 4 passed;`test_model_fast_paths.py -k wan_vae_gate` 1 passed;新增 `test_wan_rmsnorm_silu_rejects_non_dense_channel_rows`(C=1 channel-first strides 必须走回退并抛 ValueError);pre-commit 通过;CI Base/Extra 通过,AMD ROCm 7.2 为 pending。风险:扁平寻址只在 dense 行(stride(C)==1)下成立,guard 漏判会产生错误输出——由拒绝测试与 byte-exact ABBA 覆盖;C=1 及非稠密布局回退后失去该 kernel 的收益。

## 关联

同区域:Wan VAE/DiT kernel 优化链(#41710、#41830 处理 Wan VAE 首请求编译开销,本 PR 优化常驻的 RMSNorm+SiLU kernel 本体);同属 Attention / Kernel 分类。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
