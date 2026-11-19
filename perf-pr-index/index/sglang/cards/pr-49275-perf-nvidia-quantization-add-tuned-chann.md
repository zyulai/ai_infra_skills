# PR #49275: [Perf] [NVIDIA] quantization: Add tuned channelwise FP8 GEMM tile configs for RTX PRO 5000

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @muxixibbb-spec |
| 合并 | 2026-11-19 |
| 分类 | Quantization(major) |
| Label | documentation, quant, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/49275 |

## 一句话结论

为 SM120(RTX PRO 5000 72GB Blackwell)离线调优 per-token×per-channel FP8 GEMM(`triton_scaled_mm`)的 tile 表:5 个权重形状 × 24–25 个 M 桶的 JSON 配置,74/123 个桶换上调优 tile 后内核最快 2.81×,MiniCPM5-2.6B decode 输出吞吐 c=1 +66.5%、c=8 +52.2%、c=32 +28.0%,追平 vLLM 0.33。

## 动机

SM120 上 channelwise FP8(如 `compressed-tensors` W8A8 checkpoint)走固定 dispatch,decode 吞吐明显落后:vLLM 0.33 在 MiniCPM5-2.6B 上 c=1 输出 218 tok/s(ITL 4.53 ms),而 SGLang channelwise CUTLASS baseline 仅 138(7.19)/968(7.92)/2851(9.76),调优后 229.9(4.30)/1467.3(5.10)/3644.7(7.32)。解法是"调参而非加内核":复用 L40S(#34331)已合并的设备名 JSON 表机制,每 (N, K) 一表、最近 M 查找、`null` 表示 CUTLASS 更快处保持 dispatch;不触碰任何 dispatch/内核/模型/launcher 代码。

## 优化点

- 5 张表:`N={2560,2048,12288,130560},K={2048,6144},device_name=NVIDIA_RTX_PRO_5000_72GB_Blackwell,dtype=fp8_w8a8_channelwise.json`,覆盖 2560×2048(qkv)、2048×2048(o)、2048×6144(down)、12288×2048(gate_up)、130560×2048(lm_head),M=1 到 4096,每表 24–25 桶。
- 74 个桶给调优 tile(内核级 1.03×–2.81×),49 个 `null` 桶保持 CUTLASS(含 4 个复测输掉的桶与 12288×2048 的 M=80/96/128)。
- 表按精确设备名匹配,其他 GPU 返回 None 保持原 dispatch;`README.md` 与 quantization 文档补齐 channelwise 表格式说明(#34331 未写文档)。
- 与 #38006 的 per-tensor SM120 路线互不干扰:per-tensor 权重仍走 `torch._scaled_mm`,不查这些表;本 PR 保留 checkpoint 自带的 per-channel scale 与 per-token 动态激活 scale,避免重定标把相对 RMS 从 0.0265 抬到 0.0375。

## 关键改动

```text
kernels/ops/gemm/configs/N=*,K=*,device_name=NVIDIA_RTX_PRO_5000_72GB_Blackwell,dtype=fp8_w8a8_channelwise.json × 5: 新增调优表(如 2048×6144 M=1: BLOCK 16×32×512 / 4 warps / 4 stages)
kernels/ops/gemm/configs/README.md: 补 channelwise 表格式与 null 语义
docs/docs/advanced_features/quantization.mdx: 文档化 channelwise 表
test/registered/kernels/ops/gemm/test_channelwise_tuned_configs.py: 新增 6 tests / 18 cases
```

## 性能声明

(全部照抄 PR 正文,未独立核实)

- 内核级(生产入口 `apply_fp8_linear` channelwise 分支,真 FP8 权重、CUDA graph 捕获、3 轮 15 次 replay 取中位,RTX PRO 5000, torch 2.14.1+cu130):74 个带 tile 的桶 1.03×–2.81×(如 2048×6144 M=1:15.3 vs 43.1 us,2.81×;130560×2048 M=1:268.2 vs 372.3 us,1.39×);49 个 `null` 桶 1.00×。
- 端到端 MiniCPM5-2.6B(1500 输入 / 500 输出,5 次重复,表缺失 vs 存在):输出吞吐 138.2→230.0(+66.5%)/968.4→1473.5(+52.2%)/2849.3→3648.1(+28.0%),ITL 7.19→4.30(−40.2%)/7.92→5.07(−36.0%)/9.76→7.32 ms(−25.0%),TTFT 不变(prefill 不用这些形状)。
- 相对 #38006 per-tensor 路线:同权重同 harness 下调优 tile 在 74 个桶 1.02×–1.95×;GSM8K 8-shot 1319 题:orig 84.23%、tuned 84.53%(McNemar p=0.76)、bf16 参考 85.90%。

## 验证与回归风险

新测试覆盖表格式合法性、最近 M 查找语义、null 桶不借用邻居 tile、其他设备 no-op、调优 tile 对 FP32 参考的相对 RMS < 2e-2;18 cases 在 RTX PRO 5000 全过(注册 `base-b / 1-gpu-small`,20 s)。风险面:仅 5 个 (N,K) 形状对这一块 GPU 生效,纯数据文件,误配表会导致静默禁用调优或内核拒绝 tile(测试已覆盖);Base CI 通过,Extra 跳过(未加 run-ci-nv-extra 标签)。

## 关联

#34331(L40S channelwise 表机制)、#38006(per-tensor SM120 路线,互补不竞争)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
