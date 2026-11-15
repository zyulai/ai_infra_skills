# PR #47842: [Fix] diffusion: Enable Cosmos 3 Edge and Nano fused QK-norm/RoPE on SM 121 (GB10)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JackZeng0208 |
| 合并 | 2026-11-13 |
| 分类 | Attention / Kernel(tier: major) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/47842 |

## 一句话结论

给 `_can_enable_t1_fused_qk_norm_rope` 门控增加 SM121(DGX Spark / GB10,capability (12,1))分支:Cosmos 3 Edge 与 Nano 在 DiT resident 下 T=1 图像请求快 3.8–14.5% 且输出 bit-identical;TP2/SP2/Super 等未测得的 shape 与 topology 明确排除在外。

## 动机

在 DGX Spark 上、DiT resident 时,fused QK-norm/RoPE 路径让 Edge 与 Nano 的图像请求快 3.8–14.5%,输出保持 bit-identical。#34347 已让该 kernel 在 SM120 上与 split 路径以相同方式取整,为开启铺路。正文 Motivation 行写的是 "Enable `_can_enable_t1_fused_qk_norm_rope` for SM 120",diff 实际按 capability `== (12, 1)`(SM121)分支并新增 `is_sm121` 参数;SM120 本身未实测(由 RTX 5090 runner 的 parity test 检查 numerics)。SM10x 与 Hopper 不变。

## 优化点

门控新增 `is_sm121` 分支,仅当 `tp_size == 1 and sp_size == 1` 且 `(hidden_act, hidden_size)` 为 Edge `(relu2, 2048)` 或 Nano `(silu, 4096)` 时放行;每个条件都来自 Spark 上的实测:

- Edge/Nano 所有实测 T=1 输入(DiT resident)更快且 bit-identical:Edge −9.0% 到 −14.5%,Nano −3.8% 到 −8.3%。
- `tp_size == 1`:两台 DGX Spark 上 TP2 无一致收益(Nano 慢 0.9–2.2%),且 Edge 在 main 上同请求跨 run 就返回不同图像,故排除。
- `sp_size == 1`:SP 下 T=1 时 fused kernel 根本不跑,开门只改 RoPE cache dtype,SP2 慢 3–15%,故排除。
- Super 及其他 shape 不放行:Super 未用 fused 路径测过;Hopper 上 dense Super shape 反而变慢。
- CFG parallel 每 rank 在 tp 1 / sp 1 下运行,故被放行,实测快 4–10%。

测试侧:gate 单测覆盖放行/排除的 shape 与 topology(测试类改继承 `CustomTestCase`);`test_cosmos3_edge_fusions.py` 的 split-vs-fused parity test 现在也跑 capability 12,并覆盖 Edge(16 query heads)与 Nano(32 query heads)两种 layout,注册到 `1-gpu-small`(RTX 5090)使 CI 检查 SM120 numerics。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py: _can_enable_t1_fused_qk_norm_rope 增加 is_sm121 参数与分支;forward 传入 capability == (12, 1)
python/sglang/multimodal_gen/test/unit/test_cosmos3.py: 门控单测补 SM121 放行/排除用例,改继承 CustomTestCase
test/registered/kernels/ops/diffusion/test_cosmos3_edge_fusions.py: parity test 参数化 q_heads(16/32),capability 9 或 12 均跑,注册 1-gpu-small
```

```python
    if is_sm121:
        shape = (hidden_act, hidden_size)
        return (
            tp_size == 1 and sp_size == 1 and shape in (("relu2", 2048), ("silu", 4096))
        )
```

## 性能声明

(两台 DGX Sparks,BF16,arms main/PR/PR/main,每臂新起 `sglang serve`,1 warmup + 3 measured;saved-request wall time 由 client 从请求发出到响应解码测得,存图不计)

- 代表点:Edge 640x640 35 步 manual placement 1.764→1.556 s(−11.8%);Nano 1024x1024 35 步 auto placement(DiT resident)13.669→12.679 s(−7.2%)。
- DiT resident 其他输入:Edge 20 输入 −9.0% 到 −14.5%;Nano 18 输入(20 种输入与 placement 组合)−3.8% 到 −8.3%。
- CFG parallel 两台 Spark:Edge −10.1% / −8.0%,Nano −5.0% / −4.3%。
- Offload 无一致变化:Edge layerwise 8 arms 7.129→7.136 s;Nano layerwise 64.37→63.93 s;Edge auto(whole-DiT CPU offload)+3.1% / +0.5%(arm 均值波动可达 12%)。
- Profile(6 profiler steps):Edge kernel launches 11282→3890、GPU kernel time 283.3→261.2 ms;Nano launches 14258→4754、GPU kernel time 2473.2→2279.8 ms。
- `bench_diffusion_denoise.py` preset(ABBA):Edge lossless/exact/high E2E 1816.8→1605.2(−11.6%)/1822.1→1598.5(−12.3%)/1764.0→1573.6 ms(−10.8%),Denoise −13.2%/−14.1%/−11.9%;Nano lossless/exact/high E2E −6.8%/−5.9%/−6.8%,Denoise −7.0%/−6.1%/−7.0%。BCG cell 不评估(Cosmos3 不 capture graph,回退 eager)。

## 验证与回归风险

- 准确度:本 PR 放行的每个实测配置在 main 与 PR 上图像 bit-identical(SSIM 1,PSNR inf),覆盖 `cosmos3-edge-t2i` / `cosmos3-nano-t2i` preset 的 exact/lossless/high、`sglang serve` 请求、48 组额外输入与 placement 组合加一次重复 offload run;部分 case 在 auto/manual/layerwise-offload placement 与两台 Spark 的 CFG parallel 下复跑,probe build 确认其经 `fused_qknorm_rope_warp` 于 T=1 运行。
- Parity test 在 DGX Spark 10 passed:作者把 fused call 弄坏四种方式(rounding off、non-neox RoPE、K/V prefixes swapped、K norm weight 用于 Q),每种都使全部 6 个 parity case 失败。`test_cosmos3.py` 145 passed(main 143)。Super 对照(TP2 两台 Spark + layerwise offload)留在 split 路径,输出 main 与 PR 一致。
- 风险:SM120 未实测,只能靠 RTX 5090 runner 的 parity test 检查 numerics;TP2/SP2/Super 已被门控明确排除并配有单测。

## 关联

#34347(使 fused kernel 在 SM120 上与 split 路径同方式取整,为本 PR 铺路)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
