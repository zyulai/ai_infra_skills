# PR #40386: [Diffusion] Accelerate Cosmos3 Edge on Hopper with lossless fusions

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-09-23 |
| 分类 | Other(major) |
| Label | run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示 5 个文件:activation.py、cosmos3video.py、单元测试、registered benchmark、fusions 测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/40386 |

## 一句话结论

Cosmos3 Edge 在 Hopper 上启用 rounded QK norm/RoPE/KV-pack 融合与 ReLU² 单 kernel,单张 H200 图像 worker E2E 降 29.61–32.23%,视频约 +2.2%,输出逐字节一致。

## 动机

Cosmos3 Edge 的单图路径在 Hopper 上仍在使用分离的 Q/K normalization、RoPE 和 KV concatenation;其 dense MLP 也在 separate square 之前先物化 ReLU。Native profile 显示这两条链出现在全部 28 个 generation blocks 中(且正负 CFG pass 各一次)。这些已有融合 kernel 针对经验证的 Edge 形状未被放行,属于可无损消除的 launch 与访存开销。

## 优化点

两处无损融合:(1) 把 2048-wide dense Edge T=1 形状接入既有的 Hopper TP1/SP1 QK norm/RoPE/KV-pack 路径(`fused_qknorm_rope_pack_kv`),保留 rounded BF16 cache 与 Edge 单独归一化的 UND keys;更大的 dense、并行和 compiled 配置仍保持原有 gate(T2V 路径 baseline 已用 QK 融合,其收益来自 ReLU²)。(2) 给 unary activation compiler 的 `fast_math` 选项暴露参数,Cosmos contiguous BF16 CUDA 推理对 ReLU² 用 `fast_math=False`,把 F.relu + x*x 两步换成单次 `relu2` kernel 调用;其他模型路径保持 Torch。关闭 fast math 是有意为之:首个 fast-math 实验把 511 个 BF16 subnormal 乘积 flush 到零而被否决,禁用后保留全部 65,280 个有限 BF16 编码(含 signed zero 与 subnormal)。不含任何 CUDA 算术实现或 checkpoint 改动。

## 关键改动

```text
python/sglang/kernels/ops/activation/activation.py: run_unary_activation/relu2 暴露
  fast_math 参数(默认不变)
python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py: _can_enable_t1_fused
  _qk_norm_rope 增加 hidden_size==2048 放行;Cosmos3DenseMLP.forward 在 CUDA contiguous
  BF16 inference 下走 relu2(fast_math=False)
python/sglang/multimodal_gen/test/unit/test_cosmos3.py: gate 新增单测(Edge 放行/并行
  与 compiled 禁用/更大 dense 禁用)
test/registered/kernels/ops/diffusion/test_cosmos3_edge_fusions.py: split-path bitwise
  parity、全部有限 BF16 编码、CUDA-graph replay、grad 回退等测试
test/registered/kernels/benchmark/diffusion/bench_cosmos3_edge_fusions.py: registered
  marker benchmark
```

关键摘录(MLP forward 融合分支):

```python
if (up.is_cuda and up.dtype == torch.bfloat16 and up.is_contiguous()
        and not torch.is_grad_enabled() and not torch.compiler.is_compiling()):
    up = relu2(up, fast_math=False)
else:
    up = F.relu(up)
    up = up * up
```

## 性能声明

1x H200,TP1/SP1,BF16,native eager,quality=lossless,无 torch.compile;`nvidia/Cosmos3-Edge@344d602b`,每 workload 两组 A–B–B–A 取算术均值:T2I(640x640 单帧,35 steps/CFG 7/seed 0)worker E2E 降 32.23%/29.61%(0.9986→0.6768 / 0.9588→0.6749 s),saved-image client E2E 降 30.59%/26.92%;T2V(832x480、81 帧、24fps,35 steps/CFG 5/seed 42)worker E2E 降 2.25%/2.23%,denoise 降约 3.0%(T2V-R2 的 client 14.04% 含 baseline 离群点,作者不计入收益)。峰值显存不变(8.9258 GiB T2I / 13.2598 GiB T2V)。算子级:QK norm+RoPE+KV pack S400/S1024 从 64.94/85.83 us 降到 4.71/8.94 us;ReLU² [1,400,9216] 8.67→4.04 us、[1,8190,9216] 146.16→71.61 us;完整 CFG 迭代 kernel 数 T2I 1996→708、T2V 708→652。

## 验证与回归风险

145 个 Cosmos/kernel 测试 + 46 subtests、85 个 unary activation 兼容测试通过;新增有限域、生产形状、storage-ownership、CUDA-graph replay 与回退测试及 registered benchmark。输出审计 39 条记录:33 个有效 eager 请求逐字节一致(含 lossless/high 检查),视频独立解码 81/81 帧一致(SSIM1,PSNR∞);BCG 请求全部显示 disabled,作者不声明 BCG 加速。回归风险:Hopper split-path 的 bitwise 契约曾误套在未改动的 Blackwell QK 路径导致 3 个 CI 失败,修正后 Hopper lane 注册、其他架构 skip;后续一次 CI 中 H100/B200/BCG job 因无关 CPU fixture 失败被 fast-fail 跳过,作者注明不计为通过验证。grad/torch.compile/非 contiguous 输入回退 Torch 原实现。

## 关联

同方向前序:#34932、#36571(rounded Cosmos T1 与 Hopper Nano 实现)、#38530(lossless fusion 方法)、#31590(Edge 架构);#34618(closed,BCG 路径不可用)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
