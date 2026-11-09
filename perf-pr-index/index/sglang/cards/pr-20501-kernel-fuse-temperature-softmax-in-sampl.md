# PR #20501: [Kernel] Fuse temperature + softmax in sampling for decode speedup

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Godmook |
| 合并 | 2026-04-02 |
| 分类 | Attention / Kernel(tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/20501 |

## 一句话结论
把采样路径的 temperature 缩放与 softmax 融合为单个 Triton kernel,microbenchmark 下最高提速 4.46x。

## 动机
标准采样路径在 `sampler.py` 中把 temperature 缩放与 softmax 作为两个独立 CUDA kernel 执行(`logits.div_(temperatures)` + `torch.softmax`),每个 decode step 都会调用,带来 2 次 kernel launch(每次约 10us 开销)和对整个 `[batch_size, vocab_size]` 张量的 6 次全局内存读写(4 读 + 2 写)。对 Llama 3 128K、Qwen 152K 这类大词表模型,这成为 decode 延迟预算中的可感知瓶颈。

## 优化点
新增 `fused_sampling.py`,用 Triton kernel 把 `div_(temperature) + softmax` 融合,分两个变体:词表 ≤ 32768 时用 single-pass kernel,整行数据驻留寄存器,仅 1 读 + 1 写共 2 次内存访问;大词表用 multi-pass softmax(2 读 + 1 写),并按 (BLOCK_SIZE, num_warps, num_stages) autotune。temperature 除法在 load 阶段顺带完成,省去独立除法 pass。`sampler.py` 在标准路径按 batch size 阈值 128 分派:大 batch 走融合 kernel,小 batch 保留 PyTorch 原实现(小 batch 下 launch 开销主导)。启动时通过 `warmup_fused_temperature_softmax` 预编译并 autotune,out-of-place kernel 的最优配置复用给 in-place kernel,避免对线上 logits 缓冲执行 autotune。

## 关键改动

```text
python/sglang/srt/layers/fused_sampling.py: 新增融合 Triton kernel(single-pass/multi-pass、in-place/out-of-place、warmup API)
python/sglang/srt/layers/sampler.py: 标准采样路径按 batch>=128 分派到融合 kernel,保留原路径回退
python/sglang/srt/model_executor/model_runner.py: kernel_warmup 挂接 _warmup_fused_sampling(HIP 跳过)
test/registered/sampling/test_fused_temperature_softmax.py: 14 项正确性测试
benchmark/kernels/bench_fused_temperature_softmax.py: microbenchmark(含 flashinfer 对比)
```

## 性能声明
microbenchmark 条件:NVIDIA A40(SM 8.6)、bf16、`torch.cuda.Event` 计时、200 iters / 50 warmup。原文数字:
- bs=1, vocab=32000:25.5us → 23.4us(1.09x)
- bs=1, vocab=128256:43.0us → 33.8us(1.27x)
- bs=32, vocab=128256:150.5us → 46.7us(3.22x)
- bs=128, vocab=32000:140.9us → 34.7us(4.06x)
- bs=512, vocab=32000:541.7us → 121.4us(4.46x)
- bs=512, vocab=128256:2416.8us → 694.8us(3.48x)

全部配置 1.09x–4.46x。注意这是 kernel 级 microbenchmark,非端到端 decode 数据。

## 验证与回归风险
新增 14 项正确性测试(注册到 stage-b-test-small-1-gpu CUDA CI,est 15s),覆盖 batch 1–512、词表 1024–128256、温度 0.01–100、bf16/fp16/fp32、空 batch、argmax 不变性、大 logits 无 NaN/Inf,并与 flashinfer softmax 对比;bf16 容差 atol=1e-4/rtol=1e-3。风险:融合路径只作用于 batch ≥ 128 的标准采样分支,小 batch 与非 CUDA 平台回退原实现;HIP 上 warmup 直接跳过;multi-pass in-place 依赖 warmup 阶段写出的 autotune 配置,`best_config` 缺失时回落默认配置。

## 关联
无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
