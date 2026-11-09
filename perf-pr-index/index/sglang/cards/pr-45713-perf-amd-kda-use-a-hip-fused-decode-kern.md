# PR #45713: [Perf] [AMD] kda: Use a HIP fused decode kernel by default for KDA (Kimi-K3) on gfx950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Monstertail |
| 合并 | 2026-10-26 |
| 分类 | Attention / Kernel(major) |
| Label | documentation, run-ci, jit-kernel |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 含新 HIP kernel(.cu 325 行 + loader)、facade、kimi_k3/kda_backend 接线、kernels 注册与测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/45713 |

## 一句话结论

为 Kimi-K3 KDA decode 新增 HIP 融合 kernel 并在 gfx950 设为默认:kernel 级较 Triton unfused 链最高 3.46x,Kimi-K3 TP8 E2E 吞吐 +3.0-4.1%。

## 动机

MI350X/MI355X(gfx950)上 Kimi-K3 的 KDA decode 默认跑 unfused 链(f_b tiny GEMM、causal Conv1D update、Triton recurrent decode、gated RMSNorm 四段),或经 `SGLANG_ROCM_K3_KDA_FUSED_BACKEND=aiter` 走 AITER FlyDSL 融合。本 PR 为同一融合边界新增 HIP kernel 并设为 gfx950 默认;是 #45710(KDA prefill)的 decode 伴生 PR,两者改动运行时文件不相交。PR 注明 kernel 由 KDAgent(Kernel Design Agents 工作流)在 AMD 生态生成。

## 优化点

每个 (token, head) 一个 256 线程 workgroup(Kimi-K3 TP8 每卡 12 heads × 128 dim),把 f_b projection、depthwise Conv1D+SiLU、q/k L2 norm 与 gate decay、128×128 fp32 state 的 delta-rule update、gated RMSNorm 全部融进单个 kernel。工程细节:front loads 先于 fp32 state 发出以避免 vmcnt 等待覆盖;dpp(permute)实现 warp 内 sum8/sum16 归约;state 与 Conv1D slot stride 从 tensor 读取以兼容 padded hybrid cache 布局;slot 0(mamba pool padding slot)写零且不触碰任何 cache。按 RFC #29630 的 `sglang.kernels` 布局实现于 `kda_kernels`,经 `KernelBackend.KDA` 注册,运行时只经 `sglang.kernels.ops` facade 访问,facade 提供与 AITER 版相同的 fail-closed `available/covered/warmup/run` 接口。

## 关键改动

```text
python/sglang/kernels/kda_kernels/csrc/attention/kda_decode_gfx950.cu: 新 HIP kernel(首次使用时 JIT 编译,PYTORCH_ROCM_ARCH 锁 gfx950)
python/sglang/kernels/ops/attention/kda_fused_decode_gfx950.py: facade,covered() 校验形状/dtype/stride
python/sglang/srt/models/kimi_k3.py: _prepare_fused_decode 按 env 映射 gfx950(新默认)/aiter/其他(保持 unfused)
python/sglang/srt/layers/attention/linear/kda_backend.py: forward_decode 调用层上存储的 facade 而非硬编码 AITER
test/registered/kernels/ops/attention/test_kda_fused_decode_gfx950.py: MI35x 套件,4 步对比 Triton 链、slot<=0 跳过、CUDA graph replay=eager
```

## 性能声明

PR 注明条件:MI355X,`lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261020`。

- kernel 级(69 层 CUDA graph 内单层耗时,11 次取中位数):batch=8 时 Triton 链 27.42us vs HIP 7.93us(3.46x);batch=1 1.93x;batch=256 1.26x;vs FlyDSL 1.17-1.49x。参考:AITER 官方 packed KDA decode(非 SGLang backend)batch≤32 时比 HIP 慢 1.10-1.30x,batch≥64 时快 3-6%。
- 模型级 profile:Kimi-K3 TP8,10,244-token 请求 7 个 decode step × 69 层,原三 kernel(conv1d update、delta rule update、gated norm 各 483 calls)被单个 `kda_decode_gfx950_kernel`(483 calls)替代。
- E2E(bench_serving,两 PR 叠加 vs main 行为,双轮取均值,轮间差 ≤0.6%):1024in/512out conc=16 output throughput 617.7→642.7 tok/s(+4.1%),TPOT -3.6%;conc=64 +3.5%;8192in/64out conc=8 +3.0%(-3.8% TPOT)。bench_one_batch_server:batch 1/8/32/64 steady-state +4.6%/+1.9%/+3.2%/+4.9%。

## 验证与回归风险

精度:对 per-token fp32 PyTorch 参考(batch 1/2/8/37 含 padded rows、batch 16 跑 64 步)输出相对 RMSE ≤7.1e-6(Triton 链 1.2e-5、FlyDSL 2.7e-5);Conv1D state bit-identical,padded rows 返回零,slot 0 不动。Kimi-K3 TP8 GSM8K(8-shot,1319 题):0.960(HIP off)vs 0.961(本 PR + #45710)。新测试注册 MI35x 套件。风险:行为变化——gfx950 上 fused decode 默认开启,`SGLANG_ROCM_K3_KDA_FUSED_BACKEND=aiter` 选 FlyDSL、其他值回 unfused;`covered()` 只覆盖特定形状/dtype/stride,不满足自动回落;JIT 编译在首次使用时发生;cookbook 的 AMD gfx950 toggle 仍只列 `aiter`,文档滞后。

## 关联

伴生前序 PR #45710(KDA prefill HIP kernel,`SGLANG_ROCM_KDA_HIP_PREFILL`);同边界既有 AITER FlyDSL 后端;kernel 生成依据 RFC #29630 的 `sglang.kernels` 布局与 KDAgent 工作流。其余同分类 PR 本卡未收录,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
