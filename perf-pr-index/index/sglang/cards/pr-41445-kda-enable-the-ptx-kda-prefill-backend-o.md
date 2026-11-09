# PR #41445: [KDA] Enable the ptx_kda prefill backend on SM100 (B200 / GB200)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-09-30 |
| 分类 | Model-specific Fast Path (major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 含 6 个文件:按架构 JIT 构建、后端门控与测试(kernel 代码本身不变) |
| 链接 | https://github.com/sgl-project/sglang/pull/41445 |

## 一句话结论

把原来只在 SM103(B300/GB300)可用的 ptx_kda 线性注意力 prefill kernel 扩展到 SM100(B200/GB200),对 Triton 快 1.27-3.10x。

## 动机

`--linear-attn-prefill-backend ptx_kda` 此前只跑在 SM103:扩展以 `-gencode arch=compute_103a,code=sm_103a` 构建,`PtxKDAKernel.supports_prefill` 检查 compute capability (10,3),B200/GB200 上会静默回退 Triton。但 kernel 没有任何 SM103 独有依赖:全部 inline PTX(tcgen05 `cta_group::1` 的 alloc/ld/st/mma f16+tf32/commit/wait/fence、TMA、mbarrier、mma.sync、ldmatrix、cp.async、prefetch)在 sm_100a 上都可用,没有 `ld.red` 等 sm_103-only 指令,route/piece-budget 代价模型运行时读 `multiProcessorCount`。B200 上应能获得与 B300 相同的加速。

## 优化点

不改 kernel 代码,只改构建与门控:`kda_ptx_prefill` 新增 `SM_ARCHS = {(10,0): "100a", (10,3): "103a"}` 映射,按运行设备的 compute capability 选择 `-gencode` 目标(JIT 构建约 1-2 分钟,缓存于 `TORCH_EXTENSIONS_DIR`);扩展名按架构区分(`kda_prefill_ptx_sm100a` / `kda_prefill_ptx_sm103a`),使共享 extensions 目录可同时保留两份 cubin(arch 特定 cubin 不能跨架构运行)。`supports_prefill` 改为 capability 在 SM_ARCHS 内即真;`kda_backend.py` 同步更新 fallback 与错误信息文案。默认 KDA prefill backend 仍是 `triton`,`ptx_kda` 保持 opt-in。测试侧:既有两个 ptx 用例改在 SM100 上也运行,B200 runner 因需 JIT 构建把 est_time 从 180s 提到 300s(B200 实测 146s 含构建)。

## 关键改动

```text
python/sglang/kernels/ops/attention/linear/kda_ptx_prefill/__init__.py: SM_ARCHS 映射 + 按设备架构构建、按架构命名的 JIT 扩展
python/sglang/srt/layers/attention/linear/kernels/kda_ptx.py: supports_prefill 扩到 SM100 与 SM103
python/sglang/srt/layers/attention/linear/kda_backend.py: fallback/错误信息更新
python/sglang/kernels/jit/csrc/attention/kda_prefill.cu: 仅构建注释更新,kernel 代码不变
test/registered/kernels/ops/attention/test_kda_prefill.py: ptx 用例扩展到 SM100
test/registered/e2e/models/test_glm53_flash_kda_ptx_prefill_blackwell.py: 新增 GLM-5.3-Flash E2E parity 测试
```

构建目标选择:

```python
SM_ARCHS = {(10, 0): "100a", (10, 3): "103a"}
capability = torch.cuda.get_device_capability()
arch = SM_ARCHS[capability]
name=f"{_EXT_NAME}_sm{arch}",
extra_cuda_cflags=[..., f"arch=compute_{arch},code=sm_{arch}"],
```

## 性能声明

测试条件:B200(148 SMs),torch 2.13 + cu130,triton 3.7.1;`PtxKDAKernel.extend` vs `TritonKDAKernel.extend`(按 `KDAAttnBackend.extend` 的调用方式),q/k/v/raw gate 为 bf16 `[1, B·T, H, 128]`,beta 为 raw(kernel 内做 sigmoid),state pool fp32,CUDA events 计时(含 wrapper 开销)、10 次取中位数,单位 µs。

- 带 `lower_bound = -5` 安全 gate(如 Kimi-K3):对 Triton 加速 1.27x-**3.10x**;如 1×32768×4:1583→510 µs(3.10x)、1×32768×8:1835→766 µs(2.40x)、1×32768×32:3442→2498 µs(1.38x)、16×2048×32:1.49x;max diff 3e-2。同脚本 B300 为 1.09-2.98x;两卡上 ptx_kda 绝对耗时差 ±6% 内。
- 不带 lower_bound(plain softplus gate,如 Kimi-Linear):ptx_kda 快 1.34-**3.21x**,但一个 shape 产生 NaN(见 Notes)。
- 收益最大的区域是每 GPU head 数少(TP shard)+ 长 prompt:Triton `chunk_h` 递推是 latency-bound,每 64-token chunk 约 2.3-2.4 µs 且与 H 无关;ptx_kda 把序列拆 piece 并组合其 affine state map。

## 验证与回归风险

kernel 级:SM100 上 `test_kda_prefill.py` 3 passed(146.12s 含 JIT 构建),ptx 用例同时比对输出与最终 state(与 Triton 对照),raw-beta 用例覆盖 wrapper 的 padding 路径。新增 E2E 测试 `test_glm53_flash_kda_ptx_prefill_blackwell.py`(nightly,4-GPU B200/GB300,est 900s):GLM-5.3-Flash 34 个 KDA 层用安全 gate;通过读 server 日志证明 dispatcher 选中 `PtxKDAKernel` 且无任何 batch 回退 Triton,并以 GSM8K 500 题 20-shot(ptx 0.930 / triton 0.930,gap 0.000,上限 0.03)与 2040-token prompt logprob 探针(mean |dlogprob| 0.080,>1nat 比例 0.2%)做 parity。CI 状态:Base/AMD 排队中,Extra 失败。风险面:不带 lower_bound 的 softplus gate 会 NaN(该 gate 模式 kernel 不安全);wrapper 对 spec draft-extend、track-state 快照、不支持 shape 会静默回退 Triton;非 SM100/SM103 架构显式报错。

## 关联

同区域:KDA 线性注意力 backend 系列(triton/helion/flashkda/cutedsl/nvidia_kda/ptx_kda 多后端,pack 未列编号);GLM-5.3-Flash(34 KDA 层)为 E2E 目标模型。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
