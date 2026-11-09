# PR #28666: [AMD] Fuse shared_expert_gate GEMV into the MoE append kernel (HIP/aiter)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yichiche |
| 合并 | 2026-08-14 |
| 分类 | MoE(classify.py 给出,tier: major) |
| Label | amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 显示 4 个文件(含 2 个新增测试) |
| 链接 | https://github.com/sgl-project/sglang/pull/28666 |

## 一句话结论

在 AITER shared-expert-fusion 路径上,把 shared_expert_gate 的独立 GEMV + sigmoid + scale 折进 MoE append kernel,每 MoE 层 decode 时间约 12.9µs 降到 5.2µs。

## 动机

AITER shared-expert-fusion 路径上,计算 fused shared-expert 路由权重会启动一个独立 GEMV:`self.shared_expert_gate(hidden_states)`,即 `[M, hidden] × [hidden, 1]` 矩阵向量乘(`Cijk_…MT1x2x512…` kernel,decode 下约 8.9 µs),其唯一输出再喂给后续的 `_fused_append_shared_experts_with_weights_kernel`。decode batch size 下该 GEMV 纯属 kernel-launch 开销(≈0 TFLOPs,launch-bound 而非 compute-bound,单独调优无效),是折入 append kernel 的理想候选。

## 优化点

- `_fused_append_shared_experts_with_weights_kernel` 增加 `hidden_ptr`/`wgate_ptr`/`scale` 参数与 `FUSE_GATE`/`HIDDEN`/`BLOCK_H` constexpr:开启 `FUSE_GATE` 时,每个 program(对应一个 token)在 fp32 中归约 `hidden[pid,:] · W_gate[:]`,把 `sigmoid(logit) * scale` 广播到所有 shared slot,替换原来的 `shared_weights_ptr` 加载。host 侧 wrapper 新增 `fuse_gate=False, hidden_states=None, gate_weight=None, scale=1.0` 参数,`fuse_gate` 与 `apply_sigmoid` 互斥(前者已内含 sigmoid);`num_warps` 在 fuse_gate 时升到 8。
- `qwen2_moe.py`:新增 `_shared_expert_scale()`(非 DeepEP 的多 EP 时 1/ep_size 预缩放,镜像 `_get_shared_expert_weights` 的 allreduce-EP 逻辑);`_append_shared_to_topk_output` 在 `_use_aiter` 时以 `fuse_gate=True` 调 kernel(不再单独启动 gate GEMM、不再调用 `_get_shared_expert_weights`),非 aiter(CUDA)路径保持原样。
- 默认 `fuse_gate=False` 路径 byte-for-byte 不变;CUDA 路径完全不变。

## 关键改动

```text
python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py: append kernel 支持 FUSE_GATE 的 GEMV+sigmoid+scale
python/sglang/srt/models/qwen2_moe.py: aiter 路径改走 fuse_gate=True,CUDA 路径不变
test/registered/moe/test_fused_append_shared_experts.py: 新增,三路径对 eager 参考的等价性测试
test/registered/amd/accuracy/mi35x/test_qwen35_mxfp4_eval_mi35x.py: 新增,MI35x nightly 精度+性能测试
```

```python
if FUSE_GATE:
    h = tl.load(hidden_ptr + pid * HIDDEN + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    w = tl.load(wgate_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    logit = tl.sum(h * w)
    shared_val = tl.sigmoid(logit) * scale
    shared_ws = tl.zeros((BLOCK_S,), dtype=tl.float32) + shared_val
```

## 性能声明

torch profiler decode trace(同一配置):`Cijk_…MT1x2x512…`(shared-gate GEMV)从约 8.9 µs 独立启动降为 0 次出现(消除);`_fused_append_shared_experts_with_weights_kernel` 约 4.0 µs → 约 5.2 µs(现在承担 GEMV+sigmoid+append)。每 MoE 层合计 12.9 µs → 约 5.2 µs,约省 7.5 µs。未给出端到端 E2E 吞吐对比数字。精度:GSM8K(200 题,parallel 2000,greedy),Qwen3.5-397B-A17B-MXFP4,tp=2,AITER backend,MI35x——baseline 0.628 vs gate-fused 0.603,作者判定在 run-to-run 方差内(greedy + 连续批处理非确定性,σ≈0.045);离线 kernel 测试中 kernel 内 fp32 GEMV 与 `F.linear → sigmoid → scale` fp32 参考完全一致(max|diff|=0.0),routed weights bit-identical。

## 验证与回归风险

新增单元测试 `test_fused_append_shared_experts.py`(同时覆盖前序 PR #28658 的 apply_sigmoid 与本 PR 的 fuse_gate):legacy 路径必须逐位不变、apply_sigmoid/fuse_gate 必须等于各自 eager 参考(fp32 紧容差,bf16 较松,因 tl.sigmoid 与 torch.sigmoid 存在 ULP 差异)、fuse_gate 与 apply_sigmoid 互斥 assert、参数缺失 assert、零 shared 为 no-op;注册 CUDA 与 AMD CI。新增 MI35x nightly 测试覆盖两个 checkpoint:shared expert 不量化(融合关闭,参考精度,gate >0.91)与 shared expert 量化(fuse_gate 路径 e2e run-through,无精度 gate)。CI 快照 Base 通过、Extra 失败。风险:fuse_gate 路径仅覆盖 qwen2_moe 的 `_use_aiter` 分支且要求 `shared_expert_gate` 存在;fp32 归约顺序与 torch matmul 不同,输出与旧路径仅在容差内一致(非 bitwise)。

## 关联

前序:PR #28658(sigmoid + bf16→fp32 cast 折入 append kernel 的 apply_sigmoid 路径),本 PR 的 fuse_gate 是其延伸;测试文件 docstring 明确引用两者。同区域:AITER/AMD MoE 融合路径。PR 正文未列出其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
