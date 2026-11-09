# PR #34619: [Diffusion][GLM-Image] Retune QK head LayerNorm for SM103

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-08-13 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci, jit-kernel, run-ci-extra |
| Diff | 素材未注明 diffstat(diff 片段可见 1 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/34619 |

## 一句话结论

将 B300/SM103 上 fused Q/K head LayerNorm 的启动配置从每 program 16 行调到 32 行,生产 shape 微基准从 213.074 us 降到 45.035 us。

## 动机

GLM-Image 生产 shape 下的 fused Q/K head LayerNorm 在 SM103 上沿用的 `ROWS=16, num_warps=1` 启动配置并非最优。Nsight Compute 显示该内核 82.37% memory throughput、仅 28.95% compute throughput、11.45% achieved occupancy,属于带宽/launch-shape 受限而非计算受限,因此针对 B300 重新扫参以减少 program 数、增大每 program 行数。

## 优化点

改动只有一处启动配置:`_qk_head_launch_config()` 在 SM103(major==10, minor==3)分支返回 `ROWS=32, num_warps=1`(原为 16,1)。B300 扫参选定该组合;SM120(RTX 5090)的 `8,4` 与其余架构(含 Hopper)的 `64,2` 独立调优结果保持不变。内核本身逻辑未动,生产 GLM-Image shape 在两种启动配置下输出 bit-exact。

## 关键改动

```text
python/sglang/kernels/ops/diffusion/triton/layernorm_modulate.py: SM103 分支启动配置 16 行/program → 32 行/program(num_warps=1)
```

```python
def _qk_head_launch_config() -> tuple[int, int]:
    arch = get_jit_cuda_arch()
    if arch.major == 10 and arch.minor == 3:
        return 32, 1   # 原 (16, 1)
    if arch.major * 10 + arch.minor >= 120:
        return 8, 4
    return 64, 2
```

## 性能声明

- 微基准(生产 GLM-Image shape):现有 SM103 配置(`ROWS=16`)213.074 us;候选(`ROWS=32`)45.035 us。
- E2E(GLM-Image,1024px,30 steps,native backend,commit `9deb6952`):eager + BCG denoise 3.043391 s / e2e 20.58473 s;torch.compile denoise 3.049033 s / e2e 21.94464 s,BCG 在 denoise 上快 0.18%;retune 后纯 eager denoise 3.06985 s。
- Nsight Compute(选定内核):82.37% memory throughput,28.95% compute throughput,180 registers/thread,11.45% achieved occupancy。
- B300 bit-exact QK head LayerNorm 测试通过;PR 未注明 E2E 相对 retune 前的加速倍数。

## 验证与回归风险

验证手段:B300 bit-exact 一致性测试(两种启动配置下生产 shape 输出相同)与 pre-commit;PR 未新增单元测试文件。风险面极小但平台特定:仅改变 SM103 的 launch config,若 B300 之外存在 SM103 变体或驱动差异,扫描结论未必迁移;SM120/Hopper 及其他架构路径完全未动。性能基于 commit `9deb6952afa483e38f96385a375b96f463da5303` 测得,作者注明已 rebase 到最新 main 并复测。pack 内 CI 状态显示 PR Test (Base) 为失败态。

## 关联

同文件/同区域:diffusion Triton LayerNorm 系列内核;SM120(`8,4`)与 Hopper(`64,2`)配置为各自独立调优的既有选择(素材未给出对应 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
