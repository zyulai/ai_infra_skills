# PR #61718: [Bugfix] Fix NaN output of the Mamba2 chunk scan kernel on sm90

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61718](https://github.com/vllm-project/vllm/pull/61718) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-10-17 |
| labels | bug |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
sm90 上 Mamba2 chunk scan kernel 部分autotune配置被 Triton 误编译为异步 wgmma 寄存器竞争,有限输入输出 NaN;按规则在 sm90 剔除 11 个配置。

## 问题现象
Hopper(sm90)上,`_chunk_scan_fwd_kernel` 的 23 个 autotune 配置中有 8 个会对有限输入把 NaN/Inf 写进整行输出(reproducer 实测:64/64/64 4w2s 在 22–31/100 次调用出现 NaN,7 个 4w4s 配置为 53–100/100);autotuner 只按速度选配置,可能选中其一。issue #61603 的端到端现象是 NemotronH 服务返回空响应、`not JSON compliant: nan`(作者未跑 e2e,只做了 kernel reproducer 与单测)。修复后 0/12 个配置出现 NaN,且 H20-3e 上旧列表最快配置被保留、无速度损失。

## 根因
正文详细披露:K 循环中 `acc += tl.dot(cb, x)` 的 `cb` 在寄存器中构造(decay exp、dt、因果 mask、cast bf16)。当整 warpgroup(`num_warps % 4 == 0`)且 `BLOCK_SIZE_M % 64 == 0` 时,Triton 把该 dot lower 为 A 操作数在寄存器的 wgmma;`num_stages > 1` 时 Triton 的 `TritonGPUPipeline` pass 把 dot 拆成异步 wgmma 并以 `pendings = 1` 等待——循环回边时上一条 wgmma 仍在运行,下一迭代先把新 `cb` tile 覆写进同一批寄存器。PTX ISA 不允许在 `wgmma.wait_group` 之前修改 `wgmma.mma_async` 的 A 寄存器,运行中的 wgmma 把 fp32 原始位当 bf16 读,得到 NaN/Inf 及接近 3.4e38 的值,写入整行输出。竞争依赖时序,故 NaN 率随配置与运行变化。bug 本身在 Triton pipeliner(vLLM 无法修改),vLLM 侧的问题是 autotune 列表提供了会被这样编译的配置。

## 修复方案
`vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py` 新增 `_drop_racy_wgmma_configs` 包住配置列表:

```diff
+def _drop_racy_wgmma_configs(configs: list[triton.Config]) -> list[triton.Config]:
+    if not (current_platform.is_cuda() and current_platform.is_device_capability(90)):
+        return configs
+    return [
+        c for c in configs
+        if c.num_stages == 1
+        or c.num_warps % 4 != 0
+        or c.kwargs["BLOCK_SIZE_M"] % 64 != 0
+    ]
```

- 仅 CUDA sm90 生效(须 `is_cuda()`,因 ROCm gfx90a 也报 capability (9,0)),剔除 11/23 个配置;其他平台保留全部 23 个;
- 采用规则而非失败配置清单(配置 2/10/12 编译出同样的 racy 代码但在 reproducer shape 下 NaN 率低;TTGIR 检查 23 个配置在 dstate 128/256 下规则与 racy 模式 0 不匹配);
- 在列表构建时过滤而非 `early_config_prune`(单配置时 Triton 跳过 prune,`VLLM_TRITON_FORCE_FIRST_CONFIG` 不走 prune);
- 新增 `test_chunk_scan_fwd_every_config_matches` 用 monkeypatch 逐配置跑真实 kernel(旧代码与 #61664 树上均失败);其余 Mamba2 SSD kernel 经 TTGIR 扫描无此模式,不改。

## 影响与回归风险
- 回归面:仅 sm90 CUDA 上 Mamba2 SSD `_chunk_scan_fwd_kernel` 的可用 autotune 配置集合(动态形状下可能选中稍慢的合法配置,正文实测最快配置保留);Triton 3.6.0/3.7.1 的 workaround,若 Triton 修复 pipeliner 该过滤为多余但无害;新测试未验证非 sm90 平台保留完整列表;端到端 NemotronH 服务场景未复测。

## 关联
- https://github.com/vllm-project/vllm/issues/61603 (Fixes)
- https://github.com/vllm-project/vllm/pull/61664 (此前的部分修复,本 PR 补齐其遗漏配置并加真实 kernel 测试)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
