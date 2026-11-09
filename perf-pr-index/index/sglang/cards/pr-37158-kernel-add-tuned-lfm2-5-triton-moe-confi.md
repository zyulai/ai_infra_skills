# PR #37158: [Kernel] Add tuned LFM2.5 Triton MoE configs on B300

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-09-01 |
| 分类 | MoE | tier: major |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +N/-N);最终 diff 仅 2 个新增 JSON 配置文件(B300 gate/up 与 down MoE 配置) |
| 链接 | https://github.com/sgl-project/sglang/pull/37158 |

## 一句话结论

LFM2.5-8B-A1B 在 B300 上改用 8 档 tuned Triton MoE 配置替代 KDA 自定义 kernel,混合负载并发 64 吞吐 +10.2%,并删除 951 行自定义 kernel 代码。

## 动机

LFM2.5-8B-A1B 在 B300(`SM103`)上跑两个 BF16 Triton MoE GEMM:gate/up 为 `E=32, N=3584, K=2048, topk=4`,down 为 `E=32, N=2048, K=1792, topk=1`。通用 Triton 路径此前对该 model/device 组合回退到 heuristic schedule;前序工作尝试写 shape-specific 的 KDA kernel 来弥补,但把通用 Triton 路径单独调优后,发现它在中/大 token 数与高并发 serving 下反而更快——问题根源其实是"缺配置"而非"缺 kernel"。因此本 PR 转为 **config-only** 改动,与 #37159 的结论一致。

## 优化点

新增两个 JSON 配置文件(B300 `NVIDIA_B300_SXM6_AC`,Triton 3.7.1):gate/up 与 down 各 8 个 tuned M bucket(1/8/32/256/1024/2048/4096/16384),每档选定 BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps、num_stages,且**全部启用 TMA**。runtime 按 nearest-M 在 bucket 间选择;gate/up 与 down 在每个 bucket 使用相同 BLOCK_SIZE_M(满足 shared expert-row alignment 的要求)。调优过程:在 PR base `771e613d`(KDA dispatch 尚不存在的基线)上跑,100 个确定性 `E=32, topk=4` routing 样本,每个 M 点 240 个 focused configs,覆盖 BLOCK_M 16-256、BLOCK_N 32-256、BLOCK_K 64/128、4/8 warps、3/4/6 stages、group size 1/8/16/32,两条 GEMM 的 TMA 变体均测。最终 diff 无 custom kernel、无 dispatch 分支、无 model-specific runtime code;同时移除 951 行 KDA custom kernel/dispatch/test 代码。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=32,N=1792,device_name=NVIDIA_B300_SXM6_AC.json: gate/up 配置,8 个 M bucket,全 TMA
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=32,N=1792,device_name=NVIDIA_B300_SXM6_AC_down.json: down 配置,8 个 M bucket,全 TMA
```

```text
M=4096 档(gate/up): BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
GROUP_SIZE_M=8, num_warps=8, num_stages=4, USE_TMA=true
```

## 性能声明

- Kernel 级(B300,us/CUDA-Graph replay 的 gate/up + down 对,tuned Triton vs 被移除的 KDA):M=1 -3.11%、M=4 -6.31%、M=8 -4.70%、M=64 -17.18%、M=128 -12.49%、M=1024 -31.47%、M=2048 -27.31%、M=4096 -25.06%、M=16384 -37.07%(负值表示 tuned Triton 更快);KDA 仅在 M=2(+2.29%)与 M=16(+3.23%)小胜,M=32 基本持平。M=4096 时 KDA 的 gate/up 单侧快 3.4%,但其 down 慢得多,组合仍输 25.06%。
- Serving(LFM2.5-8B-A1B,TP1 双服务器并发基准,GPU 分配互换后全矩阵重测,取 GPU0/GPU1 比值的几何平均消除卡间偏差,seeds 1001/1002/1003,decode input 1/output 256,mixed input 1024/output 256,`--random-range-ratio 1.0`,每轮前 flush cache):并发 64 时 decode 吞吐 tuned Triton **+6.36%**(KDA 的 ITL 劣势 +6.29%),mixed 吞吐 **+10.20%**(KDA ITL 劣势 +8.56%);mixed 各并发点 +1.06%~+1.94%;decode 并发 ≤32 基本持平(吞吐差均在 0.8% 内)。

## 验证与回归风险

正确性:tuned generic Triton 与原 KDA 候选在两个 projection、M = 2/4/8/16/64/128/256/512/1024/4096/16384 共 22 个 case 上 `rtol=0, atol=0` 精确一致;60 组 paired serving runs + 2 个并发 16 sanity runs(含 GPU 交叉);两个 JSON 均可解析、含 8 个 M bucket、全 bucket 启用 TMA、gate/up 与 down 的 BLOCK_SIZE_M 对齐;`git diff --check` 通过,最终 diff 仅两个配置文件。风险:配置仅对 B300(`NVIDIA_B300_SXM6_AC`)+ Triton 3.7.1 + 该 (E, N) 形状生效,其他设备或 Triton 版本仍走 heuristic;nearest-M 选择在未调优的 M 值上取最近 bucket;KDA 路径被整体移除,若未来需要 shape-specific kernel 需重新引入。

## 关联

#37159(同结论先例:config-only 替代自定义 kernel)。被移除的替代方案为前序 KDA candidate(PR 正文未给编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
