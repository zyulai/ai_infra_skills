# PR #54706: [ROCm][RDNA3] Fix W4A16 split-K accuracy and determinism

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54706](https://github.com/vllm-project/vllm/pull/54706) |
| 作者 | AIwork4me |
| 合并日期 | 2026-10-03 |
| labels | rocm, ready |
| 热度 | 👍 0 · 💬 114 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
RDNA3 W4A16 GPTQ 的 split-K 部分和以 bf16/fp16 CAS 原子累加,顺序不定致误差与不可复现;改为 FP32 定序归约后一次 cast。

## 问题现象
gfx11(RDNA3)W4A16 GPTQ kernel 的数值误差存在于**每一次**受影响的前向传播、所有解码配置中:先窄化再低精度累加不满足结合律,先抢到 CAS 的 split block 决定结果。Greedy 解码只是让它可见——temperature 0 下 logit 的微小摆动会让相同输入的重复生成翻转 argmax(见 #50603 的轨迹);采样(如 T=0.6)下同一误差被采样噪声掩盖,但仍存在于每个输出指标之下。修前 Muse q_proj 在 M=1 下 100 次调用最多 200/200 个不同结果。

## 根因
正文披露:`FP32 per-block partial -> bf16/fp16 窄化(8/11 位尾数) -> CAS 原子累加,顺序随执行而变`。gfx11 没有 packed bf16/fp16 原子加,epilogue 用 CAS 重试循环模拟——这同时使累加顺序依赖调度器与竞争状况,既引入舍入误差又破坏可复现性。

## 修复方案
计算 kernel 与 dispatch 不变,仅替换两条路径的 epilogue:`FP32 per-split partials -> 固定升序-z FP32 归约 -> 一次最终 bf16/fp16 cast`。每个输出元素只写一次;`k_split == 1`(WMMA)与 `z_count == 1`(scalar)直接 store,无 scratch、无归约、无原子。FP32 partials scratch 用 `at::empty`(覆盖性由构造保证,七个 WMMA 变体与 scalar kernel 中每个 `(z, m, n)` 槽恰有一个写者),per-call 经 caching allocator 分配、按行分块使上界与调用方 M 无关。后续更新(2026-10-01):fp16 dequant 改为精确 factored 形式(fp32 y/z 对,z 修正每 32-K 轮一次),消除旧 bit-trick 把 `z = scale * (-1024 - zero)` 烘进 fp16 常数的偏差;WMMA dispatch 阈值统一为 M ≥ 12;fp16 scalar 路径把 split-K 归约熔进 GEMM kernel(per-call `at::zeros` ticket 数组,固定升序求和);MoE kernel 同样获得精确 dequant,但 MoE split-K 确定性留作后续;host 端强制 `groupsize % 32 == 0`(此前该类配置静默出错)。

## 影响与回归风险
- 影响窗口:gfx11 / RDNA3 W4A16 only,正文未列具体版本范围。
- 回归面:`csrc/rocm/moe_q_gemm_rdna3.cu` 等dense/MoE kernel 与 `qdq_4_rdna3.cuh` 的 epilogue 与 dequant 逻辑、WMMA dispatch 阈值(M≥12)、新增 per-call scratch/ticket 分配路径;gfx942 非 RDNA3 为 stub,需编译守卫通过。legacy CAS epilogue 保留作 A/B 对照但 dispatch 不会选它。性能上 W7900 实测 prefill 快 5–9%,M=1 decode 相对 legacy 控制组付 ~4–10% 残余代价(scratch 往返)。

## 关联
- #50603(调查轨迹)
- #53856(修 Muse eager/8192 残余发散,归因于 ROCm paged attention V-cache tail)
- #54809(gptq act-order/g_idx 移除,rebase 适配)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
