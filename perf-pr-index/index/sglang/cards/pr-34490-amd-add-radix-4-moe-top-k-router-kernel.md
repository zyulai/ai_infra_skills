# PR #34490: [AMD] Add Radix-4 MoE top-k router kernel for Kimi-K3 routing

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @RolaoDenthu（co-author @kkHuang-amd） |
| 合并 | 2026-08-23 |
| 分类 | MoE (tier: major) |
| Label | amd, run-ci, jit-kernel |
| Diff | pack 未提供 diffstat（diff 内容可用：5 个文件，另 1 个文件超长未入 pack） |
| 链接 | https://github.com/sgl-project/sglang/pull/34490 |

## 一句话结论

Kimi-K3 的 896-expert top-16 路由在 MI355X 上用 radix-4 select kernel 替代 aiter 逐轮 topk，decode M<=1024 时 kernel 快 1.17-1.93x，e2e TTT 改善 1-3%。

## 动机

Kimi-K3 从 896 个 expert 中不分组地选 16 个，落到 aiter 的通用 `biased_grouped_topk`——每选一个 expert 花一轮，成本随 `topk` 增长，MI355X 上每层约 10.4us。aiter 更快的 pivot-based 路径只对 DeepSeek 精确形状（256 experts、8 groups、top-8）开放，K3 永远够不到。需要为 K3 形状写一个轮数只随 key 宽度（而非 topk）增长的专用 kernel。

## 优化点

Radix-4 select：每轮固定 pivot 的 4 个 key bit——把存活 key 计数进 16-bin 直方图（每线程把 15 个 expert 的 bin 计数打包进一个 64-bit 寄存器的 nibble），从最高 bin 向下累加，第 k 大 key 所在 bin 即确定这 4 个 bit；轮数只随 key 宽度变化。CDNA 特化：bin 总数经 LDS 读出后"每 bin 一 lane"，用 4 步 DPP 前缀和 + ballot 替代 16 步逐 bin 遍历；sigmoid 用 aiter 的 exp2f+rcpf 近似序列位级对齐；`kAiterTieLaneRank` 表复制 aiter wave64 遍历的 tie-break 顺序并测试钉死，使 prefill（大 M 走 aiter）与 decode（小 M 走本 kernel）对相同 token 产出逐列一致的专家路由，batch size 不改变路由结果。`covered()` 限定 [M,896] 行连续、top-16、ungrouped、M<=1024（性能门槛：超过 ~1.5k tokens 网格饱和反而亏）；默认关，`SGLANG_K3_RADIX4_TOPK=1` 启用，不满足即回退 aiter。

## 关键改动

```text
python/sglang/kernels/jit/csrc/moe/route_radix4_hip.cuh: CDNA-only radix-4 select kernel（DPP/wave64 ballot，527 行）
python/sglang/kernels/ops/moe/moe_route_radix4.py: JIT 封装 + supported_hardware/available/covered 门控
python/sglang/srt/layers/moe/topk.py: biased_grouped_topk_gpu 的 aiter 分支前插入 radix4 dispatch
python/sglang/srt/environ.py: 新增 SGLANG_K3_RADIX4_TOPK（默认 False）
test/registered/kernels/ops/moe/test_moe_route_radix4.py: oracle + 真实 aiter 双参照测试
```

## 性能声明

MI355X，896 experts top-16，graph-captured kernel 时间（aiter vs radix4）：M=1 10.24->5.99us（1.71x）；M=8 1.84x；M=16 10.64->5.51us（1.93x）；M=64 1.81x；M=256 1.64x；M=512 1.56x；M=1024 11.09->9.44us（1.17x）；M=1536 12.08->11.95us（1.01x，超过 break-even）。e2e：TTT 改善 1-3%（未注明模型配置细节）。GSM8K 1319 题 0.952。

## 验证与回归风险

测试双参照：纯 torch fp32 oracle 定义契约（权重无 bias、NaN 永远排最后、renorm 除数非正时守卫为 1、 winners 按序输出），aiter 作为 tie 顺序的钉死参照（`test_route_radix4_tie_order_is_aiters` 等），覆盖 tie 全同、NaN、极值/饱和行、可复现性（16 次逐位相等）、graph replay、非连续 stride（pad 37 列）。风险：8-bit nibble 打包假设每 bin 计数 <=15（CHUNK=15 分块保证）；tie 顺序表是从 aiter 行为反读的，aiter 升级改变 wave64 遍历会以测试失败暴露；`available()` 故意吞掉编译失败回退 aiter（serving 优先），测试侧用 `build()` 让编译错误可见；仅 gfx942/gfx950。

## 关联

CUDA 侧同思路 kernel 为 `route_radix.cuh`（本 PR 的 tie contract 一节以其按 expert id 平局为对照）；同区域 K3 相关：`SGLANG_K3_FUSED_FRONT`（moe_front.py，默认开）、Kimi-K3 ViT CUDA graph 相关 env。PR 正文未列其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
