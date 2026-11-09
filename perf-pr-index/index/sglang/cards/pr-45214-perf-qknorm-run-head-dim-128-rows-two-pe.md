# PR #45214: [Perf] qknorm: run head_dim-128 rows two per warp with 16B lanes, bit-identical

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Monishver11 |
| 合并 | 2026-10-22 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | run-ci, jit-kernel, run-ci-extra |
| Diff | pack 未提供 diffstat（diff 内容可用，行数未注明） |
| 链接 | https://github.com/sgl-project/sglang/pull/45214 |

## 一句话结论

head_dim=128 的 QK-norm kernel 改为每 warp 处理两行、每 lane 搬运 16B，H100 上 kernel 快 1.09-1.45x 且输出逐位一致。

## 动机

`fused_qknorm_warp` 是 Qwen3 dense 与 Qwen3-MoE 在 CUDA 上的默认 QK-norm 路径（经 `apply_qk_norm` 调用）。在 head_dim=128 时它给每个 warp 分配一行 (token, head)，每 lane 每行只搬运 8B，每 SM 最多 16KB in-flight，L2 带宽上限约 2.8-2.9 TB/s。搬运粒度偏小是吞吐瓶颈，值得加宽 lane 载荷。

## 优化点

新增 `fused_qknorm_half_warp` kernel（仅 head_dim=128、编译期选择）：每行由 16 个 lane 处理、每 lane 搬运 16B，即每个 warp 处理两行。位级一致的实现方式是：每个 lane 保存原 warp kernel 中相邻两个 lane 的两个部分和（按相同顺序累加），再用 `warp::reduce_sum<16>` 重放原 kernel 的 xor butterfly 归约（偏移 16..2 变为 8..1，最后的 xor-1 步改为 part0+part1），rsqrt 与逐元素输出表达式不变。launcher 检查 q/k 指针、权重指针与行 stride 的 16B 对齐，不满足则回退到原 one-row-per-warp kernel，保证旧输入都能跑；CTA 路径（head_dim>256）与 ROCm 保持不变。

## 关键改动

```text
python/sglang/kernels/jit/csrc/elementwise/qknorm.cuh: 新增 fused_qknorm_half_warp kernel 与 16B 对齐检查的 dispatch 逻辑
python/sglang/kernels/ops/layernorm/norm.py: _jit_qknorm_module 增加 half_warp 参数，False 时构建原 kernel 供测试对照
test/registered/kernels/ops/layernorm/test_qknorm.py: 新增 test_two_rows_per_warp_match_one_row_per_warp 逐字节对比测试
```

## 性能声明

H100 NVL (PCIe, 400W)，CUDA 13.0，BF16，q/k 为 Qwen3 布局的 qkv 行切片，`do_bench_cudagraph` 中位数：32 tokens 起 Qwen3-8B (32+8 heads) 1.12-1.38x，Qwen3-30B-A3B (32+4 heads) 1.04-1.45x；8192 tokens 以上 1.09-1.10x。BF16+FP16 全量 geomean 1.14x（≤256 tokens）/ 1.17x（≥512 tokens）。≥8192 tokens 时 q/k 超出 L2，kernel 带宽从 2.8-2.9 TB/s 提升到 3.1-3.3 TB/s。模型内：Qwen3-8B TP1、32 请求 x 4096 input tokens，QK-norm kernel 720 次launch 合计 43.82 -> 34.09 ms（1.29x），占 GPU 时间 0.94% -> 0.73%；serving 吞吐持平（26597 -> 26567 tok/s 等），因该 kernel 不足 GPU 时间 1%。

## 验证与回归风险

新增逐字节对比测试（BF16/FP16、5 q heads 使 q/k 边界落在行对内、奇数行数、4099 tokens 触发 persistent grid 循环、pad=4 触发对齐回退）。作者报告 768/768 cases 位级一致，`test_qknorm.py` + `test_rope.py` 1827 passed；寄存器 28 -> 40/thread，无 spill。风险点：仅 CUDA 且仅 head_dim=128 生效；16B 对齐检查若误判会静默回退到慢路径（功能不变）；与 #43425 在 qknorm.cuh 上冲突，后合并方需 rebase。

## 关联

同一 two-rows-per-warp scheme 的前序 PR：#43337、#43508（只改 diffusion 的 qknorm_rope.cuh）；相关在途 PR #43425（Blackwell Qwen3-VL MRoPE 融合，保留 8B lane，文件冲突）；#39071、#40585 仅动该文件的 ROCm include 行。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
