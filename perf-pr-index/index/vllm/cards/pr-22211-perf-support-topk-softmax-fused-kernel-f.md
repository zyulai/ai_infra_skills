# PR #22211: [Perf] Support topk softmax fused kernel for broader num_experts

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shixianc |
| 合并 | 2025-08-13 |
| 分类 | MoE（tier: major） |
| Label | ready |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失） |
| 链接 | https://github.com/vllm-project/vllm/pull/22211 |

## 一句话结论

MoE topk+softmax 融合 kernel 从仅支持 2 的幂扩展到 64 的倍数（192/320/384/448/576），非 2 幂专家数延迟降 33%–86%。

## 动机

`topkGatingSoftmaxKernelLauncher` 此前要求 num_experts 为 2 的幂，否则回退到先 softmax、再 topK 的两 kernel 方案，对专家数不是 2 的幂的 MoE 模型（如 192、576 experts）相当慢。强制 2 的幂的原因是保证每个 warp 在固定 `MAX_BYTES_PER_LDG=16` 下恰好处理整数行；作者发现把单次加载降到 8 字节即可支持 64 的倍数的专家数，且基准显示相比两 kernel 方案提速显著。另为未来的 sparser 模型补上 512（2 的幂）档。

## 优化点

- 把 `MAX_BYTES_PER_LDG` 从 launcher 内部常量提升为模板参数：2 的幂专家数仍用 16 字节加载（`BYTES_PER_LDG_POWER_OF_2`），非 2 幂的 64 倍数用 8 字节（`BYTES_PER_LDG_MULTIPLE_64`），使每个 warp 的 VPT 划分仍保持整数行，从而走同一融合 kernel（softmax+max+argmax 单 kernel）。
- dispatcher 新增 case：CUDA-only 的 192/320/384/448/576（`#ifndef USE_ROCM`——ROCm 的 WARP_SIZE=64 下 8 字节加载对部分专家数放不下，注释建议未来测 4 字节加载再开启），以及新增 2 的幂档 512。
- 移除 `VPT`/`NUM_EXPERTS` 必须为 2 的幂的 `static_assert`；LAUNCH_SOFTMAX 宏改为编译期断言 warp size 并传 `MAX_BYTES`；不匹配的 num_experts 仍回退两 kernel 路径，但错误信息更新为"not a power of 2 or multiple of 64"。

## 关键改动

```text
csrc/moe/topk_softmax_kernels.cu: MAX_BYTES_PER_LDG 模板化 + 新增 192/320/384/448/576/512 case
tests/kernels/moe/test_moe.py: NUM_EXPERTS 增加 192 的正确性覆盖
```

```cpp
+#ifndef USE_ROCM
+        case 192:
+            LAUNCH_SOFTMAX(192, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
             break;
+        ...
+        case 576:
+            LAUNCH_SOFTMAX(576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
+            break;
+#endif
```

## 性能声明

（fused_topk 层延迟，`benchmark_moe.py` 只保留 `fused_topk`；baseline 为两 kernel 回退路径；单位 us。）小 batch（1–64）：延迟降 32.7%–46.2%；大 batch（128+）：降 39.4%–86.7%。样例（576 experts top-8）：bs=1 16.77→8.77 us、bs=4096 79.02→17.03 us、bs=76800 1,703.97→247.73 us（−85.5%）；192-top8 bs=76800 1,056.43→149.77 us；512-top8 bs=76800 1,574.37→176.89 us。576-top1 bs=1 8.77→5.84 us。测试条件（GPU 型号/精度）PR 未注明。

## 验证与回归风险

- 验证：`tests/kernels/moe/test_moe.py` 正确性测试（NUM_EXPERTS 扩到 8/64/192，EP 1/4，top_k 2/6）；延迟基准覆盖 17 个 batch size × 4 个 (experts, topk) 组合。
- 回归风险：非 2 幂融合 kernel 仅在 CUDA 上启用（ROCm 维持回退）；8 字节加载路径对 64 倍数专家数的正确性依赖测试覆盖（192 一档）；其他专家数（非 2 幂且非 64 倍数）行为不变（仍走 workspace 两 kernel 路径）；kernel 模板实例数量增加会增大编译产物。

## 关联

- 同区域：本批 MoE PR #28892（TRT-LLM fp4 MoE kernel）、PR #61457（per-layer FSE）；相关 topK kernel 工作 PR #43008（DeepSeek indexer 的 cluster-cooperative topK，不同子系统）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
