# PR #10572: [Bug] Fix Issue#10215

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10572](https://github.com/sgl-project/sglang/pull/10572) |
| 作者 | yuhyao |
| 合并日期 | 2025-09-25 |
| labels | high priority, run-ci |
| 热度 | 👍 5 · 💬 2 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
cutlass W4A8 MoE GEMM 内核缺少 `warpgroup_wait` 等待(ptxas C7517/C7514 告警),此前 #10241 只删了失败用例;本 PR 补齐等待并对齐 TensorRT-LLM 实现。

## 问题现象
编译(编译内核)时观察到 ptxas 告警:其一为 `(C7517) warpgroup.wait is injected in around line 27304 by compiler to allow use of registers defined by GMMA`,出现在 cutlass `GemmUniversal` / `CollectiveMmaArrayMixedInput` / `MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput` 的 W4A8(mixed input,integer_subbyte 4bit × bf16/fp8)分组 GEMM 内核中;其二为 `(C7514) Potential Performance Loss: wgmma.mma_async instructions are serialized due to non wgmma instructions reading accumulator registers of a wgmma between start and end of the pipeline stage`。正文未描述端侧运行时症状(如精度错误或崩溃),也未提供基准数据。

## 根因
正文披露:上述 ptxas 告警表明内核中缺少某些 `warpgroup_wait` 指令(C7517 说明编译器被迫自动注入 warpgroup.wait,C7514 说明非 wgmma 指令在流水线阶段间读取 wgmma 累加器寄存器导致 wgmma.mma_async 串行化)。同时正文指出,针对 [Issue#10215](https://github.com/sgl-project/sglang/issues/10215) 的此前修复 [PR#10241](https://github.com/sgl-project/sglang/pull/10241) 并未真正解决问题——它只是删除了 issue 中提到的失败用例;本 PR 才给出正确修复。

## 修复方案
改 `sgl-kernel/csrc/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gemm` 路径下的 `sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp`:
- 在三处关键位置(中间结果消费前、`k_block == K_BLOCK_MAX - 1` 的最后一个 k_block 处、`(k_block + 1) % NumMMAsPerChunk == 0` 且 ScaleOut::Zero 处)新增 `warpgroup_wait<0>();`,确保在非 wgmma 指令读取累加器前所有 GMMA 批次完成;
- 相应删除三处 `warpgroup_wait<K_WAIT_MAX>()`(含一处原有注释)与一处提前的 `tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;`,并将函数末尾原有的 `warpgroup_wait<0>()` 注释掉(`// warpgroup_wait<0>();`);
- 实现对齐 [TensorRT-LLM 的对应文件](https://github.com/NVIDIA/TensorRT-LLM/blob/d3467f9f12b921f987706c46ff6009e393545bcf/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp)。
- 测试 `sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py`:`test_int4_fp8_grouped_gemm_multi_experts` 的参数矩阵恢复扩大(k 增加 256,n 增加 7168)——即 #10241 删除失败用例后重新覆盖这些形状。
修改后 `test_cutlass_w4a8_moe_mm.py` 全部单测通过。

关键 diff 摘录:
```diff
           tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
+
+          warpgroup_wait<0>();
           warpgroup_fence_operand(intermediate);
```

## 影响与回归风险
- 回归面:仅 `sgl-kernel` 的 cutlass W4A8 MoE 分组 GEMM 内核(SM90 mixed-input collective)及其单测;`warpgroup_wait` 位置变化属同步/屏障语义调整,若等待过严可能影响该内核吞吐,若仍有遗漏可能再现 C7514/C7517;受影响的是 W4A8 MoE 量化路径(如 DeepSeek 类 w4a8 场景)。正文未提供精度或性能基准数据,相关 checklist 项未勾选;影响窗口正文未说明。

## 关联
- [#10215](https://github.com/sgl-project/sglang/issues/10215)
- [#10241](https://github.com/sgl-project/sglang/pull/10241)
- [TensorRT-LLM 参考实现](https://github.com/NVIDIA/TensorRT-LLM/blob/d3467f9f12b921f987706c46ff6009e393545bcf/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
