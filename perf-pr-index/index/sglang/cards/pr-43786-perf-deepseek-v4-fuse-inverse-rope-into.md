# PR #43786: [Perf] DeepSeek-V4: Fuse inverse RoPE into wo_a FP8 quantization

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Phoenix3334 |
| 合并 | 2026-11-04 |
| 分类 | Quantization |
| Label | quant, deepseek, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/43786 |

## 一句话结论

把 DeepSeek-V4 的 inverse RoPE 折进 wo_a 激活 FP8/UE8M0 量化 kernel,消除一次 BF16 中间写读,B300 prefill 吞吐 +1.6%。

## 动机

DeepSeek-V4 在 attention 输出进入 `wo_a` 投影前要施加 inverse RoPE。现有 CUDA UE8M0 FP8 路径单独启动 inverse RoPE kernel,把旋转后的 BF16 值写回显存,再由量化 kernel 重新加载,多出一次往返。#37658 已在 ROCm/aiter 引入同样的 inverse-RoPE+FP8 量化融合及 `SGLANG_OPT_FP8_WO_A_FUSED_INVROPE` 开关,本 PR 把该开关扩展到 CUDA UE8M0 路径;它与 #39957 的小批量 BF16 wo_a 融合互补(BF16 与 FP8 两个 gate 互斥,收益不可叠加)。

## 优化点

新增 JIT kernel `fp8_wo_a_group_major_quant_ue8m0_rope_kernel`:加载输入时对每个 512 维 head 的最后 64 维就地做 inverse RoPE,旋转值保留在寄存器中,先按 BF16 round 再做组级 absmax 与 UE8M0 scale 打包量化,保持与 unfused 路径相同的舍入边界。实现上把 position/frequency 读取提前并与输入加载重叠(kPrefetchRope),frequency 走 float4 向量化 `__ldg`;将量化/打包 scale epilogue 提取为 `QuantizeWoAGroup` 供融合与非融合 kernel 共享,host 侧张量校验抽成 `FP8WoAQuantParams` 复用,Python 侧输出分配复用 `_allocate_wo_a_quant_outputs`。静态 eligibility(头维 512、rope_dim 64、UE8M0、非 NPU/HIP)缓存在 `MQALayer.__init__`,执行时再检查 breakable CUDA graph、`inv_rope is None`、shape/dtype。开关沿用 `SGLANG_OPT_FP8_WO_A_FUSED_INVROPE`,默认关闭。

## 关键改动

```text
python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh: 融合 inverse RoPE 的 UE8M0 组量化 kernel
python/sglang/kernels/ops/quantization/fp8_group_major_quant_ue8m0.py: sglang_per_token_group_quant_fp8_dsv4_wo_a_with_rope 入口
python/sglang/srt/models/deepseek_v4.py: use_fused_rope_quant eligibility 与 forward 分支
python/sglang/srt/environ.py: 开关注释扩展到 CUDA UE8M0 路径
```

```cuda
// Phase 3: rope + absmax + UE8M0 quant, preserving the BF16 rounding boundary.
if (do_rope[k]) {
  const float* fr = reinterpret_cast<const float*>(freq[k]);
  for (uint32_t p = 0; p < INPUT_VEC_SIZE / 2; ++p) {
    const float c = fr[2 * p]; const float s = fr[2 * p + 1];
    const float a = static_cast<float>(input_vec[2 * p]);
    const float b = static_cast<float>(input_vec[2 * p + 1]);
    input_vec[2 * p]     = static_cast<T>(a * c + b * s);
    input_vec[2 * p + 1] = static_cast<T>(b * c - a * s);
  }
}
```

## 性能声明

B300 x8,DeepSeek-V4-Pro,aiopt v0.5.21,TP8/DP8,MegaMoE,60k input/1 output,C16/N160,3 seeds:Baseline 输入吞吐 ~101,710 tok/s → Fused ~103,349 tok/s(~+1.61%),median TTFT 9.600 s → 9.452 s(~-1.54%)。历史 v0.5.17 B300 数据:60k/128k 输入、并发 1-32 共 8 个工作点吞吐 +1.17%~+2.66%(如 60k C1 +2.66%、128k C1 +2.23%),24/24 配对比较为正。PR 明确注明:这些测量先于 cleanup commit,非当前 head 的结果;refactor 后的全模型 serving 与性能验证仍 pending。

## 验证与回归风险

新增两个 GPU 测试(`test_fp8_wo_a.py`):production 数值等价(T ∈ {0,1,32,100,777,2047,8192,8193},G ∈ {2,16},FP8 codes 与 packed scales bitwise 相等)与 CUDA Graph capture/replay(更新输入后回放)。当前 head(`fb4e80e`)B200 kernel CI 7 tests 全过,Hosted Lint 通过;pack 显示 PR Test Base/Extra 失败、AMD ROCm 10 通过,PR checklist 中剩余 CI、refactor 后全模型验证与 reviewer 验证未勾选。风险面:改动 DeepSeek-V4 MQA forward 路径的 rope/quant 分支选择,breakable CUDA graph 内强制回退 unfused;历史 B300 验证基于非 main 的 aiopt backport,scale 存储契约在 upstream 需独立重验。

## 关联

前序 #37658(ROCm/aiter 版本,引入共享开关)、#39957(BF16 wo_a 小批量融合,gate 互斥);测试 fixture 修正分别取自 #47736 与 #47758。同分类其他 PR 素材未提供。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
