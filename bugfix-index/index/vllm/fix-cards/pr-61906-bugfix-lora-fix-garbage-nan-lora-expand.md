# PR #61906: [Bugfix][LoRA] Fix garbage/NaN lora_expand output on SM90 with fp32 i…

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61906](https://github.com/vllm-project/vllm/pull/61906) |
| 作者 | BeichenHuang |
| 合并日期 | 2026-10-16 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 6 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
SM90 上 fp32 输入的单切片 lora_expand 在 rank≥64 时输出垃圾/NaN;强制该路径 num_stages=1 绕过 ptxas 误编译。

## 问题现象
SM90 上,fp32 输入(即真实 `add_lora_linear` 路径)的单切片 `lora_expand` 在 rank ≥ 64 时返回垃圾值或 NaN。main 上 `tests/lora/test_punica_ops.py -k fp32_inputs` 8 failed / 28 passed,失败项恰为 `nslices=1 × rank∈{64,128} × {fp16,bf16} × 2 batch shapes`,最多 43% 元素不匹配、最大绝对差 3006。端到端:Qwen3-8B 挂 r=64 LoRA(7 个投影),2048-token prompt 的 logprobs 2047/2047 全为 NaN(HF transformers + PEFT 对照)。内核级复现器:H20 上 main 输出 NaN 约 6000–9000 个、错误元素约 3 万–45 万。

## 根因
正文披露:kernel 内的 fp32→bf16 转换(`CAST_TYPE`)使 wgmma 从寄存器读取 A 操作数。`num_stages=2` 时 Triton 让每次迭代的第二个 wgmma 跨循环回边保持在 flight 中,ptxas < 13.3 会把该 wgmma 的 A 寄存器复用给下一次迭代的 shared-memory 加载(triton-lang/triton#9433)。Triton 3.5.0–3.8.0 组装 SM90 kernel 都用受影响的 ptxas;用 ptxas 13.3(`ptxas-blackwell`)时未修复的 kernel 可通过新测试,证实该结论。

## 修复方案
`vllm/lora/ops/triton_ops/lora_expand_op.py`:在 config 选择之后对 SM90+CUDA 的 `CAST_TYPE` 路径强制单 stage,因此也覆盖 `VLLM_TUNED_CONFIG_FOLDER` 的调优配置:

```python
CAST_TYPE = True
# Pipelined wgmma with a register A operand is miscompiled by
# ptxas < 13.3 on SM90 (triton-lang/triton#9433).
if current_platform.is_cuda() and current_platform.is_device_capability(90):
    NUM_STAGES = 1
```

并新增 `test_lora_expand_fp32_inputs` 覆盖此前无测试的 `CAST_TYPE` 路径。单 stage 下 Triton 等待每个 wgmma 在自身迭代内完成,结构性消除冒险。

## 影响与回归风险
- 影响窗口:正文未写明版本范围(受影响条件为 SM90 + CAST_TYPE + rank≥64 + ptxas<13.3,Triton 3.5.0–3.8.0)。
- 回归面:仅作用于 CUDA SM90 且 `CAST_TYPE` 的路径,ROCm 与 bf16/fp16 输入调用方不变,覆盖所有 slice 数。单切片层最差 shape(R=128, T=8192)开销 +8%(`add_inputs=False`)/+21%(`add_inputs=True`),多切片层不变且实测正确;对比 #48638 同 shape +106%。ptxas ≥ 13.3(Triton 3.9.0 经 triton-lang/triton#12035)后可移除。`fused_moe_lora_op.py` 的 fp16 MoE cast+`num_stages=3` 可能命中同一 bug,未验证,留作后续。H20 验证:全套 `tests/lora` 4309 passed;CI 的 H200 MIG(SM90)会跑新测试。

## 关联
- Fixes [#48590](https://github.com/vllm-project/vllm/issues/48590)
- [#48638](https://github.com/vllm-project/vllm/pull/48638)(替代方案)
- [triton-lang/triton#9433](https://github.com/triton-lang/triton/issues/9433)(上游 bug)
- [triton-lang/triton#12035](https://github.com/triton-lang/triton/pull/12035)(ptxas ≥ 13.3)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
