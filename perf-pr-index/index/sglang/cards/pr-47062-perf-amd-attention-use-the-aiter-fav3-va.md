# PR #47062: [Perf] [AMD] attention: Use the AITER FAv3 varlen kernel on gfx942 as well

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @avjves |
| 合并 | 2026-10-30 |
| 分类 | Attention / Kernel (tier: major) |
| Label | diffusion |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 仅含 multimodal_gen runtime aiter attention backend 一处分支删除与一个参数新增 |
| 链接 | https://github.com/sgl-project/sglang/pull/47062 |

## 一句话结论

移除 gfx942 上把 varlen 注意力强制改走 AITER Triton kernel 的陈旧 workaround,恢复 FAv3 varlen kernel,varlen shape 提速 6.6–27%(中位 15.1%),MiniMax-H3 E2E -11.4%。

## 动机

`aiter.py` 的 `forward_varlen` 曾以"仅看架构"的条件把 gfx942 上的所有 varlen 调用都路由到 AITER 的 Triton kernel,以规避 #33275 报告的 MiniMax-H3 packed sequences 挂起问题——该条件没有任何 shape、长度或 head 数限定,导致 MI300X/MI325X 上任何模型、任何 shape 都无法到达 AITER 的 FAv3 varlen kernel。作者系统复现后确认挂起已不复现,判定 workaround 过时。

## 优化点

改动很小但机制明确:删掉 `USE_AITER_GFX942` 分支,varlen 调用统一走 `aiter.flash_attn_varlen_func`(FAv3 kernel);同时为 varlen 启用 RTZ rounding mode(`how_v3_bf16_cvt=AITER_BF16_CVT_MODE`),与 non-varlen 路径的做法一致,类似 #28650 的收益,该开关仅影响 gfx942,非 gfx942 平台安全(被忽略)。作者用假设消解法验证挂起不存在:零长第二段(`used % 64 == 0`)、两段分组本身、长度阈值(1024–104448)、奇数 head 数(Ulysses8 余 7)均排除;真实 H3 的全部 32 个 varlen shape(4 种时长/分辨率组合 x 4 个 Ulysses degree,28K–104K rows)与端到端 8xMI300X Ulysses8 运行均无挂起。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/attention/backends/aiter.py: forward_varlen 删除 gfx942→Triton kernel 分支,统一 aiter.flash_attn_varlen_func;新增 how_v3_bf16_cvt=AITER_BF16_CVT_MODE(RTZ rounding)
```

```python
- if USE_AITER_GFX942:
-     attention_func = importlib.import_module(
-         "aiter.ops.triton.attention.mha").flash_attn_varlen_func
- else:
-     attention_func = aiter.flash_attn_varlen_func
+ attention_func = aiter.flash_attn_varlen_func
```

## 性能声明

- kernel 级:作者遍历 H3 提交的所有 varlen shape(28K–104K rows,全部 4 个 Ulysses degree,含零长第二段),FAv3 kernel **快 6.6–27%(中位 15.1%)**,输出与 Triton kernel 差异在 bf16 累积噪声内。
- 端到端(MiniMax-H3,8xMI300X Ulysses8,1344x768、4s、50 steps,4 requests=1+3 repeats):E2E Latency **62595.09 ms → 55479.46 ms(-7115.63 ms,-11.4%)**;分阶段中 DenoisingStage 55305.21 → 48982.85 ms(-11.4%),其他阶段变化均在噪声内;Throughput 两边均显示 0.02 req/s。

## 验证与回归风险

验证以"挂起不存在"的假设测试矩阵为主(结构化 cu_seqlens 用例、head sweep、真实 H3 32 shape、端到端 asm arm),并附 before/after/RTZ 三段视频。回归风险:改动只在 multimodal_gen(diffusion)的 aiter attention backend,若 #33275 描述的挂起在其他环境仍存在,该问题会重新暴露(作者证据表明已不复现);RTZ rounding 改变 bf16 转换舍入模式,输出数值有微小变化(作者以视频对比展示);Accuracy Tests 一节无表格化指标。CI 状态:Base/Extra/AMD ROCm 10 均记录失败(:x:),是否与改动相关未注明。

## 关联

前序:#33275(引入 gfx942 workaround 的挂起报告)、#28650(non-varlen 的 RTZ rounding 收益);同区域:multimodal_gen 的 AMD attention 后端(与 #35882 同一 diffusion 运行时)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
