# PR #20386: perf(qwen3_5): replace einops rearrange with torch.flatten in GatedDe…

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @vedantjh2 |
| 合并 | 2026-03-12 |
| 分类 | Model-specific Fast Path(major) |
| Label | run-ci |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 为单文件 2 行改动(qwen3_5.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/20386 |

## 一句话结论

把 Qwen3.5 GatedDeltaNet 输出的 einops rearrange 换成 `torch.flatten(-2)`,该调用每次 12.67us→4.74us(2.67x),GSM8K 延迟 28.075s→26.704s。

## 动机

`einops.rearrange` 每次调用都要在 Python 层做 pattern 字符串解析与校验。`Qwen3_5GatedDeltaNet.forward()` 里的 `rearrange(x, "... h d -> ... (h d)")` 实际只是把最后两维合并,等价于零拷贝 view 的 `torch.flatten(-2)`,没有任何解析必要。该调用位于模型每个 GDN 层的每次 forward 上,开销随层数与步数累积。

## 优化点

一行替换 + 清理:将 `rearrange(core_attn_out, "... h d -> ... (h d)")` 改为 `core_attn_out.flatten(-2)`,并删除文件顶部不再使用的 `from einops import rearrange`。`flatten(-2)` 与原 rearrange 语义完全相同(合并末两维),无行为变化、无 kernel 变化,收益全部来自消除 Python 层 pattern 解析开销。

## 关键改动

```text
python/sglang/srt/models/qwen3_5.py: Qwen3_5GatedDeltaNet.forward 末尾的 rearrange 改为 flatten(-2),删除 einops import
```

关键改动片段:

```python
-        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
+        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
```

## 性能声明

PR 注明条件:Qwen3.5-9B,H100 80GB,TP=1,CUDA graphs disabled,`sglang.bench_serving`(10 prompts,input 256 / output 32)+ torch profiler,Perfetto SQL 按 GatedDeltaNet 祖先切片过滤:

- kernel/调用级:720 次调用,`rearrange` 平均 12.67us / 总 9125.43us;`flatten` 平均 4.74us / 总 3410.18us(其中 aten::flatten 2.50us)。per-call 2.67x,窗口总时间降 63%,省 5715us。
- E2E(GSM8K 基准同时测性能):latency 28.075s→26.704s,output throughput 789.6→828.1 tok/s(约 +4.9%)。

## 验证与回归风险

`flatten(-2)` 与 rearrange 语义等价,PR 以 GSM8K 验证:accuracy 0.905(baseline)vs 0.900(optimized),invalid 均 0.000;PR 未对 0.005 的差异做显著性说明,按语义等价应属噪声。Checklist 各项未勾选(无新增单测);diff 仅 2 行,回归面极小。风险:einops pattern 若未来改为非纯 flatten 的形状操作需另选实现;此处为唯一 einops 使用点,删 import 后无残留依赖。

## 关联

同区域:Qwen3.5 GatedDeltaNet/GDN 混合架构模型 fast path;本卡未收录同分类其他 PR,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
