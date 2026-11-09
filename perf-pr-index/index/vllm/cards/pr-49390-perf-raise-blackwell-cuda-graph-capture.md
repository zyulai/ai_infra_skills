# PR #49390: [Perf] Raise Blackwell CUDA graph capture default to 1024

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LucasWilkinson |
| 合并 | 2026-08-07 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | performance, ready, nvidia |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 本身可用(3 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/49390 |

## 一句话结论

把数据中心 Blackwell(CC 10.x)GPU 的默认 `max_cudagraph_capture_size` 上限从 512 提到 1024;B300 上 DeepSeek-V4-Flash 固定验证 workload 在并发 64-256 下输出吞吐提升 +393% 至 +571%。

## 动机

在 B300 上,DeepSeek-V4-Flash DSpark workload 使用固定验证(7 个 draft tokens,关闭 adaptive/confidence-based verification)时,CUDA graph 捕获上限 512 成为瓶颈:大批次无法落入已捕获的图,吞吐显著低于把上限放宽到 1024 的配置。其他平台保留 512 默认值,用户显式配置不受影响。

## 优化点

改动只触及默认值推导逻辑:`VllmConfig._set_cudagraph_sizes` 在用户未显式指定 capture size 时,原默认上限为 `min(max_num_seqs * decode_query_len * 2, 512)`,现改为 `min(max_num_seqs * decode_query_len * 2, default_max_graph_size)`,其中 `default_max_graph_size` 在 `current_platform.is_device_capability_family(100)`(数据中心 Blackwell)时取 1024,否则取 512。`max_num_batched_tokens` 的既有约束与 `max_num_seqs * decode_query_len * 2` 上界仍然生效,因此推导出的 capture size ≤512 的 workload 完全不受影响。捕获尺寸序列的生成规则(1, 2, 4,然后 8 步进到 256、16 步进到上限)不变。

## 关键改动

```text
vllm/config/vllm.py: _set_cudagraph_sizes 默认上限按平台取 512/1024
vllm/config/compilation.py: max_cudagraph_capture_size 文档字符串更新
tests/compile/test_config.py: 新增 test_blackwell_cudagraph_default(Blackwell→1024, 非 Blackwell→512)
```

```python
decode_query_len = 1 + self.num_speculative_tokens
default_max_graph_size = (
    1024 if current_platform.is_device_capability_family(100) else 512
)
max_cudagraph_capture_size = min(
    self.scheduler_config.max_num_seqs * decode_query_len * 2,
    default_max_graph_size,
)
```

## 性能声明

硬件 NVIDIA B300 SXM6 AC,DeepSeek-V4-Flash DSpark,固定验证(7 draft tokens;adaptive/confidence-based verification 关闭)。上限 512→1024:并发 64 输出 tok/s 1,127→7,559(+571%);并发 128:904→5,101(+464%),mean TPOT 98.77 ms→18.38 ms;并发 256:977→4,812(+393%)。代价:该配置下捕获到 1024 耗时 80 秒、占用每 GPU 4.88 GiB。B200 验证在 PR 撰写时尚未完成(以 draft 打开)。

## 验证与回归风险

`tests/compile/test_config.py` 新增参数化测试(Blackwell→1024、非 Blackwell→512),另跑 2 个 focused Blackwell/非 Blackwell 用例与 pre-commit。作者未做精度评估(理由:仅改变 CUDA graph 覆盖,不改模型计算与采样;serving benchmark 各并发下 256 请求零失败)。回归风险:Blackwell 上启动时间与显存占用随图数量增加(本配置 80 s / 4.88 GiB per GPU);非 Blackwell 平台与显式配置路径不变。

## 关联

#46324(相关:speculative-decode capture-size 对齐,但不改最大捕获上限)。无其他同区域 PR 信息。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
