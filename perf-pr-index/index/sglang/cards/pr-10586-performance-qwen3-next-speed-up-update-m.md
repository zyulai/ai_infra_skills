# PR #10586: [Performance] Qwen3-Next: speed up update_mamba_state_after_mtp_verify by 10x; e2e up to 3.54% faster

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @byjiang1996 |
| 合并 | 2025-09-19 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | high priority, run-ci |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/10586 |

## 一句话结论

把 Qwen3-Next MTP verify 后的 mamba 状态回写从逐元素 Python 循环改为两次向量化 scatter,单次调用从约 3158 us 降到约 307 us(10x),e2e 吞吐 +2.3~3.5%。

## 动机

MTP 多 token 预测验证接受若干 draft token 后,需要把该步的中间 SSM 状态与 conv 窗口回写到状态缓存(`update_mamba_state_after_mtp_verify`)。旧实现为每个 (cache line, step) 组合跑两层 Python `for` 循环,循环体内做 `.item()`(隐式 GPU->CPU 同步)再逐条 `copy_`,在验证路径上引入大量小 kernel 与同步开销,成为可观测的热点(profile 显示平均 3157.98 us)。

## 优化点

改动集中在 `hybrid_linear_attn_backend.py` 的 `update_mamba_state_after_mtp_verify`,机制是消除逐元素循环与 `.item()` 同步:SSM 状态回写由"按 chunk 分组后逐 (ci, st) 取标量索引再 copy_"改为一条高级索引赋值——`ssm_states[:, valid_state_indices, :] = intermediate_state_cache[:, valid_state_indices, last_steps]`,conv 窗口同理。valid mask 过滤、`last_steps = accepted_length - 1` 等公共索引计算保持不变,只是把原先 256 一组、组内再逐元素的嵌套循环整体替换为对整批 valid 请求的一次 batched gather + scatter,由 PyTorch 在 GPU 上融合完成。代码从约 35 行缩到约 8 行。

## 关键改动

```text
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py: update_mamba_state_after_mtp_verify 改为向量化 scatter 回写
```

关键片段(替换后的两条回写):

```python
ssm_states[:, valid_state_indices, :] = intermediate_state_cache[
    :, valid_state_indices, last_steps
].to(ssm_states.dtype, copy=False)

conv_states[:, valid_state_indices, :, :] = intermediate_conv_window_cache[
    :, valid_state_indices, last_steps
].to(conv_states.dtype, copy=False)
```

## 性能声明

- 函数级(profile 前后对比):Before 平均 3157.98 us,After 平均 307.05 us,10x faster
- e2e GSM8k 吞吐(`benchmark/gsm8k/bench_sglang.py`):TP4 MTP 2055.889 -> 2103.981 tokens/s(+2.34%);TP4 DP2 MTP 1858.580 -> 1924.054 tokens/s(+3.53%,标题写 up to 3.54%)

准确性:GSM8K accuracy 前后保持约 0.950。PR 未注明硬件型号与输入长度等工作负载细节。

## 验证与回归风险

PR 正文以 GSM8K 准确性(约 0.950 不变)与吞吐基准作为验证,未注明新增单元测试。回归面:仅影响 hybrid linear attention(Qwen3-Next 类 mamba/GDN 混合模型)带 MTP 的 verify 路径;语义上依赖 `valid_state_indices` 无重复(重复索引时高级索引赋值与逐条 copy_ 的最后写入语义可能不同),PR 未讨论该点;非 MTP、非混合线性注意力模型不受影响。

## 关联

同区域为 Qwen3-Next / hybrid linear attention 性能优化;pack 素材未列出同分类其他 PR 编号,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
