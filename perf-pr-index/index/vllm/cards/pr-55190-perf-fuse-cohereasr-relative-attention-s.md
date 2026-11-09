# PR #55190: [Perf] Fuse CohereASR relative attention score accumulation

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Levius-Fubuki |
| 合并 | 2026-09-18 |
| 分类 | Attention / Kernel (tier: major) |
| Label | ready, multi-modality, cohere |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 2 个文件(模型实现与回归测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/55190 |

## 一句话结论

用 `torch.baddbmm` 融合 CohereASR 相对注意力的 content score 累加与缩放,48 种配置几何均值提速 1.0957x-1.1019x,T=1024 时达 1.17x,并省 10.6%-21.5% 临时峰值显存。

## 动机

`RelPositionMultiHeadAttention` 此前计算相对位置注意力的分数路径为:单独一次 `torch.matmul` 算 content score(`matrix_ac`)并物化成临时张量,再与 relative-position score(`matrix_bd`)逐元素相加,最后整体除以 `s_d_k` 缩放。三步各自产生 kernel 启动与中间张量,在长序列下开销与临时显存都随 T 增长。

## 优化点

纯 PyTorch 层融合,不引入自定义 kernel、新依赖、运行时开关或模型 gate。利用 `torch.baddbmm` 的 `alpha`/`beta` 语义,把 `matrix_bd` 作为累加项(c = beta*input + alpha*(A@B)),一次调用同时完成 content score 的 matmul、两项相加与 `1/s_d_k` 缩放:op 计数从 `{bmm: 2, add: 1, mul: 1}` 降为 `{bmm: 1, baddbmm: 1}`,消除了独立的 content-score 临时张量与逐元素 add/scale。实现细节:先按 key 序列长度截断 `matrix_bd`(`matrix_bd[:, :, :, :k.size(-2)]`),再把 `batch x heads` 展平成 bmm 的 batch 维,调用后 view 回 `(batch, heads, query_len, key_len)`。

## 关键改动

```text
vllm/model_executor/models/cohere_asr.py: forward 中 score 累加改为单次 baddbmm(alpha=beta=1/s_d_k)
tests/models/multimodal/test_cohere_asr.py: 新增 _ReferenceRelPositionMultiHeadAttention 回归测试(方/非方 q-k 长度,断言 baddbmm 恰调用一次)
```

融合调用(节选):

```python
matrix_bd = matrix_bd[:, :, :, : k.size(-2)]
batch, heads, query_len, key_len = matrix_bd.shape
scale = 1.0 / self.s_d_k
scores = torch.baddbmm(
    matrix_bd.flatten(0, 1),
    q_with_bias_u.flatten(0, 1),
    k.transpose(-2, -1).flatten(0, 1),
    beta=scale,
    alpha=scale,
).view(batch, heads, query_len, key_len)
```

## 性能声明

Microbenchmark(单张 RTX 4090,生产维度:8 heads、model dim 1280、head dim 160,每点 5 次重复取中位、25 warmup + 200 测量迭代):48 种配置中 eager 几何均值 1.1019x,CUDA Graph 几何均值 1.0957x,最低 1.0307x,T=1024 几何均值 1.1700x;估算临时峰值显存降低 10.61%-21.52%。典型点(b=1,T=1024,FP16):213.924 对 172.569 µs,1.2396x;(b=8,T=1024,BF16):2595.200 对 2274.406 µs,1.1410x。完整 attention 层基准:T=375 为 1.016x-1.039x,T=1024 为 1.073x-1.089x。真实模型 Earnings-22(511 样本,/v1/audio/transcriptions,8 并发):34.826 s/14.673 req/s 对 32.496 s/15.725 req/s,吞吐 1.0717x;WER 11.8192% 对 11.8552%(上游阈值 12.1392%)。作者注明真实模型吞吐为单次配对运行,仅作辅助证据;单请求 e2e 延迟噪声大,不作声明。

## 验证与回归风险

focused tests 5 passed;GPU 数值校验 36/36(FP16/BF16 配置)通过;两侧均通过上游 WER 阈值;Ruff 与 `git diff --check` 通过。回归测试用保留旧实现的 reference 子类对比融合路径输出(`torch.testing.assert_close`),覆盖 query/key 长度不等的场景,并用 monkeypatch 计数断言 `torch.baddbmm` 恰好调用一次。风险面:仅影响 CohereASR 的 `RelPositionMultiHeadAttention`;`matrix_bd` 截断时机从"与 matrix_ac 比对"改为"与 k.size(-2) 比对",语义等价但依赖 q/k 长度关系;数值顺序变化(加法融合进 baddbmm)可能产生与旧路径的微小浮点差;跨平台覆盖依赖 CI(作者仅测 RTX 4090)。

## 关联

正文 duplicate check:检索 open PR 中 CohereASR relative attention / baddbmm / RelPositionMultiHeadAttention / matrix_bd 未发现同类优化。其他关联 PR 素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
