# PR #7636: [TPU] Optimize RoPE forward_native2

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2024-08-18 |
| 分类 | Attention / Kernel(major) |
| Label | tpu |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用(单文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/7636 |

## 一句话结论

重写 `RotaryEmbedding.forward_native2` 使其更利于编译器优化,TPU 上消除 cos-sin cache 的冗余拷贝,延迟约降 10%。

## 动机

原 `forward_native2` 用复数表示(`view_as_complex` / `freqs_cis = cos + 1j*sin`)实现 RoPE,并按 batch/seq 四维 shape 处理。这种写法对编译器(尤其在 TPU 的 XLA 场景)不友好,会引入对 cos-sin cache 的冗余 copy 操作,拖慢前向。

## 优化点

改用实数域 rotate-half 形式:cache 统一为 `cos_sin_cache`(fp32 算好后 cast 到目标 dtype 注册为 buffer,不再构造复数 `freqs_cis`),forward 里 `index_select` 后直接 `chunk` 出 cos/sin。`_apply_rotary_emb` 改为对 `[num_tokens, num_heads, head_size]` 的 `x` 做 chunk 后 `o1 = x1*cos - x2*sin; o2 = x2*cos + x1*sin`,去掉 transpose/cat/reshape 链条;shape 处理简化为 flatten 后的 `num_tokens` 一维视角,减少动态 shape 分支,整体更 compiler-friendly。

## 关键改动

```text
vllm/model_executor/layers/rotary_embedding.py: _apply_rotary_emb 改实数域实现;
  forward_native2 用 cos_sin_cache + chunk,去掉复数 freqs_cis 与 batch_size 维度
```

## 性能声明

TPU 上约 10% latency reduction(正文原话:"this removes redundant copy operations to the cos-sin cache, leading to ~10% latency reduction")。未注明具体模型、TPU 型号与 workload 条件。

## 验证与回归风险

pack 中未列出新增测试;PR 未附 benchmark 数值表。回归风险:仅 `use_native2`(TPU 且 neox style)路径行为改变,cache 注册从 `freqs_cis` 复数 buffer 变为 `cos_sin_cache`,依赖该 buffer 名的外部代码会受影响;CUDA/CPU 等其他平台走 `forward_native`,不受影响。TODO 注明后续会把 `forward_native` 一并重写并合并两个实现。

## 关联

同区域为 RoPE 实现优化;pack 正文未列出具体关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
