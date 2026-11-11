# PR #48406: [Perf] [AMD] qwen3: Let the fused QK-norm + RoPE + KV write serve the vectorized_5d layout

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @siliangchen-amd |
| 合并 | 2026-11-09 |
| 分类 | Attention / Kernel(tier: major) |
| Label | run-ci |
| Diff | +138/-23,2 files(pack diffstat 缺失,按 pack 内 diff 文本计数) |
| 链接 | https://github.com/sgl-project/sglang/pull/48406 |

## 一句话结论

Qwen3 decode 的 fused QK-norm+RoPE+KV write 扩展到 vectorized_5d KV 布局,fp8 KV 下输出 tok/s 最高 +20.1%。

## 动机

Qwen3 的 decode 路径把 QK-norm、RoPE 与 KV-cache 写入融进一个 aiter kernel(#44267),但此前只服务 NHD 布局:使用 `vectorized_5d` KV-cache 布局(`SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`)时回退到 unfused 路径。而该 kernel 在给定 `use_shuffle_layout` 与 cache 的 `block_size`、`x` 时,本来就写 aiter 的 shuffle 布局——`vectorized_5d` 正是该布局,所以只需让入口正确传参。

## 优化点

`aiter_fused_qk_norm_rope_set_kv` 用 `k_cache.dim() == 5` 识别 5D cache:5D 时原样传入 kernel 并设 `use_shuffle_layout=True`、`block_size=k_cache.shape[3]`、`x=k_cache.shape[4]`;NHD cache 保持 `unsqueeze(1)` + `block_size=0` 的旧约定(一个 slot 一个 block)。`Qwen3Attention.forward` 的 fused-path 门控从 `k_cache.dim() == 3` 放宽为 `k_cache.dim() in (3, 5)`,dtype 条件(activation dtype 或 aiter 的 fp8)不变;RoPE 与 mRoPE 两条 fused kernel、bf16 与 fp8 cache 均覆盖。

## 关键改动

```text
python/sglang/srt/models/qwen3.py: aiter_fused_qk_norm_rope_set_kv 的 5D 传参与 forward 门控放宽
test/registered/amd/test_qwen3_aiter_fused_qk_norm_rope.py: 新增 5D RoPE/mRoPE bf16+fp8 用例(经 gather_shuffle_5d_to_linear 读回)
```

```python
shuffle = k_cache.dim() == 5
...
k_cache=k_cache if shuffle else k_cache.unsqueeze(1),
v_cache=v_cache if shuffle else v_cache.unsqueeze(1),
use_shuffle_layout=shuffle,
block_size=k_cache.shape[3] if shuffle else 0,
x=k_cache.shape[4] if shuffle else 0,
```

## 性能声明

单卡 MI355X(gfx950),rocm720-mi35x image(2026-11-03),`sglang.benchmark.serving` random-ids 1024 in/256 out,每点两轮均值,`--mem-fraction-static 0.85`,fp8 行加 `--kv-cache-dtype fp8_e4m3`。输出 tok/s(main→PR):Qwen3-8B fp8 C1/C16/C64 = 209.75→239.99(+14.4%)/2701.68→3245.59(+20.1%)/6809.71→7680.24(+12.8%),TPOT -12.8%/-18.3%/-13.6%;Qwen3-14B-FP8 +9.7%/+18.0%/+10.3%;Qwen3-8B bf16 +6.4%/+8.9%/-0.8%(C64 两轮分歧大:PR 侧 6911.85/6045.08);Qwen3-32B bf16 +3.6%/+5.8%/+3.4%。one_batch decode step(Qwen3-8B bf16):bs1 -5.5%、bs16 -7.3%、bs64 -6.5%。GSM8K 1319 题(vectorized_5d):Qwen3-8B bf16 0.903→0.901、fp8 0.907→0.902;Qwen3-14B-FP8 0.895→0.905;Qwen3-32B 0.843→0.847。

## 验证与回归风险

注册测试扩展:5D cache 的 RoPE 与 mRoPE 写入(bf16/fp8、多组 scale、heads 32/8 与 8/2、tokens 1/7/64/256),读回经 `gather_shuffle_5d_to_linear`,校验 q/K/V 对齐 reference 且未写其他 slot;NHD 旧用例在 main 与本分支均保持通过。风险:该 fused kernel 的 `k_scale`/`v_scale` 须为 CPU tensor(kernel 在 host 读取,docstring 注明),device tensor scale 不适用此路径;Qwen3-8B bf16 的 C64 点为两轮均值,噪声较大。

## 关联

前序 #44267(引入 fused QK-norm+RoPE+KV write);同批 #48405(vectorized_5d fp8 KV 写入融合量化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
