# PR #17420: [VLM] Optimize get_rope_index for GLM4v

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-02-01 |
| 分类 | Other(tier: major) |
| Label | performance, Multi-modal, run-ci, vlm |
| Diff | diffstat 缺失(素材未提供 +N/-N);pack diff 片段含 2 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/17420 |

## 一句话结论

重写 GLM4v 的 get_rope_index 热路径,长 token 长度下提速 12%~600%。

## 动机

`get_rope_index_glm4v` 构建 multimodal mrope position ids 时大量使用 `.item()` 同步取值、逐 token append、未指定 device 的 `torch.arange` 与 `flatten`,长 token 长度下 CPU 开销随序列长度显著增长,对长上下文多模态输入(GLM4v)值得优化;同时补一个可复跑的 benchmark 脚本量化收益。

## 优化点

`get_rope_index_glm4v` 热路径重写:用 `int(t)/int(h)/int(w)` 替代重复的 `.item()`;`input_token_type` 预分配为 `[""] * len(input_tokens)` 并按索引单趟赋值,替代逐个 append;`torch.arange` 显式传 `device=position_ids.device` 避免隐式 H2D 拷贝;`expand + reshape` 替代 `flatten`;text 分支改为一次性 in-place expand;`position_ids.max(0)[0].max(-1)[0]` 改为 `amax`;`mrope_position_deltas` 最后一次性建 tensor。无多模态输入的 fallback 分支做同样的 amax/arange 清理。`get_rope_index`(Qwen 路径)也做了同类 `.item()`/arange device/amax 清理。

## 关键改动

```text
python/sglang/srt/layers/rotary_embedding.py: get_rope_index_glm4v/get_rope_index 热路径重写
benchmark/bench_rope/benchmark_rope_index.py: 新增 GLM4V rope index benchmark 脚本
```

## 性能声明

正文宣称 "Speedup **12%** to **600%**(long token length) for get_rope_index for GLM4v",lmms_evals no drop。附带的 benchmark(GLM4V、cuda、B=1、100 iters、1 图 256 patch tokens)分别给出 PR 与 main 两次运行,长序列处 Multimodal 分支 mean 对比:T=65536 0.11286658s(main)→ 0.01308443s(PR),T=131072 0.16182978s → 0.02458367s;T=1024 时 PR 的 Multimodal mean 0.00073278s 略高于 main 的 0.00057760s。输出中的 "Fallback Speedup over Multimodal"(10.66x@1024 至 336x@131072)是 PR 版本内 fallback 分支与 multimodal 分支的比值,并非新旧版本对比。

## 验证与回归风险

PR 未新增单测(检查项未勾选);正确性依据是 benchmark 脚本对返回 shape 的断言(position_ids (3,B,T)、deltas (B,1))与 "lmms_evals no drop"。改动同时触碰 Qwen 路径的 `get_rope_index`,flatten→reshape 与 amax 的等价性需依赖现有 CI;mrope_position_deltas 的 dtype/device 行为预期不变。

## 关联

同文件前序 PR #11722(引入 mrope Triton kernel)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
