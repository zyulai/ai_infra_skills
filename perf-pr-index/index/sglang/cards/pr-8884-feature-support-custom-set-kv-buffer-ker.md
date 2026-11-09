# PR #8884: [Feature] Support custom set kv buffer kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DarkSharpness |
| 合并 | 2025-08-12 |
| 分类 | Attention / Kernel(major) |
| Label | (none) |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/8884 |

## 一句话结论

新增融合的 `store_kv_cache` CUDA kernel,一次写入 K/V 两级 cache,LLaMA 3.1 8B 小 batch e2e 约 +3%。

## 动机

nsys profile 显示在 H100 上 `set_kv_cache` 会带来少量开销:原有的索引赋值(`k_cache[loc] = k; v_cache[loc] = v`)是两次独立的 scatter 写,存在可消除的 kernel 启动与访存开销。作者据此写了一个把"写入 key cache 与 value cache"融合起来的 kernel。

## 优化点

新增 `sgl-kernel/csrc/memory/store.cu`,以 `torch.ops.sgl_kernel.store_kv_cache` 暴露 `store_kv_cache(k_cache, v_cache, out_loc, k, v)`,把 K/V 两路 scatter 融合为单个 kernel。实现按"每 warp 每轮拷贝字节数"分两个变体:当每 token 的 K/V 尺寸(字节)能被 256 整除时用 `store_kv_cache_256x1`(一个 warp 同时搬 K 和 V,各 lane 覆盖 `uint64_t` 粒度的连续 8B),否则能被 128 整除时用 `store_kv_cache_128x2`(warp 内 16 个 lane 搬 K、16 个 lane 搬 V);均不满足则报错。warp 按 `out_loc` 中的目的位置定位 cache 行,以 `uint64_t` 为单位做连续拷贝;host 侧把 stride 统一折算成 8B 单元并做 shape/stride/contiguity 校验。Python 侧 `sgl_kernel.memory.set_kv_buffer_kernel` 封装该 op,RuntimeError 时回退到原 torch 索引赋值,提供 `fallback` 开关。

## 关键改动

```text
sgl-kernel/csrc/memory/store.cu: store_kv_cache 融合 kernel(256x1 / 128x2 变体)
sgl-kernel/python/sgl_kernel/memory.py: set_kv_buffer_kernel 封装 + torch fallback
sgl-kernel/csrc/common_extension.cc 与 sgl_kernel_ops.h: 注册 store_kv_cache op
sgl-kernel/CMakeLists.txt: 新增源文件
sgl-kernel/python/sgl_kernel/__init__.py: 导出 set_kv_buffer_kernel
```

注:pack diff 仅包含 sgl-kernel 侧文件,未见 sglang 主仓库中 `set_kv_buffer` 调用点的集成改动。

## 性能声明

PR 正文声明:对 LLaMA 3.1 8B、H100、小 batch size 场景,带来约 3% 的 e2e 性能提升;并指出对于大 TP(每 rank KVCache 更小,如 70B 模型 TP=8 时每 GPU 仅 1 个 head)的场景收益应更大。未给出详细 benchmark 表与 workload 参数;正文附 nsys profile 前后对比截图(素材中为图片链接)。除此之外 PR 未注明其他测试条件。

## 验证与回归风险

PR 正文无 accuracy 测试结果,Checklist 各项均未勾选,pack diff 中未见新增单测。kernel 有较多硬性约束(全部张量 CUDA、k/v cache 尺寸与 stride 一致、最后一维连续且尺寸相同、`out_loc` 为 1D 连续 int32/int64、每 token K/V 字节数至少能被 128 整除),不满足时直接 TORCH_CHECK 失败;Python 封装以 RuntimeError 捕获回退到 torch 实现,降低了破坏性,但 fallback 意味着失败场景会退回性能而非报错。kernel 以 `uint64_t` 粒度拷贝,依赖 8B 对齐的 stride 折算,非常规 dtype/布局路径依赖 host 侧校验兜底。仅涉及 KV 写入路径,不影响注意力计算。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
