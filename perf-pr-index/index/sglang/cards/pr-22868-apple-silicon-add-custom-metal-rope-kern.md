# PR #22868: [Apple Silicon] Add custom Metal RoPE kernel with fused KV cache store

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @adityavaid |
| 合并 | 2026-05-29 |
| 分类 | Attention / Kernel(major) |
| Label | documentation, sgl-kernel, run-ci |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 含 Metal shader/C++ host、AOT registry、attention wrapper 接线等,4 个文件因超长未收入 pack |
| 链接 | https://github.com/sgl-project/sglang/pull/22868 |

## 一句话结论

为 Apple Silicon MLX 后端加入首个 AOT Metal kernel(融合 NeoX RoPE + KV pool scatter),RoPE 步从约 224us 降到约 127us(1.8x),但微基准显示仅对 LLaMA 形状有收益。

## 动机

Apple Silicon 上每次 kernel dispatch 因 Metal command buffer 开销引入约 1-4ms GPU idle(#22114 的 profile),减少 dispatch 数能直接改善 decode 延迟。batched decode 现状是两次 `mx.fast.rope`(Q、K 各一)加每请求的 KV-pool scatter;本 PR 将其合并为一个 C++ 入口、一次 Metal command buffer 的 AOT kernel。属 Apple Device Support 路线图(#19137)的一部分,基于 #23449 的脚手架。

## 优化点

构建期用 Apple 官方 `xcrun metal` + `xcrun metallib` 产出单个 `.metallib`(与 MLX 自身同一流水线),按 f16/bf16/f32 各导出三个专用 kernel:`rope_q`(Q 旋转)、`rope_k_pool`(K 旋转 + 把旋转后 K scatter 进 `k_pool[slot]`)、`v_to_pool`(V scatter),import 时 `register_library` 一次加载。相对朴素融合 kernel 的优化:3D 线程网格(pos.x=dim, pos.y=token, pos.z=head)消除 div/mod 热路径;function constant 特化 HEAD_DIM/头数/INV_DIM_LOG2_BASE/HEADS_PER_THREAD 使编译器按 pipeline 折叠常量;inline trig(`metal::exp2` + `fast::cos/sin`)免去预计算 cos/sin cache;threadgroup 自动取 4-8 SIMD groups;pool buffer 经 `copy_shared_buffer` zero-copy 捐赠(`copy_shared_buffer` 返回输入 buffer,同一显存分配);通过 `mlx::core::Primitive` 集成使 kernel 写入对 MLX lazy graph 可见。bf16、hd≥128、nk≥8、tokens≥256 时还有单次 rectangular dispatch 分支。slot=-1 sentinel 支持仅 RoPE 不写 pool 的模式。

## 关键改动

```text
sgl-kernel/csrc/metal/rope_pool_fused.metal: Metal shader(3 kernel + rect 版本,function constants)
sgl-kernel/csrc/metal/rope_pool_fused.cpp: nanobind host + RopePoolFused Primitive,3D grid dispatch 与 zero-copy 捐赠
python/sglang/srt/hardware_backend/mlx/aot.py: 新增 MLX AOT kernel registry(rope_dim==head_dim 且非 traditional 才启用)
python/sglang/srt/hardware_backend/mlx/kv_cache/attention_wrapper.py: _batched_decode 加 AOT 分支,否则回落 mx.fast.rope
python/sglang/srt/environ.py: 新增 SGLANG_MLX_USE_CUSTOM_ROPE(默认 False)与 SGLANG_RPF_N override
```

## 性能声明

- RoPE 步(#22114 profile,Apple Silicon dispatch 开销背景):约 224us(2×112us)→ 约 127us,1.8x。
- kernel 微基准(Apple M4 Pro 48GB):Qwen3-0.6B 维度(32 tokens,16 Q heads,2 KV heads,head_dim=64)下 MLX baseline 28.22us vs AOT fused 39.96us,即 0.71x——**Qwen 形状反而回退**。LLaMA-like bf16 BS=512(nq=32, nk=8, hd=128):88.13→64.23us(1.37x)。
- 大 batch bf16 sweep:Qwen3 形状全程 0.64-0.86x(回退);LLaMA 形状 BS=256 起为正收益(1.11x),BS=512 1.38x、BS=1024 1.36x。PR 明确承认"不是普遍加速",fused AOT 路径只在较大 LLaMA-like 形状(尤其大 bf16 batch)有帮助,Qwen-like 小 KV head 形状会退化,因此实现保持 guarded、可 A/B。

## 验证与回归风险

E2E server 测试(Qwen/Qwen3-0.6B):启动日志出现 "Custom RoPE kernel enabled",单请求推理文本正确,KV pool 42165 slots × 28 layers。Checklist 中"添加 unit tests"与"提供 benchmark 结果"两项未勾选(微基准为 PR 作者本机 focused profile)。风险:默认关闭(`SGLANG_MLX_USE_CUSTOM_ROPE=False`),启用需装 sgl-kernel Metal 扩展;仅支持 NeoX 非 traditional 且 rope_dim==head_dim 的模型,其余回落;Qwen-like 形状开启会负优化;CI Base/Extra 均失败。

## 关联

前序:#22114(dispatch 开销 profile)、#19137(Apple Device Support roadmap)、#23449(scaffolding)。内核灵感来自 MLX 的 rope.metal。同区域其他 Apple Silicon kernel PR 本卡未收录,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
