# PR #57101: [Perf] Optimize Molmo2 pooling preparation

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Levius-Fubuki |
| 合并 | 2026-10-21 |
| 分类 | Model-specific Fast Path(major) |
| Label | performance, ready, multi-modality, pooling |
| Diff | 素材未提供 diffstat 数字;pack diff 收录 tests 与模型实现两个文件的完整改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/57101 |

## 一句话结论

Molmo2 视觉 pooling 准备阶段用全局行索引 `index_select` 取代 tiled 张量 + 多维 advanced indexing,微基准几何平均提速 1.29x。

## 动机

Molmo2 的 pooling 准备路径要先 materialize 一个按 batch 平铺(batch-index tile)的索引张量,再用多维 advanced indexing 从 image features 中取数,之后才做 invalid patch 掩码和 attention query 的 sum/mean 归约。这个 tile + 多维索引的组合引入了不必要的张量物化和 gather 开销,在多 crop、多 patch 的图文/视频输入下被放大。

## 优化点

改动保持在原生 PyTorch 内:把 batch-local 的 patch 索引直接加上 batch offset 展平成全局行索引,用一次 `torch.index_select` 完成gather,再用 `mul_` 就地掩掉无效 entry,后续的 masked/ordinary sum-mean query 归约逻辑不变。没有引入 Triton kernel 或 custom op,因此没有额外 JIT 预热开销,也不需要平台相关的 fallback 分支。同时用 `clamp_min(0)` 复用无效索引位置,减少中间张量数量。

## 关键改动

```text
vllm/model_executor/models/molmo2.py: 新增 _prepare_molmo2_pooling 辅助函数,forward 中以 index_select 路径替换原 batch tile + 多维 advanced indexing 实现
tests/models/multimodal/processing/test_molmo2.py: 池化验证合并进既有处理测试套件,新增参数化用例覆盖 masked/普通均值与无效组过滤
```

关键片段(molmo2.py):

```python
flat_indices = token_pooling.clamp_min(0)
batch_offsets = torch.arange(batch_size, ...).view(batch_size, 1, 1)
flat_indices.add_(batch_offsets * (num_crops * num_patches))
to_pool = torch.index_select(
    image_features.reshape(-1, dim), 0, flat_indices.reshape(-1),
)
to_pool = to_pool.reshape(-1, pool_size, dim)
to_pool.mul_(valid.reshape(-1, pool_size, 1))
```

## 性能声明

RTX 4090、PyTorch 2.12.1+cu130、Triton 3.7.1、`VLLM_TARGET_DEVICE=cpu` 下的 pooling 准备 microbenchmark(历史数据,本 follow-up 未重跑):跨 24 个 image/video 行,所有用例均提升,几何平均 1.29x,范围 1.06x–1.50x。代表性数据:B=1 image K=4 FP16 550 groups 165.55→118.42 µs(1.40x);B=1 video K=9 FP16 256 groups 1.50x;B=1 video K=9 BF16 4096 groups 1.07x;B=2 video K=9 FP16 4096 groups 1.07x。注意这是对 pooling preparation 的隔离测量,非全模型 E2E 延迟。

## 验证与回归风险

验证集中在 `tests/models/multimodal/processing/test_molmo2.py`:用轻量替身替代 encoder 与 learned 层直接测 `Molmo2VisionBackbone.forward`,参数化覆盖 masked 与普通 mean、部分/全部无效组、重复索引、跨 batch 索引与无效组过滤。风险面:仅触及 Molmo2 模型的 pooling 路径,不影响其他模型;后续 pool/投影/ token 过滤行为不变。需注意本 PR 为 review follow-up,CUDA 与全模型 benchmark 未重跑,验证为 agent-run。

## 关联

同区域(多模态/pooling 性能)相关:#57214(pooling 路径消除 GPU seq lens 阻塞同步)、#60500(去掉图像 rescale/normalize 冗余)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
