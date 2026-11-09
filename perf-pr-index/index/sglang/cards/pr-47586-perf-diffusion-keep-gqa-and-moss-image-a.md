# PR #47586: [Perf] diffusion: keep GQA and MOSS-Image attention on cuDNN SDPA on Blackwell

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @niehen6174 |
| 合并 | 2026-11-01 |
| 分类 | Attention / Kernel(tier: major) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(素材未统计行数);diff 覆盖 sdpa backend 路由、moss_image 模型与路由单测 |
| 链接 | https://github.com/sgl-project/sglang/pull/47586 |

## 一句话结论

让 B200 上 diffusion 的 GQA(Cosmos3)与 MOSS-Image 注意力路由到 cuDNN SDPA,而非回退 FA4,e2e 快 13%。

## 动机

#33655 已使 B200(sm_100)上 dense diffusion attention 默认走 `DYNAMIC_CUDNN_SDPA`(cuDNN SDPA、回退 FA4),但两类模型没享受到:Cosmos3 是 GQA(32 query head / 8 KV head),`DynamicCudnnSDPAImpl._use_cudnn_sdpa` 在 query 与 key head 数不同时直接返回 False,所有 DiT 层都落到 FA4;MOSS-Image 的 DiT block 则硬编码 `default_attention_backend=FA`,平台默认对其不生效。作者在 B200 上跑 cookbook 命令时发现,强制 `--attention-backend torch_cudnn_sdpa` 使 Cosmos3-Nano T2V 720p 端到端快 13%。

## 优化点

只改路由,不改 kernel。`_use_cudnn_sdpa` 在 sm_10x 上当 query heads 是 KV heads 整数倍时不再因 GQA 而拒绝 cuDNN(`CudnnSDPAImpl.forward` 本就传 `enable_gqa=True`),短边 tile-size 规则保留;Pre-Blackwell GPU 与 head 分组不整除的情况仍走 FA。MOSS-Image 的 `_Block` 移除 `default_attention_backend=FA` 钉死项,使平台默认生效(Hopper 上仍是 FA,sm_10x 上是 `DYNAMIC_CUDNN_SDPA`),两个 text-refinement block 保留 FA,支持的后端集合不变。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/attention/backends/sdpa.py: _use_cudnn_sdpa 放行 sm_10x 的整除 GQA
python/sglang/multimodal_gen/runtime/models/dits/moss_image.py: _Block 移除 FA 钉死,改用平台默认后端
python/sglang/multimodal_gen/test/unit/test_dynamic_cudnn_sdpa_routing.py: 新增 GQA 路由用例
```

```python
# sdpa.py:_use_cudnn_sdpa(节选)
-        if query.shape[2] != key.shape[2]:
+        # sm_10x keeps GQA on cuDNN; CudnnSDPAImpl passes enable_gqa.
+        if query.shape[2] != key.shape[2] and (
+            not self._is_sm100 or query.shape[2] % key.shape[2]
+        ):
             return False
```

## 性能声明

单卡 B200,main(d2cf2371b0)与本分支同卡背靠背,cookbook 命令、不加 attention flag;e2e 为 HTTP 客户端延迟中位数(视频 2 次、图像 5 次),denoise 取自 server stage log:

| Workload | main | 本 PR | 变化 | denoise(main→PR) |
| --- | --- | --- | --- | --- |
| Cosmos3-Nano T2V 1280x720, 189 frames | 112.32 s | 97.55 s | -13.1% | 102.37 → 87.67 s |
| MOSS-Image-Base 1024x1024, 50 steps, CFG 3 + uncond skip | 4.772 s | 4.540 s | -4.9% | 4.362 → 4.157 s |
| MOSS-Image-Base 1024x1024, 50 steps, ordinary CFG 3 | 4.621 s | 4.412 s | -4.5% | 4.422 → 4.206 s |
| MiniMax-H3 T2VA 960x544, 124 frames(MHA 对照组) | 27.714 s | 27.707 s | 0.0% | 25.15 → 25.18 s |

4 卡(CFG 2 x Ulysses 2,`--attention-backend fa` vs `torch_cudnn_sdpa`):Cosmos3 T2V 29.40 s → 26.39 s(-10%),I2V 29.42 s → 26.65 s(-9%)。正文另提到强制 cuDNN 使 Cosmos3 denoise 快 14%。

## 验证与回归风险

新增 `test_dynamic_cudnn_sdpa_routing.py` 用例:GQA 的 self/cross-attention 形状在 sm_100 上留在 cuDNN;head 分组不整除与 pre-Blackwell GQA 走 FA;与 `test_dynamic_cudnn_sdpa_selection.py` 共 8 个用例在 B200 通过。kernel 本身是 Blackwell MHA 已在用的 cuDNN SDPA(`CudnnSDPAImpl`,FA4 兜底),仅路由变化;正文注明 cuDNN 与 FA4 舍入不同,输出与 FA4 路径不 bit-identical。CI 状态:PR Test (Base) 通过,Extra 与 AMD ROCm 10 失败(素材所示)。

## 关联

前序:#33655(引入 `DYNAMIC_CUDNN_SDPA` Blackwell 默认)。同属 diffusion attention 后端路由优化;其他同分类 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
