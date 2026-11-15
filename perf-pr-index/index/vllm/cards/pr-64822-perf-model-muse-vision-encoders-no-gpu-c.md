# PR #64822: [Perf][Model] Muse vision encoders: no GPU<->CPU syncs, build Glimmer's vision prep once per call

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @subhashpolisetti |
| 合并 | 2026-11-13 |
| 分类 | Other(tier: major) |
| Label | ready, multi-modality |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/64822 |

## 一句话结论

Muse Glimmer 与 Muse Spark(NaViT)vision encoder 把窗口布局计算搬到 host(numpy + `async_tensor_h2d`),Glimmer 的 vision prep 从每组一次改为每 call 一次:126 次 encoder 调用 GPU<->CPU sync 由 Glimmer 879 / NaViT 817 降到 0,32 帧视频编码快 1.15–1.22x,输出与 main bit-identical。

## 动机

Glimmer 的 `_get_sparse_permutation` 在 GPU 上构建 32x32 窗口布局,然后做 boolean-mask select 和 `.tolist()`——每图像/帧组两次 sync;NaViT 的 `_get_sparse_attention_perm` 有同样的 mask select,外加每 call 一次 `.tolist()` 和 `_seqlens_to_cu_seqlens` 里 blocking 的 `torch.tensor(list, device=cuda)`。这些只依赖已在 CPU 上的 patch grid。`VLLM_GPU_SYNC_CHECK=error`(CI test image 默认设置,#43107)下每个 Glimmer 图像/视频请求都会报错;该检查默认关闭,用户看不到错误,但 sync 每个 request 都在发生。

## 优化点

- 布局改为 numpy 上以相同运算计算,经 `async_tensor_h2d` 拷贝上 GPU;Glimmer 按 distinct grid 缓存布局(`layouts` dict),`ln_pre`、RoPE、window permutation、`ln_post` 与 pixel shuffle 对整个 call 只跑一次(patch projection 仍每组一次 matmul)。
- NaViT:`_get_sparse_attention_perm` 与 `_seqlens_to_cu_seqlens` 改 numpy/cumsum + `async_tensor_h2d`,`max_seqlen` 保持留在 CPU。
- Tradeoff:patch projection 不合并为整个 call 一次 matmul——单 matmul 在 L4 上改变了 126 次调用中 48 次的输出位(H100 无变化),CI 硬件上的 bit-exact 测试可能抓不到,故用 `test_glimmer_vision_encoder_projects_patches_per_group` 钉住调用次数;position embedding 仍在 GPU 上按 distinct grid 插值,使其临时显存不超过 main。
- 无 config、flag 或 API 改动。

## 关键改动

```text
vllm/model_executor/models/muse_glimmer.py: 布局计算改 numpy(_get_pos_emb_layout/_get_pixel_shuffle_order/_get_sparse_permutation),forward 按 distinct grid 建 layouts 并整 call 单次 ln_pre/RoPE/permutation/ln_post/pixel shuffle
vllm/models/muse_spark/common/vision.py: _get_sparse_attention_perm 与 _seqlens_to_cu_seqlens 改 numpy + async_tensor_h2d
tests/models/multimodal/generation/test_muse_vision_encoders.py(新增): 布局正确性、batch invariance、无 GPU sync 等测试
```

```python
for grid in grids:
    grid_height, grid_width = grid
    if grid not in layouts:
        rows, cols = np.divmod(np.arange(grid_height * grid_width), grid_width)
        layouts[grid] = (
            rows + 1,
            cols + 1,
            self._get_pixel_shuffle_order(grid_height, grid_width),
            self._get_sparse_permutation(grid_height, grid_width)
            if has_sparse_layers
            else None,
        )
    height, width, shuffle_order, sparse_layout = layouts[grid]
```

## 性能声明

- 无 GPU<->CPU sync:Glimmer 879 → 0,NaViT 817 → 0(126 次 encoder 调用)。
- Glimmer 32 帧视频编码快 1.15–1.22x:`vllm bench mm-processor`(32-frame 720p)encoder forward 87.5 ms vs main 106.2 / 107.8 ms(1.22x);standalone script harness 给出 83.0 vs 95.1 ms(1.15x),两个 harness 的 host 与输入不同,作者未隔离各自贡献。
- 图像与 serving 无变化:126 次 encoder 调用逐类汇总(one image 1.000x;2–4 images 1.011x;Poisson-sized 1.002x;size extremes 0.999x;video 8/32/96 frames 1.121x;bursts 8k/16k tokens 1.021x;total 1.015x),噪声约 1%,作者把图像行读作无变化;VisionArena p99 TTFT 509 ms vs main 507/506 ms(1,000 请求)。Prep launch 数:单图 94 → 20,32 帧视频 1,490 → 37。
- 输出在全部 126 次调用与 main bit-identical(真实 Glimmer vision 权重与随机像素,H100/FA3 与 L4/FA2);峰值 GPU 显存从不高于 main(最差比率 0.998),encoder profiling budget 不增长。
- Serving(`vllm bench serve`,`RedHatAI/Muse-Glimmer-30B-FP8-block`,H100,main → PR → main):VisionArena c1 1,000 请求 req/s 3.52/3.51 → 3.53、median TTFT 156/159 → 157 ms;32 帧视频 c1 req/s 0.65/0.68 → 0.67;4 图/请求 c16 median TTFT 6939/7111 → 5527 ms。作者明确不声称 serving 收益:4 图差距远大于约 1% 的 encoder 变化所能解释(更早一轮同 workload main 单独 6209/5156 ms),视频行只有 30 个请求。

## 验证与回归风险

- `pytest tests/models/multimodal/test_muse_vision_encoders.py`(L4;文件现移至 `tests/models/multimodal/generation/`,不在 core set):main 上 31 failed / 5 passed(28 个失败是调用本 PR 新增的 helper/签名;`_get_sparse_permutation` 报 `RuntimeError: GPU<->CPU sync detected`),本 PR 36 passed;更早版本(每 call 一次 patch-projection matmul)在 `test_glimmer_vision_encoder_projects_patches_per_group` 失败(`assert [92] == [60, 16, 16]`),故保持每组一次 matmul 并钉住调用次数。
- 引擎级测试(`test_glimmer_engine_has_no_gpu_sync`)以 dummy weights 发图像+视频请求;这些测试尚未在 H200 CI lane 上跑过。
- 限制:Muse Spark checkpoint 是 gated,NaViT 用独立 12 层 encoder 验证(bit-identical、无 sync),不是完整模型;真实权重 bit-identity 与 benchmark 在 CI 外跑。
- 风险:CI 覆盖仅上述测试;encoder 对比脚本不属于本 PR。

## 关联

- #59470 与 #61448(也改 `muse_glimmer.py`,但只涉及 language model 与 RMSNorm);#43107(CI test image 设置 `VLLM_GPU_SYNC_CHECK`)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
