# PR #14292: [VLM] Introduce Cache for positional embedding ids for Qwen-VL family

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-12-04 |
| 分类 | Model-specific Fast Path |
| Label | performance, Multi-modal, run-ci, vlm |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);diff 含 3 个文件(qwen2_5_vl.py、qwen3_vl.py、models/utils.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/14292 |

## 一句话结论

为 Qwen-VL 系 ViT 的 rot_pos_emb 索引计算加 lru_cache 并改用 numpy,索引计算路径提速,VLM 预处理 E2E 改善 7%。

## 动机

Qwen2.5-VL / Qwen3-VL 的 vision tower 每次前向都要在 `rot_pos_emb` 里为每个图像 grid 逐个构造 hpos_ids/wpos_ids:对每个 `(t, h, w)` 做 torch `arange`、`reshape`、`permute`、`flatten`、`stack`、`repeat` 一整套 CPU 侧张量操作,再拼成位置 id。VLM 高并发下每请求都重复同样的纯函数计算(h、w、spatial_merge_size 相同的结果完全一致),是 prefill 预处理路径上的可缓存热点。PR 正文未附 profile 数据,直接给出缓存 + numpy 化的方案。

## 优化点

新增 `RotaryPosMixin` mixin 类,核心是 `@staticmethod @lru_cache(maxsize=1024)` 的 `rot_pos_ids(h, w, spatial_merge_size)`:用 numpy(`np.broadcast_to`/`reshape`/`transpose`/`flatten`/`np.stack`)计算 hpos/wpos id,一次算好后 `torch.from_numpy` 转换并缓存,键为 `(h, w, spatial_merge_size)`;t>1 时仅在调用侧 `base.repeat(t, 1)`。`Qwen2_5_VisionTransformer` 与 `Qwen3VLMoeVisionModel` 改为继承该 mixin,`rot_pos_emb` 中十几行的 per-grid torch 构造逻辑各替换为一次 `self.rot_pos_ids(h, w, self.spatial_merge_size)` 调用。numpy 标量路径还避开了 `grid_thw[i].tolist()` 的 GPU→CPU 同步(改为直接迭代 tensor 行)。作者称 numpy 化本身带来 "E2E improved 7%" 的额外提速。

## 关键改动

```text
python/sglang/srt/models/utils.py: 新增 RotaryPosMixin.rot_pos_ids(lru_cache 1024 + numpy 实现)
python/sglang/srt/models/qwen2_5_vl.py: VisionTransformer 继承 mixin,rot_pos_emb 改调缓存
python/sglang/srt/models/qwen3_vl.py: Qwen3VLMoeVisionModel 同样接入
```

关键机制(纯函数缓存键 + numpy 计算):

```python
@staticmethod
@lru_cache(maxsize=1024)
def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
    hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
    hpos_ids = hpos_ids.reshape(h_div, spatial_merge_size, w_div,
                                spatial_merge_size).transpose(0, 2, 1, 3).flatten()
    ...
    return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))
```

## 性能声明

PR 正文 bench(bench_serving image 数据集,500 prompts,每 prompt 1 张 jpeg 1120x700,random input 100 / output 1,max-concurrency 20;Qwen3-VL-8B-Instruct,TP=2,SGLANG_USE_CUDA_IPC_TRANSPORT=1 等 env):自述 "TTFT drop 2%",Mean TTFT 1950.93 → 1926.97 ms,Total token throughput 8937.61 → 9047.44 tok/s;另自述索引计算 numpy 化带来 "E2E improved 7%"。PR 未注明该 7% 的具体测试条件(未附对应表格),数字以正文原话为准。

## 验证与回归风险

PR 未注明新增单元测试与精度数据(checklist 各项未勾选);正确性依赖 lru_cache 键覆盖 `(h, w, spatial_merge_size)` 全部输入——均为 int/tensor,函数内已做 tensor→int 归一,但键的哈希基于归一前参数,tensor 参数逐次新对象会导致缓存 miss(不影响正确性,只影响命中率);`maxsize=1024` 在极端多分辨率负载下可能逐出。风险面:仅影响 Qwen2.5-VL / Qwen3-VL 的位置 id 构造,输出应与旧实现逐位一致(numpy transpose 与 torch permute 语义相同);其他 VLM 模型不受影响。

## 关联

同区域(Qwen-VL 预处理优化):#12240(video preprocess)、#13487、#7724(pybase64)、#17420(get-rope-index for GLM4V)、#18124(multitoken processor safetensors load)。同作者前作:#14123(MmItemMemoryPool CUDA IPC)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
