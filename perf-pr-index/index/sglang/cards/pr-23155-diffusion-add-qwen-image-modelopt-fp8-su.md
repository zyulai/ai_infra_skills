# PR #23155: [Diffusion] Add Qwen Image ModelOpt FP8 support

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-05-03 |
| 分类 | Quantization(major) |
| Label | documentation, high priority, quant, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/23155 |

## 一句话结论

让 Qwen Image / Qwen Image Edit DiT 支持 ModelOpt FP8 权重,H100 上 E2E 快 10.4-10.5%、Denoising 快 11.5-13.1%。

## 动机

SGLang 扩散运行时此前已支持 FLUX.1/FLUX.2/Wan2.2 的 ModelOpt FP8 transformer override,但 Qwen Image 系列未接入:其 DiT 权重全是非 quant-aware 的 `nn.Linear`/diffusers `FeedForward`,无法挂载 FP8 checkpoint。直接按默认 profile 量化又会产生严重画质回退(旧 FP8 default 出现明显暗化/模糊),需要一个经过消融验证的 BF16 fallback 档案与正确的转换器写入顺序。

## 优化点

三件事:(1) 把 `QwenImageTransformer2DModel` 全部大矩阵投影改为 quant-aware——attention 的 to_q/to_k/to_v、add_q/add_k/add_v、`img_in`/`txt_in`/`proj_out` 换成带 `quant_config` 的 `ReplicatedLinear`,并用 `QwenImageFeedForward`/`QwenImageGELU` 替换 diffusers `FeedForward` 使 `img_mlp`/`txt_mlp` 的 net.0/net.2 可量化、前缀与 checkpoint 对齐;(2) 转换器新增 qwen-image BF16 fallback 档案(`img_in`、`txt_in`、timestep embedder、`norm_out.linear`、`proj_out`、`img_mod`/`txt_mod`、`img_mlp.net.2`),其中 `img_mlp.net.2` 是画质消融后加入的;并修复写入顺序 bug——显式 BF16 fallback 张量必须先于 ModelOpt ignore-preservation 写出,否则 stats 报了 fallback 而输出 checkpoint 仍保留源 FP8 张量,导致严重画质回退;(3) 在 lmsys HF 发布干净的 transformer override 权重并文档化,B200 CI 增加两个用例。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py: DiT 各投影改为 quant-aware ReplicatedLinear,新增 QwenImageFeedForward
python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py: qwen-image fallback preset + fallback 写出顺序修复
python/sglang/multimodal_gen/test/server/gpu_cases.py / testcase_configs.py: B200 CI 新增 t2i/ti2i 两个 FP8 用例
docs/diffusion/quantization.md: 校验矩阵从 6 个 checkpoint 扩到 8 个
```

```python
# build_modelopt_fp8_transformer.py: fallback 必须先于 ignore-preservation 写出
if name in fallback_tensors:
    shard_tensors[name] = fallback_tensors[name]
    continue
if name.endswith(".weight") and is_ignored_by_modelopt(
    name, ignore_patterns
):
    preserved_ignored_weight_count += 1
    continue
```

## 性能声明

均在 H100(rank0,`CUDA_VISIBLE_DEVICES=0`)用原生 `sglang generate --backend=sglang --warmup` 测得,BF16 vs 修复后 FP8:

- Qwen Image 1024x1024、50 steps:E2E 13589.20 ms → 12159.39 ms(10.5% faster);Denoising 12928.76 ms → 11437.40 ms(11.5% faster)。画质目测与 BF16 一致(旧 FP8 default 严重暗化/模糊)。
- Qwen Image Edit 512x512、8 steps:E2E 6791.97 ms → 6085.03 ms(10.4% faster);Denoising 5204.32 ms → 4524.01 ms(13.1% faster)。

转换器统计(两个 checkpoint 相同):660 个量化权重、1320 个 scale 张量、186 个 BF16 fallback 权重、3 个输出 shard。

## 验证与回归风险

验证在 H100 上做 BF16-vs-FP8 生成对比与 benchmark,`compileall`/py_compile/black/ruff 通过;Qwen Image Edit 有 TI2I smoke。回归风险主要在两处:(1) DiT 的 `img_in`/`txt_in`/`proj_out` 等改返回 tuple(`output, _`),非量化路径的行为依赖 `ReplicatedLinear` 兼容;(2) 转换器写入顺序修复只影响新转换的 checkpoint,旧 FP8 checkpoint(错误保留源 FP8 张量)不可复用。B200 用例仅作为 CI 冒烟,正文未给 B200 数字。用 `--model-type qwen-image` 强制该 fallback 档案,或由 `QwenImageTransformer2DModel` 类名自动识别。

## 关联

同族 ModelOpt FP8 checkpoint:FLUX.1-dev、FLUX.2-dev、Wan2.2(已在支持矩阵中,共享同一转换器与 fallback 机制);同文档矩阵还有 NVFP4 系列。pack 未列出具体前序 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
