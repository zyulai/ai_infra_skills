# PR #35724: [diffusion] Enable LongCat breakable CUDA graphs

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-08-21 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | documentation, run-ci, diffusion, mergeable |
| Diff | pack diffstat 缺失;diff 为 allowlist/padder 注册、新 padder 文件、docs 与 skills 更新及测试补充 |
| 链接 | https://github.com/sgl-project/sglang/pull/35724 |

## 一句话结论

让 LongCat-Image 复用现有 breakable CUDA graph runner:512-token prompt 直通免 padding,B300 e2e −3.42%、denoise 阶段 −11.93%。

## 动机

LongCat 的 native eager DiT trace 在 attention 调用之间 launch 开销密集。现有的 breakable CUDA graph(BCG)runner 本可以把这些区域捕获为 31 个 graph segments,但 LongCat 此前未被 model 与 pipeline allowlist 收录,无法启用。另一个阻碍是 prompt shape:LongCat 总是把完整 512-token prompt body 送给 DiT,若走通用 prompt padding,bucket padding 只会制造更大且用不到的 graph signatures。

## 优化点

改动是接线而非新 runner:(1) `server_args.py` 的 BCG 支持 allowlist 加入 `meituan-longcat/longcat-image` 与 `LongCatImagePipelineConfig`;(2) 新增 `model_padders/longcat_image.py`——`is_longcat_image_transformer` 按类名与调用参数识别,`keep_longcat_prompt_shape` 对 call_kwargs 原样直通(固定 512-token body,不做 bucket padding),注册进 `prompt_padding` 的 model padder 体系,使短/长 raw prompt 复用同一个 graph signature;(3) CLI 文档给出 1024x1024 部署命令(`--enable-breakable-cuda-graph --warmup-resolutions 1024x1024 --enable-torch-compile false`),并刷新 checked-in 的 diffusion performance/profile skills。BCG 仍是 opt-in 且 resolution-specific,与 torch.compile、Cache-DiT 互斥(BCG 优先)。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/breakable_cuda_graph/model_padders/longcat_image.py: 新增,固定 512-token prompt 直通 padder
python/sglang/multimodal_gen/runtime/server_args/server_args.py: BCG allowlist 加入 LongCat 模型与 pipeline config
python/sglang/multimodal_gen/runtime/breakable_cuda_graph/prompt_padding.py: 注册 longcat_image padder
docs/docs/sglang-diffusion/api/cli.mdx: LongCat BCG 部署命令文档
```

## 性能声明

- B300(NVIDIA B300 SXM6 AC,native `--backend sglang`,BF16,1 GPU,1024x1024,50 steps,guidance 4.5,seed 42,prompt rewrite disabled,`quality=lossless`,`--performance-mode manual`):main eager e2e 7.422 / 7.421 / 7.445 s(mean 7.429 s,denoise 7.109 s)→ PR BCG 7.178 / 7.174 / 7.175 s(mean 7.175 s,denoise 6.261 s),e2e −3.42%、denoise 阶段 −11.93%。输出 PNG byte-exact(sha256 一致)。e2e 含 prompt encoding、DiT、VAE decode、落盘与 PNG sha256;不含 model loading、一次性 16.43 s BCG capture 与 benchmark warmup。
- H200(同配置,三种 prompt 长度):eager 177.3 / 177.0 / 177.2 ms 每步 vs BCG 173.1 / 173.3 / 173.1 ms(约 2.3% faster);三份 BCG 输出 pixel-exact;31-segment graph 捕获一次、三个 prompts 复用。default BCG text buckets 也恰好捕获一个 graph、无 bucket warning,8-step 输出 pixel-exact。
- 显存:H200 峰值约 33.4 GB(BCG)vs 31.7 GB(eager)。

## 验证与回归风险

`test_diffusion_bcg_padding.py` 22 passed,含新增 `test_longcat_keeps_its_fixed_512_token_prompt_shape`(断言 kwargs 原样返回、shape 不变)与 LongCat 模型/配置注册测试;所有改动文件 pre-commit 通过。风险:BCG 需为每个生产分辨率声明 `--warmup-resolutions`;一次性 capture 约 16.43 s 摊入启动;显存约 +1.7 GB;与 torch.compile/Cache-DiT 互斥。

## 关联

同机制已支持模型:Ideogram4、LTX-2/2.3、MiniMax-H3、Qwen-Image、SANA1.5、Z-Image、GLM-Image(正文列表,无 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
