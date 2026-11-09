# PR #33115: [ModelOpt FP4] Support online MoE weight quantization

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zianglih |
| 合并 | 2026-08-06 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, quant, speculative-decoding, blackwell, run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);5 files dropped |
| 链接 | https://github.com/sgl-project/sglang/pull/33115 |

## 一句话结论

把 `modelopt_fp4` 扩展为支持在线 MoE 专家权重 NVFP4 量化(BF16/FP16/FP8 源、per-tensor 激活 scale),复用 nvfp4_online loader 并理顺 per-token/per-tensor 双契约与 draft 继承。

## 动机

此前在线 NVFP4 只存在于 `nvfp4_online`(per-token 激活 scale)名下,而 #31382 把 per-tensor 激活行为(CuTe DSL + FlashInfer A2A/DeepEP)错误地也标成了 `nvfp4_online`,契约混乱。需要把 per-tensor 在线转换归位到 `modelopt_fp4`,提供共享在线 weight loader,并明确 draft 模型量化的继承/显式覆盖规则。

## 优化点

(1) 新增 `_ModelOptFp4OnlineConfig` 适配器:继承 `NvFp4OnlineConfig` 但 `is_nvfp4_online=False`、per-token 关闭(`get_name()` 返回 `modelopt_fp4`);`ModelOptFp4Config.from_config` 遇 `quant_method=fp8` 直接返回在线配置,让 literal FP8 checkpoint 的 MoE 专家在加载期转换。(2) 加载逻辑:非 serialized checkpoint 下 linear/lm_head 走 `UnquantizedLinearMethod`(保持源精度),`FusedMoE` 走 `ModelOptNvFp4FusedMoEMethod`;FP8 源先按 checkpoint scale 反量化到 BF16 再量化 NVFP4,且 FP8 激活 scale 不描述 requant 后的输入,loader 直接跳过 `input_scale`。(3) `get_online_weight_loader` 重构为 classmethod,注入 `fp8_dequantizer`,`make_nvfp4_online_weight_loader` 供两个量化名复用;缺失的 per-tensor FP32 激活 scale 默认 1.0,checkpoint 值覆盖;exact FP4 量化数学用 `temp_set_env` 限定在加载期。(4) Draft:`ModelConfig.is_draft_quantization_explicit` 区分显式指定与继承;serialized 目标 checkpoint 中 BF16 MTP 专家被 exclude 时,显式 `--speculative-draft-model-quantization modelopt_fp4` 让 draft 走在线转换;Qwen3.5 MTP 区分 serialized 与 online `modelopt_fp4`,DeepSeek/GLM/Bailing/Kimi 的 NextN/EAGLE 构造器维持 opt-out。(5) 普通在线 `modelopt_fp4` 改用 `DefaultModelLoader`,显式 ModelOpt restore/save/export 工作流保留。

## 关键改动

```text
python/sglang/srt/layers/quantization/nvfp4_online.py: _ModelOptFp4OnlineConfig 适配器 + loader classmethod 化与 fp8_dequantizer 注入
python/sglang/srt/layers/quantization/modelopt_quant.py: ModelOptFp4Config 契约文档、fp8 源路由、draft 显式量化辅助
python/sglang/srt/model_loader/weight_utils.py: _resolve_explicit_draft_quant_config 处理 excluded MTP experts
python/sglang/srt/model_loader/loader.py: 在线 modelopt_fp4 走 DefaultModelLoader,FP4 数学限定加载期
python/sglang/srt/models/qwen3_5_mtp.py: serialized 与 online modelopt_fp4 区分
```

## 性能声明

PR 明确 "Not run; this PR adds load-time conversion and quantization routing, not a new kernel",未提供性能数字。B200 focused 单测 27 passed + 15 subtests(`test_modelopt_nvfp4.py` + `test_modelopt_loader.py`;8x B200,PyTorch 2.11.0+cu130,CUDA 13.0);模型级 accuracy 未在本地运行。

## 验证与回归风险

依赖 B200 单测覆盖 loader/量化路由;CI Extra 通过、Base 失败(x)。风险:契约变更有破坏面——`nvfp4_online` 不再接受 per-tensor/DeepEP 路径(报错信息指向 `modelopt_fp4`),`ModelOptNvFp4OnlineFusedMoEMethod` 仅允许 flashinfer_trtllm 系;`modelopt_fp4` 的兼容白名单从 `["modelopt"]` 扩为 `["modelopt", "fp8"]`;draft 量化继承行为变化影响投机解码部署。

## 关联

正文明确:#26083(初始在线 NVFP4)、#31382(契约错误来源)、#28354(后续 CuTe DSL per-token 集成,复用本 PR 的共享在线 loader)。
