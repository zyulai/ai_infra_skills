# PR #38655: Fix Nano Nemotron VL regressions

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38655](https://github.com/vllm-project/vllm/pull/38655) |
| 作者 | netanel-haber |
| 合并日期 | 2026-04-03 |
| labels | ready, multi-modality |
| 热度 | 👍 2 · 💬 4 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
集中修复 Nano Nemotron VL 三个近期回归:worker 启动时深拷贝 VllmConfig 失败、元数据热路径调 get_hf_processor 触发 Already borrowed、移除 hacky 的 feature_size_cache。

## 问题现象
三个近期回归,均影响 Nano Nemotron VL 多模态模型:
1. 自 #37467 起,`get_mamba_state_shape_from_config()` 在 worker 启动阶段运行,其中对完整 `VllmConfig` 的 deepcopy 会因存活参数报 `BasevLLMParameter.__new__` / `torch.nn.Parameter.__deepcopy__` 不匹配而失败(启动崩溃);
2. 元数据热路径调用 `get_hf_processor()`,触发 tokenizer 的 `RuntimeError: Already borrowed`(正文称可能由 #34789 暴露);
3. 动态分辨率图像的 feature size 依赖 `feature_size_cache`(以 `id(image)` 为键、取一次即删的 hacky 缓存),脆弱且与 `DynamicResolutionImageTiler` 耦合。

## 根因
正文披露根因:
1. mamba state helper 在启动期被调用,而它用 `copy.deepcopy(vllm_config)` 临时替换 hf_config;#37467 之后启动路径上存在活的量化参数对象,`torch.nn.Parameter.__deepcopy__` 与 `BasevLLMParameter.__new__` 签名不匹配,deepcopy 失败。
2. 元数据热路径(如 `is_dynamic_tiler`、`supports_video`、`audio_extractor` 等)都通过 `get_hf_processor()` 读取属性,反复取 processor 会触碰 tokenizer,在多线程/借阅检查下抛 `RuntimeError: Already borrowed`。
3. `feature_size_cache` 以 `id(image)` 存 feature size 并在消费时删除,依赖调用顺序假设,属 hacky 实现;processor 输出的 `out_mm_data` 里已有 `num_tokens_per_image` 可直接使用。

## 修复方案
1. `nano_nemotron_vl.py`:`get_mamba_state_shape_from_config` / `get_mamba_state_dtype_from_config` 弃用 `copy.deepcopy(vllm_config)`,改用 `vllm_config.with_hf_config(text_config)` 构造临时配置。
2. 元数据信息类改从 `get_hf_config()` 取属性:`is_dynamic_tiler` 用 `use_dynamic_resolution(config)` 判定、`supports_video`/`supports_audio`/`sound_config` 改为属性(不触碰 processor);`audio_length` 改为 `ParakeetExtractor` 的 staticmethod,接收 raw config;`parakeet.py` 的 `ExtractorConfig.from_hf_config` 补 `hop_length`(默认 160)。
3. 删除 `feature_size_cache` 及 `get_cached_feature_size`;动态分辨率路径改用 `out_mm_data["num_tokens_per_image"][item_idx]`。
4. `radio.py`:移除 `_video_embedder_loaded` 标记与报错,`forward_video` 改为断言 `temporal_patch_size > 1`。
5. processor(`transformers_utils/processors/nano_nemotron_vl.py`):`frames_indices` 为参差 list 时从 combined_inputs 摘出、按逐元素 `torch.as_tensor` 重建,避免 BatchFeature 校验问题;`supports_video` 固定 True、`video_token_id` 直接查表。
6. 测试侧:`tests/models/registry.py` 将 `NemotronH_Nano_VL_V2` 指向真实权重 `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` 并配 dummy overrides(触发动态分辨率与 conv3d);`tests/models/utils.py` 的 `dummy_hf_overrides` 支持 `text_config` 子配置覆盖。

关键 diff 摘录:
```diff
-        temp_vllm_config = copy.deepcopy(vllm_config)
-        temp_vllm_config.model_config.hf_config = text_config
+        temp_vllm_config = vllm_config.with_hf_config(text_config)
...
-    def is_dynamic_tiler(self) -> bool:
-        return self.get_hf_processor().dynamic_tiler is not None
+    def is_dynamic_tiler(self) -> bool:
+        return BaseNanoNemotronVLProcessor.use_dynamic_resolution(self.get_hf_config())
...
-            elif tiler := hf_processor.dynamic_tiler:
-                image = images.get(item_idx)
-                feature_size = tiler.get_cached_feature_size(image)
+            elif self.info.is_dynamic_tiler:
+                feature_size = out_mm_data["num_tokens_per_image"][item_idx]
```

## 影响与回归风险
- 影响窗口:正文指出回归 1 由 #37467 引入、回归 2 可能被 #34789 暴露,具体版本范围未写明。
- 回归面:`nano_nemotron_vl.py` 模型与 processor 代码、`parakeet.py` 音频长度计算、`radio.py` 视频嵌入器加载路径、测试基础设施。元数据能力判定从 processor 移到 config,行为前提是 config 信息足够(如 hop_length 有默认值兜底);`with_hf_config` 依赖 VllmConfig 提供该方法;`feature_size_cache` 移除后依赖 processor 输出包含 `num_tokens_per_image`。

## 关联
- [#37467](https://github.com/vllm-project/vllm/pull/37467)(正文引用:回归 1 的引入者,使 mamba state helper 在 worker 启动期运行)
- [#34789](https://github.com/vllm-project/vllm/pull/34789)(正文引用:回归 2 的 `Already borrowed` 错误疑似由其暴露)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
