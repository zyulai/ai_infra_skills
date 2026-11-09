# PR #63001: [Bugfix][Quantization][Spec Decode] Load ModelOpt mixed-precision DFlash and DSpark drafts

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#63001](https://github.com/vllm-project/vllm/pull/63001) |
| 作者 | sragrawal |
| 合并日期 | 2026-10-24 |
| labels | bug, documentation, speculative-decoding, ready, ci/build, qwen, quantization, verified, dflash |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
ModelOpt `MIXED_PRECISION` 嵌套元数据被拒、草稿模型按 checkpoint 名找不到量化层、FNUZ GPU 缺 per-tensor FP8 归一化;三处分别修复。

## 问题现象
两类失败。其一,加载嵌套 `MIXED_PRECISION` 元数据的检查点时报 `Value error, Unknown ModelOpt quant algo: MIXED_PRECISION`(混合 Llama 检查点经 `LLM` 加载时 config 转换直接失败)。其二,修复配置后,DSpark 混合精度草稿加载时报 `ValueError: There is no module or parameter named 'fc.weight_scale' in Qwen3DSparkModel. The available parameters belonging to fc (ReplicatedLinear) are: {'fc.weight'}`,草稿以未量化投影构建后无法加载量化权重。其三,AMD CI(MI300)暴露 per-tensor FP8 加载缺陷:引擎初始化在 `input.dtype == weight.dtype` 失败——加载器保留 FN 权重而激活量化器选了 FNUZ。

## 根因
正文披露:
1. config 转换在现有 `modelopt_mixed` 后端能加载之前就拒绝嵌套 `MIXED_PRECISION` 元数据。
2. 草稿加载器用运行时名(如 `model.fc`)查找 per-layer 量化设置,而检查点把该层命名为 `fc`,查不到就构建未量化投影,随后加载 `fc.weight_scale` 失败。
3. per-tensor FP8 权重与静态 input scale 未做 FN→FNUZ 归一化(FNUZ GPU 如 AMD MI300 需要)。

## 修复方案
- `vllm/transformers_utils/model_arch_config_convertor.py`:`_normalize_quantization_config` 增加 `MIXED_PRECISION → modelopt_mixed` 映射(吸收已关闭未合并的 #36312 的解析修复)。
- `vllm/model_executor/models/utils.py`:新增共享 `map_draft_quant_config` 辅助函数,在构造量化层之前把 ModelOpt 草稿量化设置经权重别名、运行时模块前缀、decoder 层偏移映射到运行时名;仅对 ModelOpt 应用,其他后端保留原语义键并只加层别名。`qwen3_dflash.py` 的 `hf_to_vllm_mapper` 由 `orig_to_new_substr` 改为 `orig_to_new_regex` 以匹配完整模块名,并用新辅助函数替换 `_add_global_draft_layer_exclusions`。
- `vllm/model_executor/layers/quantization/modelopt.py`:`process` 中对 FNUZ 平台且 `float8_e4m3fn` 权重复用现有 `normalize_e4m3fn_to_e4m3fnuz` 做归一化,含 weight 与静态 input scale。
- 另更新 ModelOpt 文档、CI 依赖,并新增 config 归一化、别名映射、嵌套/DSpark 检查点回归等测试。

## 影响与回归风险
- 影响窗口:正文未写明版本范围;GPU 验证基于 vLLM 0.32.0 回移植,当前 main 全量环境与全套件未跑。
- 回归面:影响 ModelOpt 混合精度检查点加载路径(config 转换、草稿量化配置映射)、Qwen3 DFlash 草稿模型、ModelOpt per-tensor FP8 的 `process` 权重处理路径(仅 FNUZ 平台触发归一化分支);其他量化后端保留原有 quantization-config 处理,无需 FP8 kernel 改动。验证:5 个混合配置用例通过、17 个 CPU 草稿映射用例通过、小型混合 Llama/DSpark 检查点经 `LLM` 加载输出与 SpecDec OFF 精确一致;B200 GPT-OSS-20B DSpark 冒烟 TP=1 K=2 下 1120 个输出 token 与关闭推测解码完全一致;FNUZ 转换 CPU 验证保持反量化权重不变、scale 翻倍。AMD CI 新鲜验证待跑。

## 关联
- #36312(已关闭未合并,parser 修复来源)
- #49900(另一 proposer 路径处理 `draft_model.` 前缀)
- #53122(草稿量化别名与 context-KV,无本 PR 的运行时根/教师层重映射)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
