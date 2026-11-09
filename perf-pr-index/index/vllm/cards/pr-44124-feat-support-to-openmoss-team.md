# PR #44124: feat: support to OpenMOSS-Team

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @nagisa-kunhah |
| 合并 | 2026-06-24 |
| 分类 | Other(tier: major) |
| Label | documentation, performance, new-model, ready, multi-modality, mistral |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 中 5 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/44124 |

## 一句话结论

新增 MOSS-Audio(OpenMOSS-Team 4B/8B Instruct/Thinking 四个模型)音频多模态模型支持。

## 动机

closes #43431:vLLM 尚不支持 OpenMOSS-Team 的 MOSS-Audio 音频输入模型,用户无法用 vLLM 服务这些 checkpoint。

## 优化点

新增 `MossAudioModel`:Whisper 风格音频 encoder + Qwen3 语言模型骨架(MossQwen3Model/MossQwen3ForCausalLM)。custom processor 按处理后音频长度动态展开 audio placeholder(注入 BOS/AUDIO/EOS token),支持文本 prompt 与预展开 token prompt 两条输入路径及 processor-only cache;DeepStack 多层音频 embedding 注入,按音频打包缓存 `deepstack_input_embeds` 并支持 PP 分段经 `IntermediateTensors` 传递;`GatedMLP` 支持 TP 切分与 packed `gate_up_proj` 权重加载;encoder 权重名保持 HF 原名(q_proj/k_proj/v_proj/out_proj)。另让 `vllm/benchmarks/throughput.py` 支持 `custom_audio` 数据集与 `--no-oversample`、`--enable-multimodal-chat`、`--custom-output-len` 选项。

## 关键改动

```text
vllm/model_executor/models/moss_audio.py(新): 模型/encoder/processor 完整实现
tests/models/multimodal/processing/test_moss_audio.py(新): placeholder 展开、DeepStack 缓存、TP/PP 路由、LoRA 映射、TP 形状
tests/models/multimodal/generation/test_moss_audio.py(新): 生成 smoke 与 TP2/PP2 并行 smoke
vllm/benchmarks/throughput.py: custom_audio 数据集与多模态 chat 请求构造
docs/models/supported_models.md: 注册 MossAudioModel 条目
```

## 性能声明

PR 未提供性能/benchmark 数字。功能验证:处理测试 32 passed、e2e 2 passed;A100 40GB 上四个模型(`MOSS-Audio-4B/8B-Instruct/Thinking`)经 `audio_url` 与 `input_audio` 两条路径加载并返回非空内容。

## 验证与回归风险

HF 对照精度测试因远程 MOSS-Audio config 无法经 `AutoModelForCausalLM` 加载而显式 skip(生成覆盖由 smoke 承担);`test_common` 对 MOSS-Audio 跳过通用多模态处理正确性检查(custom processor 特性所致)。风险:新增模型路径本身不影响既有模型;benchmark CLI 新增参数为增量;TP 校验(adapter_hidden_size/attention heads 整除)在构造期报错。

## 关联

无(素材中无同区域其他 PR;上游 issue #43431)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
