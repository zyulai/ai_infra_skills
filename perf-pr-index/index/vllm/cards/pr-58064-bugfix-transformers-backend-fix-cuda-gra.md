# PR #58064: [Bugfix][Transformers backend] Fix CUDA graph capture for dynamic-rope models

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @YongfengX |
| 合并 | 2026-10-14 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | bug, ready, ci/build, nvidia, verified |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含 CI 配置、新测试文件与 transformers backend 两个源文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/58064 |

## 一句话结论

跳过不可能触发的 Transformers dynamic rope 设备同步,修复 Hunyuan 等动态 rope 模型 CUDA graph capture 崩溃,Hunyuan-7B-Instruct 吞吐 1.7x。

## 动机

Transformers 的 @dynamic_rope_update 在每次 rotary forward 中对 torch.max(position_ids) 做设备同步,dynamic-rope 模型在 Transformers backend 上 CUDA graph capture 直接崩溃;#53615 之后 Hunyuan V1 dense 模型(如 Hunyuan-7B-Instruct)在 main 上以默认设置无法启动(Closes #57890,Completes #23738)。

## 优化点

当 max_model_len <= max_position_embeddings 时,位置永远不会超过 max_position_embeddings,dynamic rope 的重算永不触发。因此在模型初始化后把 decoder 中 dynamic rotary 模块的 rope_type 改标为 "default",跳过该同步;inv_freq 保持 init 时的值,输出 bit-identical。改标在模块上而非 config 上,因为 Hunyuan 的 alpha 等模型专属 dynamic init 只在 rope_type=="dynamic" 时运行。can_enable_torch_compile 与改标共享同一 config 级判定;longrope 层类型(同步是真实的)与超过 max_position_embeddings 的场景不改标,行为不变。

## 关键改动

```text
vllm/model_executor/models/transformers/utils.py: dynamic_rope_update_is_noop 判定
  与 disable_noop_dynamic_rope 模块改标逻辑
vllm/model_executor/models/transformers/base.py: 初始化时调用 disable_noop_dynamic_rope
tests/models/transformers/test_dynamic_rope.py: 新增测试(不再同步、输出不变、
  compile 判定)并纳入 models_basic CI
```

## 性能声明

tencent/Hunyuan-7B-Instruct,1x B200,默认 max_model_len 32768:main 直接 capture crash;main + cudagraph_mode NONE 吞吐 34.78 req/s、输出 4451 tok/s、gsm8k 0.321;本 PR 59.44 req/s(1.7x)、7608 tok/s、gsm8k 0.316、GPQA-Diamond 61.1(model card 60.1)。

## 验证与回归风险

新增 tests/models/transformers/test_dynamic_rope.py(5 passed),并加入 .buildkite models_basic CPU 步骤。风险:仅当 max_model_len ≤ max_position_embeddings 时改标,超过该长度或含 longrope 层的模型仍走原同步路径;影响 Transformers backend 所有 dynamic-rope 模型的 torch.compile 与 CUDA graph 捕获路径。

## 关联

Closes #57890;涉及 #53615(问题引入)与 #23738(完成的 issue)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
