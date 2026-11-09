# PR #23036: [Core] Support weight_loader_v2 for `UnquantizedLinearMethod`

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @kylesayrs |
| 合并 | 2025-09-24 |
| 分类 | Quantization(major) |
| Label | documentation, performance, new-model, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, llama, qwen, deepseek, gpt-oss |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,涉及 3 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/23036 |

## 一句话结论

把 `UnquantizedLinearMethod` 的权重参数升级为 `ModelWeightParameter`,使其支持 weight_loader_v2,为 transforms+TP 等组合铺路。

## 动机

`UnquantizedLinearMethod`(未量化线性层)此前创建的是裸 `torch.nn.Parameter`,不在 weight_loader_v2 支持列表内。而一些量化方法会包裹未量化参数,例如 CompressedTensors transforms(PR #22486):transform 需要套在未量化权重之上。本 PR 不为 transforms 支持所必需,但 transforms + TP 组合需要未量化层同样走 v2 加载路径,否则这类结构无法统一加载。

## 优化点

三处机制改动:(1) `UnquantizedLinearMethod.create_weights` 把 `Parameter(torch.empty(...))` 换成 `ModelWeightParameter`,携带 `input_dim=1`、`output_dim=0` 与 weight_loader,属性设置方式相应调整;(2) 把 `"UnquantizedLinearMethod"` 加入 `WEIGHT_LOADER_V2_SUPPORTED` 列表,使该层的权重加载进入 v2 通道;(3) 基础设施配套:`BasevLLMParameter.weight_loader` 从只读 attribute 改为 property(支持 setter/deleter,兼容 mamba_mixer2 等覆盖/删除 loader 的模型),新增 `__torch_function__` 透传,并在 `vllm/compilation/decorators.py` 增加 `_torch27_patch_tensor_subclasses`,让 `BasevLLMParameter`/`ModelWeightParameter` 等 tensor subclass 可被 torch.compile 追踪,免去 compile 前替换回普通张量。这是加载路径/编译基础设施现代化,不在热路径上引入更快 kernel。

## 关键改动

```text
vllm/model_executor/layers/linear.py: create_weights 改用 ModelWeightParameter;加入 V2 支持列表
vllm/model_executor/parameter.py: weight_loader property 化(setter/deleter)与 __torch_function__
vllm/compilation/decorators.py: _torch27_patch_tensor_subclasses 使 subclass 可 trace
```

关键片段(linear.py):

```python
weight_loader = extra_weight_attrs.pop("weight_loader")
weight = ModelWeightParameter(data=torch.empty(
    sum(output_partition_sizes), input_size_per_partition,
    dtype=params_dtype),
    input_dim=1, output_dim=0, weight_loader=weight_loader)
```

## 性能声明

PR 正文未给出性能数字(perf claims 检测为 none);本 PR 目标是功能/结构统一,预期收益是 enabling(解锁 transforms+TP)。PR 未注明 benchmark 测试条件。

## 验证与回归风险

作者测试:`meta-llama/Llama-3.1-8B-Instruct` 与 `Qwen/Qwen1.5-MoE-A2.7B`,各跑 TP=1、TP=2,加 CI。风险面:所有未量化线性层权重从裸 Parameter 变为 tensor subclass,任何按 `isinstance(weight, Parameter)` 之外假设写的代码(取 `.data`、torch.compile 行为)都可能受影响;torch 2.7 上 subclass trace 通过 patch 实现,版本升级(2.8+)后 patch 逻辑需复核。weight_loader 变为可删除属性,自定 loader 的模型行为需保持兼容。

## 关联

关联 PR #22486(CompressedTensors transforms,本 PR 是其 transforms+TP 的前置)。同分类 Quantization 下参数体系(Parameter → vLLMParameter)相关 PR;PR 未注明其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
