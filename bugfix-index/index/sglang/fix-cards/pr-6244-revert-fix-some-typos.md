# PR #6244: Revert "fix some typos"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6244](https://github.com/sgl-project/sglang/pull/6244) |
| 作者 | merrymercy |
| 合并日期 | 2025-05-12 |
| labels | (none) |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
整体回滚 #6209 的全库拼写/大小写规范化改动:改动量大、收益小且会制造合并冲突。

## 问题现象
非运行时 bug:#6209 "fix some typos" 对大量文件的注释、docstring 与少量用户可见文案做了大小写/拼写统一(如 `CUDA`→`cuda`、`LoRA`→`lora`、`vLLM`→`vllm`、`HuggingFace`→`huggingface` 等),波及 distributed、attention backend、quantization、lora、configs、examples、docker 等众多模块。正文说明回滚原因:这些大小写改动属小问题(minor),却会给在途分支带来合并冲突。

## 根因
正文披露:#6209 的改动只是大小写层面的次要规范化,却会引入 merge conflicts(大量行被触碰导致与并行开发的 PR 冲突),故由维护者整体回滚。

## 修复方案
原样回滚 #6209,把散布在数十个文件中的大小写改写全部恢复原状,例如:
```diff
-        # Disable CUDA graph and torch compile to save time
+        # Disable cuda graph and torch compile to save time
```
被回滚恢复的方向为 #6209 的逆操作,涉及(可见部分):`docker/k8s-sglang-distributed-sts.yaml`、`examples/runtime/hidden_states/*`、`python/pyproject.toml`、`compile_deep_gemm.py`、`_custom_ops.py`、`configs/chatglm.py`、`configs/load_config.py`、`distributed/device_communicators/*`(custom_all_reduce、custom_all_reduce_utils、pynccl、pynccl_wrapper)、`distributed/parallel_state.py`、`hf_transformers_utils.py`、`layers/attention/*`(base、double_sparsity、flashinfer、flashinfer_mla、triton、vision)、`layers/dp_attention.py`、`layers/logits_processor.py`、`layers/quantization/*`(含 compressed_tensors 系列、fp8_utils、blockwise_int8)、`layers/rotary_embedding.py`、`lora/*`(backend、layers、lora、lora_manager)等。
注:pack 的 diff 在 40000 字符处按文件粒度截断,其后文件未展示,未编造。

## 影响与回归风险
- 影响窗口:正文未说明;按 revert 对象,#6209 合并之后、本 PR 之前的提交受影响。
- 回归面:纯注释/文案层面恢复,无逻辑变化;个别为异常/日志消息文本(如 quantization 模块 ImportError 提示 "vLLM is not installed" 恢复为大写),若有测试或脚本对消息文本做精确匹配需注意;对在途分支而言,回滚本身消除了 #6209 造成的合并冲突面。

## 关联
- https://github.com/sgl-project/sglang/pull/6209 (被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
