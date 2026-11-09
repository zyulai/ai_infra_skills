# PR #42332: [Bugfix] Fixes MiniCPM-O resampler device placement to avoid tensor device mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42332](https://github.com/vllm-project/vllm/pull/42332) |
| 作者 | j9smith |
| 合并日期 | 2026-06-17 |
| labels | bug, ready |
| 热度 | 👍 2 · 💬 3 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
MiniCPM-O 启动即崩:`pos_embed` 缓冲留在 CPU 导致 CUDA/CPU 张量相加报错;在 `MiniCPMOBaseModel.load_weights` 补调 `_ensure_resampler_device()` 修复。

## 问题现象
MiniCPM-O 2.6 启动时崩溃,报 `Expected all tensors to be on the same device`。触发条件:直接启动 `vllm serve openbmb/MiniCPM-o-2_6 --trust-remote-code --max-model-len=16000`(在全新 vLLM 安装 v0.20.1 上测试)。影响面:修复前 engine core 在 profile run 阶段启动失败(与 #42322 中相同的 traceback),服务完全不可用;修复后 engine core 正常初始化、权重加载完成、API server 启动并可接受请求。

## 根因
正文完整披露了根因链条:`Resampler2_5._set_2d_pos_cache` 在初始化时于 CPU 上创建 `pos_embed` 缓冲(位于 `minicpmv.py`)。正常路径下,该缓冲会通过 `MiniCPMVBaseModel._ensure_resampler_device` 移到 GPU——该方法一般在 `MiniCPMVBaseModel` 的 `load_weights` 中被调用;但 `MiniCPMOBaseModel` 覆写了父类的 `load_weights`(为了跳过 `tts` 前缀),却从未调用 `_ensure_resampler_device`,导致 resampler 一直未被搬到 GPU、`pos_embed` 缓冲留在 CPU。profile run 执行到 `x + pos_embed` 时出现 CUDA 与 CPU 张量相加,直接崩溃。

## 修复方案
改动极小,只改 `vllm/model_executor/models/minicpmo.py` 的 `MiniCPMOBaseModel.load_weights`,补上父类已有的设备搬迁调用,对齐 `MiniCPMV` 的模式:
```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(self, skip_prefixes=["tts"])
    loaded = loader.load_weights(weights)
    self._ensure_resampler_device()
    return loaded
```
`MiniCPMO2_6` 与 `MiniCPMO4_5` 均继承自 `MiniCPMOBaseModel`,两个模型同时被修复。

## 影响与回归风险
- 回归面:仅 `MiniCPMOBaseModel.load_weights` 一处调用时序——`_ensure_resampler_device` 现在在 AutoWeightsLoader 加载完 `tts` 之外的全部权重后执行;若权重加载中途抛异常,该调用不会执行(与父类模式一致);对不使用 resampler 的 MiniCPM-O 配置,`_ensure_resampler_device` 的行为与父类路径相同。

## 关联
- [Fixes #42322](https://github.com/vllm-project/vllm/issues/42322)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
