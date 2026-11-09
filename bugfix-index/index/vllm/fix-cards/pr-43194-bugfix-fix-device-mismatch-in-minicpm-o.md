# PR #43194: [Bugfix] fix device mismatch in MiniCPM-o-4_5 resampler

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43194](https://github.com/vllm-project/vllm/pull/43194) |
| 作者 | yma11 |
| 合并日期 | 2026-05-26 |
| labels | bug, ready |
| 热度 | 👍 2 · 💬 1 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
MiniCPM-o-4_5 resampler 的 2D pos_embed 只转 dtype 未搬设备,与 xpu:0 张量相加报错,补 .to(device) 修复。

## 问题现象
运行 `openbmb/MiniCPM-o-4_5` 时在 resampler 前向中崩溃。正文贴出错误堆栈:`vllm/model_executor/models/minicpmv.py` 第 422 行 `k = x + pos_embed_2d` 处抛 `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, xpu:0 and cpu!`(Worker_TP0,经 multiproc_executor 上报)。复现命令(照抄正文):`VLLM_WORKER_MULTIPROC_METHOD=spawn python3 examples/basic/offline_inference/generate.py --model openbmb/MiniCPM-o-4_5 --enforce-eager -tp=2 --trust_remote_code`。正文 Test Result 一节为空,未附修复后输出。

## 根因
正文通过错误堆栈披露现象与出错位置(x 与 pos_embed_2d 设备不一致,一个在 xpu:0、一个在 CPU),未另行文字说明根因。由 diff 推断:构造 `pos_embed_2d` 时对 `self.pos_embed` 切片只做了 `.reshape((tgt_h * tgt_w, -1)).to(dtype)`——仅转换 dtype、未移动设备,导致该张量仍留在模型加载/默认设备(此处表现为 CPU),与已在加速器上的 x 相加时报设备不匹配(由 diff 推断)。

## 修复方案
改 `vllm/model_executor/models/minicpmv.py` resampler forward 中 pos_embed_2d 的构造,把 `.to(dtype)` 改为显式 `.to(device=device, dtype=dtype)`:
```python
pos_embed_2d.append(
    self.pos_embed[:tgt_h, :tgt_w, :]
    .reshape((tgt_h * tgt_w, -1))
    .to(device=device, dtype=dtype)
)  # patches * D
```
即 2D 位置嵌入在 reshape 后同时搬到目标设备并转 dtype,与 x 所在设备对齐。

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:MiniCPM-o-4_5 / MiniCPM-V 系列(minicpmv.py)resampler 的 2D pos embed 路径;每次前向多一次设备搬运(该张量随后参与加法,本身必须与 x 同设备,无正确性风险);其他仅转 dtype 未转设备的同类调用点不在本次修改范围。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
