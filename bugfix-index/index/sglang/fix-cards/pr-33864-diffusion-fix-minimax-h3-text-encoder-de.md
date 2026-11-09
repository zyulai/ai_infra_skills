# PR #33864: [diffusion] fix: MiniMax-H3 text encoder device mismatch under --text-encoder-cpu-offload

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33864](https://github.com/sgl-project/sglang/pull/33864) |
| 作者 | triple-mu |
| 合并日期 | 2026-08-07 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 1 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论
MiniMax-H3 开启 `--text-encoder-cpu-offload` 后每次请求都在文本编码阶段报设备不匹配;让 `device` 属性改报计算侧设备即修复。

## 问题现象
MiniMax-H3 上 `--text-encoder-cpu-offload` 完全不可用:每个请求都在文本编码阶段失败,报

```
RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cpu,
different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_bmm)
  transformers/models/qwen3_vl/modeling_qwen3_vl.py:367  (Qwen3VLRotaryEmbedding.forward)
```

在 1x RTX PRO 5000 上 3/3 复现。该问题同时阻断了「仅模型级 offload」以及任何与 DiT layerwise offload 混用的组合。插桩运行显示:`self.device=cpu  cpu_params=902/902`,`inputs={'input_ids': cpu, 'attention_mask': cpu}`,而 `hidden=cuda:0  inv_freq=cuda:0`。

## 根因
PR 正文披露:`MiniMaxH3Qwen3VLEncoder.device` 返回 `next(self.parameters()).device`。`--text-encoder-cpu-offload` 会将该组件经 `maybe_load_fsdp_model` 以 FSDP CPU offload 策略加载——分片参数留在 CPU 上,前向时才 all-gather 到加速器;此时参数设备代表的是**存储侧**而非**计算侧**。`encode_ids` 用 `.to(self.device)` 放置 `input_ids`、`attention_mask`、`position_ids`,于是这些输入留在 CPU,而前向在加速器上执行。rope 前向把一个操作数硬性对齐到激活设备(`inv_freq_expanded = ... .to(x.device)`),但 `position_ids_expanded` 留在 `position_ids` 到来的设备上,错配最终以 `bmm` 失败的形式暴露。

## 修复方案
`MiniMaxH3Qwen3VLEncoder.device` 改为返回 `get_local_torch_device()`——即前向真正运行的设备——而不是参数存储设备。正文指出该属性恰好只有 4 处使用,全部是为前向放置输入。另新增 `test_minimax_h3_encoder_device.py` 固定不变量:参数在 CPU 上时 `device` 仍须报告本地计算设备。

```diff
     @property
     def device(self) -> torch.device:
-        return next(self.parameters()).device
+        """Device this encoder's forward runs on.
+
+        Deliberately not `next(self.parameters()).device`. ...
+        parameter device then names the storage side, not the compute side ...
+        """
+        return get_local_torch_device()
```

当编码器常驻 GPU 时两种写法本就得到同一设备,因此只有 offload 路径的行为发生变化。

## 影响与回归风险
- 回归面:diff 仅改动 `minimax_h3_qwen3vl.py` 的 `device` 属性并新增单测文件。正文验证了不使用 `--text-encoder-cpu-offload` 的路径峰值显存逐字节相同(23138.0 MB),e2e 84.4s→85.4s 属于运行间噪声("after" 与其他租户共享节点)。改动面限于 MiniMax-H3 文本编码器的设备解析。

## 修复验证
MiniMax-H3 t2va,1344x768,124 frames @ 24fps,5 denoise steps,BF16,`quality=lossless`,1x RTX PRO 5000 (72 GB) 独占节点:
- `--layerwise-offload-components dit --dit-layerwise-resident-layers 20 --text-encoder-cpu-offload`:修复前设备不匹配,修复后 ok,峰值 43.4 GB,e2e 85.2 s。
- `--dit-cpu-offload --text-encoder-cpu-offload --vae-cpu-offload`:修复前设备不匹配,修复后到达内存检查并 OOM——正文说明模型级 DiT 需要完整 62 GB 常驻,单张 72 GB 卡上 OOM 是正确结果。
- 输出经 `ffprobe` 验证:1344x768、124 frames、24 fps、含音轨。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
