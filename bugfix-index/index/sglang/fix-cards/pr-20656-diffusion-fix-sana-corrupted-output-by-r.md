# PR #20656: [Diffusion] Fix Sana corrupted output by removing spurious QK norm layers

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20656](https://github.com/sgl-project/sglang/pull/20656) |
| 作者 | ChangyiYang |
| 合并日期 | 2026-03-22 |
| labels | documentation, run-ci, diffusion |
| 热度 | 👍 2 · 💬 8 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
Sana 管线多出 checkpoint 中不存在的 norm_q/norm_k 随机 RMSNorm 层,破坏注意力输出导致乱图;直接删层修复。

## 问题现象
原生 SGLang Sana pipeline 生成的图像损坏/乱码,输出看起来是没有任何语义内容的随机图形混杂。

## 根因
正文披露:`SanaLinearAttention` 与 `SanaCrossAttention` 两个类都带有 `norm_q` / `norm_k` RMSNorm 层,但这些层在 Sana checkpoint 中并不存在。它们以随机权重初始化,并在注意力计算前作用于 query/key 张量,从而完全破坏注意力输出。作者通过检查 checkpoint 验证:`Efficient-Large-Model/Sana_600M_1024px_diffusers` 的 `diffusion_pytorch_model.safetensors` 中不存在任何 `norm_q` 或 `norm_k` 键。

## 修复方案
按真实 Sana 模型结构,从 `SanaLinearAttention` 与 `SanaCrossAttention` 中移除 `norm_q` / `norm_k`:删除两个类的 norm 层定义与 forward 中的 `query = self.norm_q(query)`、`key = self.norm_k(key)` 调用,`SanaLinearAttention.__init__` 去掉 `qk_norm_dim` 参数,外层调用点也不再传 `qk_norm_dim=num_attention_heads * attention_head_dim`。关键摘录(共 12 行,两个类各 6 行):
```diff
-        self.norm_q = RMSNorm(qk_norm_dim)
-        self.norm_k = RMSNorm(qk_norm_dim)
...
-        query = self.norm_q(query)
-        key = self.norm_k(key)
```
改动仅涉及 `python/sglang/multimodal_gen/runtime/models/dits/sana.py`。验证:`pytest python/sglang/multimodal_gen/test/server/test_server_a.py -k sana_image_t2i` PASSED。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 Sana DiT 的线性注意力与交叉注意力两个类的构造与 forward 路径;删除层后模型结构与官方 checkpoint 一致,权重加载不再期望 norm_q/norm_k 键(此前这些键本就缺失、走随机初始化)。对其他 DiT/模型无影响。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
