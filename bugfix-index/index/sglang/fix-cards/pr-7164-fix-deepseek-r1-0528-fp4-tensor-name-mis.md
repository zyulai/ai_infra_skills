# PR #7164: Fix Deepseek R1 0528 FP4 tensor name mismatch issue during weights loading.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7164](https://github.com/sgl-project/sglang/pull/7164) |
| 作者 | pyc96 |
| 合并日期 | 2025-06-17 |
| labels | high priority |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
DeepSeek-R1-0528-FP4 权重加载时 k_scale 张量名替换错误(生成 `kattn_mqa`),且 FP4 量化配置类不匹配,服务启动即 KeyError;修正替换逻辑并按算法选配置类。

## 问题现象
启动 `python3 -m sglang.launch_server --port=7080 --model-path=nvidia/DeepSeek-R1-0528-FP4 --trust-remote-code --tp=8 --quantization=modelopt_fp4 --kv-cache-dtype=auto`(blackwell image)时,Scheduler 在权重加载阶段抛异常:
```
File ".../models/deepseek_v2.py", line 2202, in load_weights
    param = params_dict[name]
KeyError: 'model.layers.5.self_attn.kattn_mqa.k_scale'
```

## 根因
正文披露:这是权重加载时的张量名不匹配——对照权重文件与模型配置,k_scale 实际名为 `model.layers.5.self_attn.attn_mqa.k_scale`,而加载逻辑用 `name.replace("_proj", "attn_mqa")` 把 `k_proj` 变成了 `kattn_mqa`(查不到对应参数);正文并指出 `attn_mha` 的 k_scale 也没有被更新。由 diff 描述的另一处机制:modelopt 产出的 FP4 checkpoint 在 `get_quant_config` 中仍走通用 `quant_cls.from_config`,未按 quant_algo 选择 `ModelOptFp4Config`。

## 修复方案
1. `python/sglang/srt/models/deepseek_v2.py` `load_weights`:k/v scale 的改名逻辑由 `name.replace("_proj", "attn_mqa")` 改为 `name.replace(f"{scale[0]}_proj", "attn_mqa")`(只替换 `k_proj`/`v_proj` 前缀,得到正确的 `attn_mqa`);
2. `python/sglang/srt/model_loader/weight_utils.py` `get_quant_config`:modelopt 产出且 `quant_algo` 含 `FP4` 时返回 `ModelOptFp4Config.from_config(config)`,否则维持原逻辑;
3. `deepseek_v2.py` `post_load_weights`:`_weight_requant_ue8m0` 前增加 `hasattr(self.quant_config, "weight_block_size")` 且非 None 的守卫。

```diff
-                            if any(scale in name for scale in ["k_scale", "v_scale"]):
-                                name = name.replace("_proj", "attn_mqa")
+                            for scale in ["k_scale", "v_scale"]:
+                                if scale in name:
+                                    name = name.replace(f"{scale[0]}_proj", "attn_mqa")
```

## 影响与回归风险
- 影响窗口:正文未说明;触发条件为加载 nvidia DeepSeek-R1-0528-FP4(modelopt_fp4)权重。
- 回归面:deepseek_v2 系模型加载 modelopt checkpoint 时 k/v scale 的参数名映射(影响 attn_mqa/attn_mha 各处 k_scale 落位)、modelopt FP4 量化配置解析路径、UE8M0 requant 守卫;其他量化格式的 `get_quant_config` 走 else 分支不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
