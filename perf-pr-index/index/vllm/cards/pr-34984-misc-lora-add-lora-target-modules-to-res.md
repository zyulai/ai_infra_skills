# PR #34984: [Misc][LoRA] Add --lora-target-modules to restrict LoRA to specific modules

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @bhoomit |
| 合并 | 2026-03-17 |
| 分类 | Model-specific Fast Path |
| Label | documentation, performance, new-model, rocm, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 内容本身可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/34984 |

## 一句话结论

新增 --lora-target-modules 部署级 LoRA 模块过滤,只 wrap 必要模块,单模块限制 TPOT 最高 -14%(graph)/-47%(eager)。

## 动机

vLLM 的 LoRA 会把模型中所有受支持模块(qkv_proj、o_proj、gate_up_proj、down_proj 等)全部替换成带 LoRA 的包装层,即使 adapter 本身只含少数模块的权重。适配器层面"只放部分模块的权重"并不能减少包装——vLLM 仍 wrap 全部受支持模块,未用到的包装层在推理路径上产生额外开销。缺少一个部署期开关,让用户按需只对特定模块启用 LoRA 以做性能调优。

## 优化点

新增部署期过滤:`LoRAConfig.target_modules` 字段与 `--lora-target-modules` CLI 参数,接受模块后缀(如 `o_proj`、`qkv_proj`)。过滤点在 wrap 之前——`LoRAModelManager._match_target_modules` 拆成两级判定:先 `is_supported_lora_module`(沿用原 regex 后缀匹配,判断模型是否支持该模块),再 `is_in_target_modules`(target_modules 为 None 时全放行,否则按 `module_name.split(".")[-1]` 后缀集合精确匹配,不经过包装)。即被排除的模块完全不被替换成 LoRA 层,而不是替换后运行时跳过。配套:`compute_hash` 把 target_modules 纳入编译缓存键;`WorkerLoRAManager._load_adapter` 对 adapter 中因不支持或被 target_modules 排除而将被忽略的模块打 warning_once;未指定参数时行为与原先完全一致(全部受支持模块)。

## 关键改动

```text
vllm/lora/utils.py: 新增 is_supported_lora_module / is_in_target_modules 两个纯函数
vllm/lora/model_manager.py: _match_target_modules 改为两级判定(supported + target_modules),wrap 前过滤
vllm/config/lora.py + vllm/engine/arg_utils.py: target_modules 字段、--lora-target-modules 参数、hash 因子
vllm/lora/worker_manager.py: 加载 adapter 时对将被忽略的模块告警
tests/lora/test_lora_utils.py 等: 过滤逻辑与 manager 单测、CLI 解析测试
```

关键片段(部署期过滤):

```python
def is_in_target_modules(module_name, target_modules):
    if target_modules is None:
        return True
    module_suffix = module_name.split(".")[-1]
    return module_suffix in set(target_modules)
```

## 性能声明

Qwen/Qwen3-32B(bf16),单卡 H200(143GB),LoRA rank 16,随机权重 adapter(PEFT,64 层),serving 配置 input 256 / output 128 / 32 prompts / 2 req/s。Baseline = adapter 只含目标模块权重但不加 flag;With TM = 全量 adapter + `--lora-target-modules`:

- CUDA graphs + torch.compile(生产模式):单模块限制下 TTFT 最高 -21.1%(qkv_proj:94.2 → 74.3 ms)、TPOT 最高约 -14%(o_proj:23.4 → 20.2 ms,-13.6%);gate_up_proj+down_proj 组合 TTFT -9.6%、TPOT -6.8%;all 模块时无开销(-0.1%~-2.6%)。
- enforce_eager:单模块 TPOT 最高 -47.3%(qkv_proj:72.3 → 38.1 ms)、TTFT 最高 -42.6%;组合 TPOT -33.0%。
- 作者要点:全模块激活时无开销(<1% 噪声);adapter 层面的限制无效,只有本参数的 wrap 级跳过才有效。

## 验证与回归风险

测试:`tests/entrypoints/openai/test_cli_args.py`(单/多模块与默认 None 解析)、`tests/lora/test_lora_manager.py`(target_modules 限制/多模块/None 全量/不支持的模块与被排除模块的告警行为,覆盖 CPU 与 GPU)、新增 `tests/lora/test_lora_utils.py`(两个过滤函数的边界:suffix/exact/regex 锚定/空列表)。风险:target_modules 语义是"后缀精确集合匹配"(不像 supported 列表用 regex),若模块名以非最后段形式命中需注意;adapter 含被排除模块时其权重被静默忽略(有 warning_once);compute_hash 变化会使既有 torch.compile 缓存失效一次。

## 关联

同区域:LoRA 子系统(punica wrapper、packed modules mapping);pack 未列出前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
