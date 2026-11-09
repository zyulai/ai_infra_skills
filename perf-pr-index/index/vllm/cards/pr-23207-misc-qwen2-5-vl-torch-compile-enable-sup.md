# PR #23207: [Misc][qwen2_5_vl][torch.compile] Enable `supports_torch_compile` on generic nn.Module and demonstrate speedup on Qwen Vision model

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Lucaskabela |
| 合并 | 2025-10-28 |
| 分类 | CUDA Graph / Compiler |
| Label | documentation, ready, llama, qwen |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 内容本身可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/23207 |

## 一句话结论

让 @support_torch_compile 支持任意子 nn.Module,并用它编译 Qwen2.5-VL 视觉骨干,吞吐 +5%、TPOT/ITL 约 +7%。

## 动机

vLLM 的 torch.compile 装饰器此前只用于顶层模型架构(vllm_config 驱动),多模态编码器(vision/audio 等模态组件)没有 vLLM config、也不是顶层模块,无法被编译;Qwen2.5-VL 的视觉骨干因此完全跑在 eager 模式。同时这些子模块里存在 graph break 之源:xformers attention 内部的 `to_list`、flash attention 的 `.item()`,以及以 Python `list`/`int` 传入的 seqlens/max_seqlen。

## 优化点

三层改动。其一,`support_torch_compile` 装饰器泛化:`__init__` 里 vllm_config 允许为 None 时回退 `get_current_vllm_config()`,并按被装饰类 `__init__` 的签名决定是否注入 vllm_config/prefix,使任意子 nn.Module 都可打标编译;新增 `mark_unbacked_dims` 参数,用 `torch._dynamo.decorators.mark_unbacked` 强制 dynamo 不对 0/1 维度特化(视觉 dummy 输入需要)。其二,attention 封装成 custom op(带 fake impl 避免断图):`flash_attn_maxseqlen_wrapper`(含 ROCM aiter/upstream FA 分支)与 `xformers_attn_seqlens_wrapper`,`max_seqlen`/`seqlens` 从 Python int/list 改为 GPU tensor,消除 `.item()`/`.tolist()` 的隐式同步;`compute_attn_mask_seqlens` 相应改为返回 tensor。其三,Qwen2.5-VL 落地:`Qwen2_5_VisionBlock`(dynamic_arg_dims 覆盖 x/cu_seqlens/rotary_pos_emb/seqlens + mark_unbacked seqlens)、`Qwen2_5_VisionPatchEmbed`、`Qwen2_5_VisionPatchMerger` 三个子模块打编译标;用 `set_model_tag` 让 32 个 VisionBlock 共享同一编译 tag;visual 调用包在 `set_forward_context(None, vllm_config)` 里;新增编译计数测试。

## 关键改动

```text
vllm/compilation/decorators.py: 泛化 support_torch_compile(vllm_config 回退 + 签名注入)+ mark_unbacked_dims
vllm/attention/ops/vit_attn_wrappers.py: flash/xformers attention 的 custom op 封装(带 fake impl)
vllm/model_executor/models/qwen2_5_vl.py: 三个视觉子模块 @support_torch_compile、set_model_tag、seqlens 张量化
vllm/model_executor/models/qwen2_5_omni_thinker.py: visual 调用包 set_forward_context
tests/compile/test_multimodal_compile.py: 断言 35 个模型被编译计数覆盖
```

关键片段(视觉块的编译标注):

```python
@support_torch_compile(
    dynamic_arg_dims={
        "x": 0,
        "cu_seqlens": 0,
        "rotary_pos_emb": 0,
        "seqlens": 0,
    },
    mark_unbacked_dims={"seqlens": 0},
)
class Qwen2_5_VisionBlock(nn.Module):
```

## 性能声明

Qwen/Qwen2.5-VL-3B-Instruct,`vllm bench serve`,VisionArena-Chat(hf train split)1000 prompts,torch nightly:

- Main:Request throughput 7.37 req/s,Output token throughput 782.79 tok/s,Total 1478.44 tok/s,Mean TTFT 63924.39 ms,Mean TPOT 681.74 ms,Mean ITL 596.16 ms。
- 本 PR:7.71 req/s,821.69 tok/s,1548.80 tok/s,TTFT 61043.37 ms,TPOT 648.66 ms,ITL 569.48 ms。
- 作者总结:吞吐相对提升约 5%,TTFT 约 2%,TPOT 约 7%,mean ITL 约 7%(median ITL 仅小幅改善)。另注:视觉骨干在 H100 上单独编译(封装 attention 后吞吐相对提升约 5%、token 延迟约 7%——vit_attn_wrappers.py 头注释口径)。

## 验证与回归风险

测试:`tests/compile/test_multimodal_compile.py`(forked,关多进程,断言 num_models_seen=35:LLM 后端 + PatchEmbed + PatchMerger + 32 个 VisionBlock;注释说明未来应只编译一次子模块并复用,关联 issue #27590);e2e 用 `examples/offline_inference/vision_language.py -m qwen2_5_vl` 验证。回归风险:所有 Qwen2.5-VL(含 omni thinker)视觉路径受影响——attention 后端 FA/xformers 走 custom op、TORCH_SDPA 保持逐图循环;启动时多一次视觉骨干编译(Dynamo ~3.16s + 缓存加载 ~2.1s,缓存命中后 0.02s);`vllm_config=None` 的构造路径变化可能影响依赖原构造签名的代码;mark_unbacked 只对 seqlens 维度生效,极端 0/1 batch 形状行为需依赖 dynamo 正确性。

## 关联

同区域:vLLM piecewise compile / VllmBackend、多模态编译支持(issue #27590:VisionBlock 编译复用、#27044/#27045:prefix 与 usage 的 TODO);后续可为其他多模态编码器(audio 等)复用此机制。pack 未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
