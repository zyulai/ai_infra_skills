# PR #10062: Piecewise CUDA Graph Support & Torch Compile Backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Oasis-Git |
| 合并 | 2025-10-12 |
| 分类 | CUDA Graph / Compiler |
| Label | high priority, run-ci |
| Diff | diffstat 缺失;素材 diff 截去 13 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/10062 |

## 一句话结论

引入 piecewise CUDA graph 与自定义 torch.compile 后端,按 attention 切子图捕捉,降低 prefill launch 开销。

## 动机

受 vLLM v1 的 torch.compile 设计启发,把完整推理图按 attention 操作切成小子图再捕捉 CUDA graph,相对全图捕捉有三个潜在优势:降低 prefill 的 kernel launch 开销(短输入尤其明显);允许 attention 内部出现复杂/外部操作(如 decode 时的 layerwise KV cache 读写),这是全图 CUDA graph 做不到的;支持 prefill 与 decode 在同一 running batch 内混合执行。实现主要改编自 vLLM 开源项目。

## 优化点

新增 compilation 包:SGLangBackend 作为自定义 torch.compile 后端,`split_graph` 以 `sglang.unified_attention_with_output` 为切分 op 把 FX 图分成 piecewise 子图(keep_original_order 保留 mutation 语义),PiecewiseCompileInterpreter 用 Inductor 编译每个可捕捉子图并挂 CUDAPiecewiseBackend 做 per-shape CUDA graph 捕捉/重放,编译缓存落盘到 SGLANG_CACHE_DIR;PiecewiseCudagraphRunner 负责分段 capture 与 replay。RadixAttention 在 extend 模式且处于 piecewise 上下文时改走注册的 custom op(输出通过 `output.copy_` 返回,mutates_args 声明),使其对 torch.compile 透明;启用 torch_compile 时 custom allreduce 的 max_size 从 8MB 提到 256MB 以避免 illegal cuda memory access;由 `--enable-piecewise-cuda-graph` 开启。

## 关键改动

```text
python/sglang/srt/model_executor/compilation/backend.py(新增): SGLangBackend、split_graph、PiecewiseCompileInterpreter(改编自 vLLM v0.10.0)
python/sglang/srt/layers/radix_attention.py: 注册 unified_attention_with_output custom op;extend+piecewise 上下文改走该 op
python/sglang/srt/distributed/parallel_state.py: torch_compile 时 custom allreduce max_size 提至 256MB
python/sglang/srt/model_executor/compilation/compile.py(新增): install_torch_compiled trampoline(运行时切换编译/原生 forward)
```

```python
if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
    output = torch.zeros_like(q)
    torch.ops.sglang.unified_attention_with_output(
        q, k, v, output, save_kv_cache, self.layer_id
    )
    return output
```

## 性能声明

H100,Qwen3-8B,GSM8K 200 题:开启 piecewise:Accuracy 0.950,Latency 7.752 s,Output throughput 3076.427 token/s;关闭:Accuracy 0.950,Latency 11.384 s,Output throughput 2091.467 token/s。prefill-only 用 bench_one_batch(Qwen/Qwen3-14B,batch 1,input-len 16-512,output-len 1,--enable-piecewise-cuda-graph)给出曲线图,PR 未提供具体数字;未注明更大规模 serving benchmark。

## 验证与回归风险

GSM8K 精度开启/关闭一致(0.950,200 题)。影响面:torch.compile/piecewise 路径下所有模型 forward;torch_compile 时 custom allreduce 缓冲从 8MB 变 256MB,增加显存占用;layernorm 的 flashinfer fused_add_rmsnorm import 被改为 sgl_kernel(注释掉 flashinfer 分支);documentation checklist 未完成;大量编译基建文件改编自 vLLM v0.10.0,行为与 vLLM 版本耦合。

## 关联

设计改编自 vLLM v1(编译/分段捕捉);素材未列出其他同区域 SGLang PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
