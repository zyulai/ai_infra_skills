# PR #20059: [Core] Allow full cudagraph with separate attention routines and orthogonal to compilation, add support for FA2 and FlashInfer

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fhl2000 |
| 合并 | 2025-08-15 |
| 分类 | Speculative Decoding (tier: major;与投机解码验证阶段的 full cudagraph 兼容相关) |
| Label | rocm, tpu, ready, ci/build, v1, llama |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 内容可用,列出 9 个文件,另有 26 个文件因超长被丢弃) |
| 链接 | https://github.com/vllm-project/vllm/pull/20059 |

## 一句话结论

重构 v1 CUDA graph 基础设施:full cudagraph 与编译解耦、支持 FA2/FlashInfer 的独立 attention 例程,新 `cudagraph_mode` 五模式,Triton attention 小模型 ITL -38%。

## 动机

旧设计(#16072)把 `splitting_ops` 置空、在展平的 fx graph 里捕获 full cudagraph,仅支持 FA3;#18581 为 FlashMLA 只捕获纯 decode,混合 prefill-decode 阶段回退 eager——而 profile 显示展平 graph 的 eager 执行 CPU 侧约 56ms,几乎两倍于 piecewise fx graph 的 28ms(Python 执行无子模块的大展平模块慢),prefill 小 batch 时会退化。另外 FA2/FlashInfer/FlashMLA 等 backend 对 prefill-decode 与纯 decode 有两条不同的 attention 例程,难以塞进单一 graph。

## 优化点

引入三个核心组件与两个辅助概念:(1) `CUDAGraphMode` 枚举:`NONE`/`PIECEWISE`/`FULL` 三个单模式,`FULL_DECODE_ONLY=(FULL,NONE)` 与 `FULL_AND_PIECEWISE=(FULL,PIECEWISE)` 双模式(decode 阶段 full graph 消 ITL,非均匀批次保留 piecewise 的低 TTFT),取代 `use_cudagraph`/`full_cuda_graph` 两个旧旗标;(2) `BatchDescriptor(num_tokens, uniform_decode)` 作为最小 dispatch key;(3) `CUDAGraphDispatcher` 是可用 cudagraph key 的唯一事实来源,按 FULL>PIECEWISE 优先级查找,无 key 则 NONE eager;(4) `CUDAGraphWrapper` 绑定单一 runtime_mode、信任 forward context,不隐含契约;(5) 嵌套 wrapper 设计:FULL wrapper 包整个模型,PIECEWISE wrapper 包编译内的 piecewise backend,两种模式互不冲突,因而 cudagraph 逻辑与 torch.compile 正交(无编译也能 full cudagraph);(6) `AttentionCGSupport` 枚举(ALWAYS/UNIFORM_BATCH/UNIFORM_SINGLE_TOKEN_DECODE/NEVER 默认)声明 attention backend 的 graph 能力,UNIFORM 类 backend 只支持 decode-only 双模式,FULL 模式自动降级为 FULL_AND_PIECEWISE 或 FULL_DECODE_ONLY;FA2 通过在捕获阶段控制 max_query_len 激活正确例程(两套 graph 分别覆盖混合与纯 decode),FlashInfer 用持久 buffer + 每 batch size 一个 decode warper;(7) full cudagraph 相关模式兼容投机解码验证阶段(当时 FA2/3、Triton attention)。

## 关键改动

```text
vllm/config/compilation.py: CUDAGraphMode 枚举(decode_mode/mixed_mode/max_cudagraph_mode 等)、cudagraph_mode 字段取代 use_cudagraph
vllm/compilation/cuda_graph.py: CUDAGraphWrapper(捕获/回放/直通)
vllm/compilation/base_piecewise_backend.py, cuda_piecewise_backend.py: piecewise backend 适配双模式 runtime dispatch
vllm/compilation/base_static_graph.py, backends.py: 静态图抽象与 backend 注册
vllm/compilation/wrapper.py: 模型级 wrapper 接线
```

关键代码(模式枚举):

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)
```

## 性能声明

A100 40G、torch 2.6 cuda 12.4,Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4,benchmark_serving(ShareGPT 100 prompts、rate 20):FA2 输出吞吐 1898.67→1992.27 tok/s(+5%),FlashInfer 1907.38→1946.35 tok/s(+2%);TPOT 分别 -2.9%/-3.1%;lm_eval gsm8k 两种 backend 均无变化。ITL:Triton attention 在小 LLM 上 FULL_AND_PIECEWISE 较 main 的 piecewise 降低 38%(见 PR 评论链接,正文未给原始表)。作者注明:以上数字测于 PR 初版,CLI 与代码结构后来大幅重构,仅供参考;另附 CPU profile(展平 graph ~56ms vs piecewise ~28ms 启动)。

## 验证与回归风险

Test plan 为 benchmark serving + lm_eval(FA2/FlashInfer);作者无 Hopper 无法测 FlashMLA/FA3。风险面:替换了 v1 引擎整个 cudagraph 捕获/调度路径 CompilationConfig 语义变化(`use_cudagraph`→`cudagraph_mode`,splitting_ops 默认值变化);attention backend 需声明 AttentionCGSupport,后端不兼容时依赖降级逻辑;warm-up 逻辑移到 gpu model runner;启动时间与显存开销增加(多一套 graph 捕获),作者建议后续缩减 prefill-decode 阶段捕获尺寸列表。

## 关联

同区域:前序 #16072(展平 fx graph full cudagraph,FA3)、#18581(FlashMLA decode-only);后续 #21367(FA2/FlashInfer full cudagraph 支持移入并继续优化)、#20283(RFC);投机解码验证阶段兼容为 #52861 等后续 DSA/breakable cudagraph 工作铺路。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
