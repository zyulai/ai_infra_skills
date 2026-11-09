# PR #20769: SM100 Cutlass MLA decode with unrestricted num_heads (< 128) for DeepSeek TP

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @alexm-redhat |
| 合并 | 2025-07-15 |
| 分类 | Attention / Kernel |
| Label | documentation, performance, ready, ci/build, v1, deepseek |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 部分可用,pack 注明另 dropped 10 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/20769 |

## 一句话结论

解除 SM100 Cutlass MLA decode 内核 num_heads==128 的限制,DeepSeek 小头数配置 TPOT 最高 -65%。

## 动机

vLLM 原有的 SM100(B200)Cutlass MLA decode 内核把 num_heads 硬性限制为 128:像 DeepSeek-Coder-V2-Lite-Instruct 这类 TP 后 num_heads < 128 的配置根本无法走该内核(甚至无法运行)。SGLang 侧已通过 sgl-project/sglang#6929 解除了该限制,本 PR 把这组改动移植回 vLLM,让 B200 上 DeepSeek MLA 的 TP 部署也能吃到这条快速路径。

## 优化点

本 PR 以新文件形式引入移植自 SGLang PR #6929 的 cutlass_sm100_mla 内核族:warpspecialized 的主 kernel(TMA 加载 + 2SM UMMA,TileShapeH 固定 128)、按模板参数 `kNumHeads` 参数化的 split-KV reduction kernel,以及统一的 device 层封装。device 层的 `set_split_kv` 在 split_kv 未指定时按 SM 数、batch、max_splits=ceil(K/128)(上限 16)做 wave-aware 启发式,平衡 split 并行度与 reduction 开销。reduction kernel 用 LSE(softmax 归一化因子)加权合并各 split 的部分输出,支持 per-batch 的 seq/split 指针,从而在 head 数不再是 128 的约束下仍能正确做分块归并。

## 关键改动

```text
csrc/attention/mla/cutlass_sm100_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp: SM100 warpspecialized MLA 主 kernel(TMA + 2SM UMMA,模板化 head 数)
csrc/attention/mla/cutlass_sm100_mla/kernel/sm100_fmha_mla_reduction.hpp: 按 kNumHeads/kHeadDimLatent 参数化的 split-KV reduction kernel
csrc/attention/mla/cutlass_sm100_mla/device/sm100_mla.hpp: device 层封装 + set_split_kv 启发式
```

关键片段(set_split_kv 的启发式):

```cpp
int max_splits = ceil_div(K, 128);
max_splits = min(16, max_splits);
int sms_per_batch = max(1, sm_count / B);
int split_heur = min(max_splits, sms_per_batch);
int waves = ceil_div(B * split_heur, sm_count);
int k_waves = ceil_div(max_splits, split_heur);
args.split_kv = ceil_div(max_splits, k_waves);
```

## 性能声明

- deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,单张 B200(该模型 num_heads < 128,本 PR 前无法运行):b1 TPOT 提升 65%,更大 batch 约 8-10%。
- DeepSeek V2,4xB200:b1 与 b64 TPOT 提升 10%。
硬件为 B200(SM100),精度与 workload 细节(序列长度/请求数)PR 未在正文中注明(仅附截图)。

## 验证与回归风险

PR 正文未列新增 pytest 测试,验证以 B200 上的端到端基准截图为主(pack 的 diff 中未见测试文件,且 pack 声明 dropped 10 个文件,不排除测试文件在完整 diff 中)。风险:内核仅面向 SM100;TileShapeH 静态断言 128、`TileShapeL % TileShapeR == 0` 等约束仍存在;split_kv 启发式改变长上下文/不同 batch 下的 split 策略,可能影响其他 DeepSeek 系模型在 B200 上的性能表现;非 SM100 架构不受影响。

## 关联

移植自 SGLang PR sgl-project/sglang#6929(Alcanderian JieXin Liang);同属 DeepSeek MLA / SM100 attention 内核区域。pack 中未列出 vLLM 侧前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
