# PR #57421: [Core][Kernel] Share persistent workspaces for Marlin and Humming

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin(co-authored: ErinYin) |
| 合并 | 2026-09-20 |
| 分类 | Quantization |
| Label | performance, ready, nvidia, quantization |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面);diff 因超长截断,另有 5 个文件未收录 |
| 链接 | https://github.com/vllm-project/vllm/pull/57421 |

## 一句话结论

把 Humming MoE permute scratch 与 Marlin lock buffer 收归 WorkspaceManager 共享持有,40 层 MoE 模型初始化后常驻显存最多省 9.8 GiB。

## 动机

Humming grouped MoE 的 permute scratch 此前由每层各自持有,常驻显存随层数线性增长(40 层 MoE 即 40 份);Marlin 的 lock/workspace buffer 也由单个 layer 或 kernel 拥有,生命周期被绑死在 weight preparation/reload 上,既浪费显存又与 CUDA graph 捕获的地址耦合。此前 #50626(tensor-only Marlin)与 #56764(Humming 专用 cache)分别处理,缺一个统一资源 API。

## 优化点

`WorkspaceManager` 新增两个原语:`get_persistent_resource(key, factory)` 管理存储稳定的对象、`get_persistent(...)` 管理固定 shape 张量,均按 `(ubatch, lane)` 执行槽位分别缓存,且必须在 workspace lock 之前完成分配(lock 后访问未预热资源会 assert)。在此基础上:Humming 的 `MoEPermuteScratch` 改经 `get_moe_permute_scratch` 获取,兼容的(容量、路由维度、device、hidden size、dtype 相同)相邻层共享同一份;Marlin lock 改为执行时经 `get_marlin_workspace` 获取,并发 CUDA stream、ubatch、lane 各有独立 buffer,仅零初始化一次且成功的 Marlin 调用会把计数器归零;dense Marlin 的 workspace 查找包进 runtime custom op,避免 torch.compile 追踪执行上下文选择;无 workspace manager 时保留独立分配的 standalone 路径,也支持显式传入 workspace。各 Marlin kernel 封装(mixed_precision/mxfp4/mxfp8/nvfp4/scaled_mm)与 quantization 配置类均移除 layer 级 `workspace` 属性。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py: 新增 get_moe_permute_scratch(按 key 经 WorkspaceManager 共享)
vllm/model_executor/kernels/linear/*/marlin.py 等: 移除 layer.workspace,apply_weights 传 workspace=None 运行时解析
vllm/v1/worker/workspace.py: get_persistent_resource / get_persistent 原语(diff 截断部分)
tests/v1/worker/test_workspace.py 等: 槽位隔离/lock/replay 回归测试
```

## 性能声明

显存(RedHatAI/Qwen3.6-35B-A3B-NVFP4,40 MoE 层,hidden 2048,256 experts top-k 8,单 B300 TP=1/DP=1,BF16,Humming grouped,固定 8 GiB KV cache):初始化后 live PyTorch 分配,max batched tokens 2048 时 33.412→30.953 GiB(省 2.459 GiB),8192 时 41.352→31.516 GiB(省 9.836 GiB);scratch 对象 40→1,8192 tokens 下 scratch 存储 10330.684→258.267 MiB(-97.5%)。作者注明该测量不建立延迟改进。质量:GSM8K 全量 1319 题 5-shot before/after 91.58%/91.21%(重复基线对照 91.05%),在基线波动范围内,作者明确不据此声称 bit 级等价;Marlin 后端 GSM8K 128 题 92.97%。

## 验证与回归风险

测试覆盖资源复用与 locking、standalone fallback、ubatch/lane 与 CUDA stream 间隔离、weight-preparation 行为、torch.compile 下的 Marlin 执行;permute 测试对比有无 persistent scratch 的输出,CUDA graph 测试改变输入/路由/graph size 以检测 stale metadata 或 aliasing(定向运行 488/20/112/60 passed 等)。作者列出未验证项:无 H200、无端到端分布式 DeepEP v2/HT/LL、DBO、DSpark serving 运行;BF16/SILU 覆盖,不含 FP8/SITU 情形(#55105)。风险主要在资源生命周期:错误共享或跨 stream 复用会引入数据竞争,由 (ubatch, lane) 隔离与 lock 前分配约束兜底。

## 关联

合并方向:#50626(Marlin tensor-only)、#56764(Humming 专用 cache);解决 #56759 内存问题,关联 #48312;互补 #55105(dispatch capacity/runtime dtype)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
