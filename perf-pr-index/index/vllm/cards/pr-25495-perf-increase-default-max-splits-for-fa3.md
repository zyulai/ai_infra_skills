# PR #25495: [Perf] Increase default max splits for FA3 full cudagraphs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LucasWilkinson |
| 合并 | 2025-09-23 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未记录 +N/-N);diff 极小(vllm/envs.py 两处 16→32) |
| 链接 | https://github.com/vllm-project/vllm/pull/25495 |

## 一句话结论

把 FA3 在 CUDA graph decode 下的最大 KV splits 默认值从 16 提到 32,依据 #25274 的证据使长 context decode 获得更好的默认并行度。

## 动机

`VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH` 在 envs.py 中的注释为 "max number splits for cuda graph decode",即 FA3 attention 在 CUDA graph 捕获的 decode 下做 KV split 的数量上限,默认 16。作者引用 #25274:"provides evidence that 32 would be a much better default";同时由于 full-CG potential 即将成为默认(#25444),"seems like a good time to improve this"。素材 pack 未收录 #25274 的具体测量数据。

## 优化点

纯默认值调整,无逻辑变化:`vllm/envs.py` 中该环境变量的类默认值与 getenv fallback 字符串由 16 改为 32。split 上限翻倍意味着 FA3 在 graph decode 下可以把 KV 序列切给更多 SM 并行再做归并(素材未展开机制细节);用户仍可用同名环境变量覆盖回 16 或其他值。

## 关键改动

```text
vllm/envs.py: VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH 默认值 16 → 32(类默认与 getenv fallback 两处)
```

## 性能声明

PR 正文未附 benchmark 数字;性能依据是 #25274 的结论("32 would be a much better default"),其数据不在本 pack 内,PR 未注明具体测试条件。

## 验证与回归风险

无新增测试;属默认值翻转。回归风险:所有 FA3 + full cudagraph decode 的默认配置用户都受影响——split 数翻倍会增加 combine 归并工作量,短序列/小 batch 场景理论上可能非最优(素材未给反向数据);可用 `VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH=16` 回退。

## 关联

#25274(提供 32 更优的证据)、#25444(full-CG potential 成为默认,是时机动机)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
