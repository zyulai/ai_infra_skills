# PR #28345: [Performance][gpt-oss] Revert gpt-oss max cudagraph size to 1024

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mmangkad |
| 合并 | 2025-11-08 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | performance, ready, gpt-oss |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用(单文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/28345 |

## 一句话结论

gpt-oss 默认 max cudagraph capture size 从 992 恢复到 1024,捕获图数量 67→83。

## 动机

gpt-oss 的 max capture size 曾因"full cuda graph with FA3 时最大支持 992"的限制(#25508)被从 1024 降到 992。随着 #27840 合并,该限制不复存在,992 的折衷不再必要,恢复 1024 可减少大 batch 回落 eager 的区间。

## 优化点

vllm/model_executor/models/config.py 中 gpt-oss 的 verify_and_update_config:在用户未显式设置 cudagraph_capture_sizes 与 max_cudagraph_capture_size 时,把 max_cudagraph_capture_size 从 992 改回 1024,同时移除关于 FA3 992 上限的 FIXME 注释。捕获档位从 67 增至 83(代码注释),graph 覆盖的 batch size 上限更高;仅是默认值 override,用户显式配置时不干预。

## 关键改动

```text
vllm/model_executor/models/config.py: gpt-oss 默认 max_cudagraph_capture_size 992 → 1024
```

```python
# Increase the max capture size from 512 to 1024 for performance.
# NOTE(woosuk): This will increase the number of CUDA graphs
# from 67 to 83.
compilation_config.max_cudagraph_capture_size = 1024
```

## 性能声明

PR 未给出 benchmark 数字:Test Plan 仅 "Tested",Test Result 仅 "Works"。收益方向为大 batch decode 更多命中 cudagraph,但 PR 未量化。

## 验证与回归风险

验证为作者自述的功能性测试,无 CI 数字。风险很小:只影响 gpt-oss 且仅在用户未显式设置 capture size 时生效;CUDA graph 数量增加 16 档,显存占用(每图捕获内存)相应上升,是吞吐/显存的默认权衡;不影响其他模型。

## 关联

前序:#25508(把 gpt-oss max capture size 从 1024 降到 992,本 PR 基本回退该改动)、#27840(移除 FA3 full cuda graph 992 上限,使恢复成为可能)。同模型卡:#39904(gpt-oss H100 fused_moe 配置)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
