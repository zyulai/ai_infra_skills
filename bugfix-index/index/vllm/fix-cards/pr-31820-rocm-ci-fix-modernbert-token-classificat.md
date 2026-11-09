# PR #31820: [ROCm][CI] Fix ModernBERT token classification test numerical accuracy on ROCm

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31820](https://github.com/vllm-project/vllm/pull/31820) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-01-06 |
| labels | rocm, ready |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
ROCm 上 ModernBERT token classification 测试偶发数值超差(max diff ~0.03 > 0.01);在测试 conftest 中按 ROCm 调整 SDP 后端与 matmul 精度设置修复。

## 问题现象
`test_modernbert_models` 在 ROCm 上有时失败:vLLM 自定义 kernel 与 HuggingFace eager attention 之间存在数值精度差异,最大 diff 约 0.03,超过 0.01 阈值,且仅出现在 2 个浮点值上——属偶发性精度边界失败。

## 根因
正文披露:ROCm 默认的 matmul 精度设置会产生略有不同的数值结果。

## 修复方案
新增 `tests/models/language/pooling/conftest.py`,通过 `pytest_sessionstart` 钩子在 ROCm 平台上调整 SDP 后端与 matmul 精度:

```python
if not current_platform.is_rocm():
    return
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision("high")
```

即关闭 flash/mem-efficient SDP、启用 math SDP 以规避 HuggingFace Transformers 的精度问题,并把 float32 matmul 精度设为 "high"。作者称循环运行测试 100+ 次(每次清缓存、从头重编译模型)验证通过。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:改动仅新增测试目录下的 conftest,只影响 `tests/models/language/pooling/` 下 pytest 会话在 ROCm 上的 SDP 后端与 matmul 精度配置,不影响生产代码路径;非 ROCm 平台直接返回,行为不变。

## 关联
- 正文未列出(diff 注释中引用了 https://github.com/vllm-project/vllm/issues/30167 )

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
