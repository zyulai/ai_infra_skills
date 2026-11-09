# PR #12803: Fix the run-time error when calling fused_rms_mxfp4_quant that change return output number

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#12803](https://github.com/sgl-project/sglang/pull/12803) |
| 作者 | kkHuang-amd |
| 合并日期 | 2025-11-10 |
| labels | amd, deepseek, run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Quantization |
| bug 类型 | hang |

## 一句话结论
fused_rms_mxfp4_quant 返回输出数量变化,调用处解包不匹配导致运行时错误,改为兼容多输出解包。

## 问题现象
运行 DeepSeek mxfp4 模型(AMD/aiter 路径)时,attention 调用中出现 incorrect type 的运行时错误,根源是 `fused_rms_mxfp4_quant` 的返回输出数量发生了变化。正文未给出具体报错堆栈;作者附了 gsm8k 基准结果(bench_sglang.py --num-questions 2000 --parallel 2000,Accuracy 0.939,Invalid 0.000,Output throughput 1574.955 token/s)证明修复后精度正常。

## 根因
正文披露:`fused_rms_mxfp4_quant` 的返回输出数量发生了变化(return output number changed),而各调用点仍按旧的两输出解包,导致运行 deepseek mxfp4 模型时 attention 调用中的类型/解包错误。修复思路是「Handle multiple output number」。正文未进一步说明输出数量为何变化;diff 中同时把 `docker/rocm.Dockerfile` 的 `AITER_COMMIT` 从 v0.1.6.post1 升到 v0.1.6.post3,暗示与 aiter 版本相关(由 diff 推断)。

## 修复方案
修改所有调用 `fused_rms_mxfp4_quant` 的位置,按位置解包并吸收多余的返回值:
- `python/sglang/srt/layers/communicator.py` `prepare_attn` 两处:

```python
hidden_states, *_, _ = fused_rms_mxfp4_quant(...)
...
hidden_states, *_, residual = fused_rms_mxfp4_quant(...)
```

- `python/sglang/srt/models/deepseek_v2.py` `forward_absorb_prepare` 与 `forward_npu_sparse_prepare` 两处:

```python
q, k_nope, *_ = fused_rms_mxfp4_quant(...)
```

- `docker/rocm.Dockerfile`:两个 stage 中 `AITER_COMMIT` v0.1.6.post1 → v0.1.6.post3。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:影响所有使用 `fused_rms_mxfp4_quant` 的路径(communicator 的 prepare_attn、DeepSeek-V2 的 forward_absorb_prepare / forward_npu_sparse_prepare),即 aiter + gfx95 + mxfp4 量化的 DeepSeek 模型前向;解包写法 `*_` 对输出数量继续变化具有容忍性。Docker 镜像的 aiter 版本升级会改变 ROCm 构建产物的行为面。

## 关联
正文未列出(未附 issue/PR 链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
