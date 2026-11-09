# PR #13922: diffusion: fix the issue where the qwen-edit&wan model produces incorrect output during sequence parallelism

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#13922](https://github.com/sgl-project/sglang/pull/13922) |
| 作者 | yhyang201 |
| 合并日期 | 2025-11-26 |
| labels | run-ci, diffusion |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Parallelism / Communication |
| bug 类型 | correctness |

## 一句话结论
qwen-edit&wan 在 SP 下输出错误:adjust_num_frames 晚于 SP 对齐执行破坏帧数整除约束;调整顺序,并补上缺失的 @dataclass。

## 问题现象
qwen-edit&wan 模型在序列并行(SP)下产生错误输出。正文描述两点:①SP alignment 原本在 `pipeline_config.adjust_num_frames` 之前执行,存在竞态——adjust_num_frames 可能改动已对齐的帧数,使最终值不再满足多 GPU 执行所需的整除检查;②`QwenImageEditPipelineConfig` 缺少 `@dataclass` 装饰,导致该 pipeline 的类型仍是 t2i。正文未附具体报错或输出样例。

## 根因
正文披露:根因一是执行顺序——SP 对齐先于 pipeline 特定的帧数调整完成,后者会破坏前者保证的「num_frames 可被 num_gpus 整除」约束;根因二是类缺少 `@dataclass` 装饰器导致配置类型判定错误(仍为 t2i)。

## 修复方案
- `multimodal_gen/configs/sample/base.py::adjust()`:把 `adjust_num_frames` 的调用移到 SP 对齐逻辑之前,先执行 pipeline 特定帧数调整,再按 GPU 数做对齐,保证最终 `num_frames` 同时满足两者约束。
- `multimodal_gen/configs/pipeline_configs/qwen_image.py`:给 `QwenImageEditPipelineConfig` 加上 `@dataclass`。

```python
+            # NOTE: We must apply adjust_num_frames BEFORE the SP alignment logic below.
+            self.num_frames = server_args.pipeline_config.adjust_num_frames(
+                self.num_frames
+            )
```

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:多模态生成采样配置的帧数调整流程(所有视频/图像 pipeline 共用 base.adjust)与 QwenImageEdit 的配置类;调整后 SP 对齐基于已调整的帧数,非视频任务(num_frames=1 分支)不受影响。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
