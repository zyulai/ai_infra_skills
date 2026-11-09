# PR #51703: [Bugfix] Record non-ImportError attention backend probe failures instead of crashing engine init

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51703](https://github.com/vllm-project/vllm/pull/51703) |
| 作者 | Eoin-Houstoun |
| 合并日期 | 2026-08-20 |
| labels | bug, ready, nvidia |
| 热度 | 👍 2 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
CUDA attention backend 探测只捕获 ImportError,PermissionError/OSError 等会打断引擎初始化;改为记录 "ExceptionType: message" 并继续探测后续后端,选中路径保留异常链。

## 问题现象
`CudaPlatform.get_valid_backends` 的后端探测循环本意是标记不可用后端并继续,但只捕获 ImportError:探测中抛出的其他异常——如只读 flashinfer 缓存目录导致的 PermissionError、损坏的驱动安装导致的 OSError——会直接逃逸并使引擎初始化崩溃,即使优先级列表中下一个有可用后端。且被捕获时记录的原因只是裸字符串 "ImportError",丢失真实失败细节。

## 根因
正文披露(修复 #51658):`vllm/platforms/cuda.py` 两处探测路径的 `except ImportError` 过窄,未覆盖环境类异常;同时记录的失败原因不含异常信息。

## 修复方案
改 `vllm/platforms/cuda.py` 两处:
- 枚举路径(`get_valid_backends`)捕获范围扩大并记录具体原因,以 debug 级别输出 traceback:

```diff
-            except ImportError:
-                invalid_reasons_i = ["ImportError"]
+            except (ImportError, OSError) as e:
+                logger.debug(
+                    "Attention backend %s is unavailable", backend.name, exc_info=True
+                )
+                invalid_reasons_i = [f"{type(e).__name__}: {e}"]
```

- 选中后端路径(`get_attn_backend_cls`)改为 `raise ValueError(...Reason: [{type(e).__name__}: {e}]...) from e`,根因保留在异常链。

新增 mock 型、CUDA 门控的 `tests/v1/attention/test_cuda_backend_probe_errors.py`。正文说明 rocm.py 与 `get_vit_attn_backend` 存在相同模式,但本 PR 只改 issue 点名的两处(ViT 有安全的 TORCH_SDPA 回退)。A100 实测:main 上 7 failed/1 passed,修复后 8 passed;不涉及数值变化。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:CUDA 平台 attention backend 探测——环境类失败(ImportError/OSError)不再中断引擎初始化,改为记录原因并继续探测;非这两类的意外异常(如 RuntimeError/AttributeError)仍会传播,KeyboardInterrupt 不被吞(测试断言);显式选中后端失败时的 ValueError 信息格式变化(由 diff 推断)。

## 关联
- #51658

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
