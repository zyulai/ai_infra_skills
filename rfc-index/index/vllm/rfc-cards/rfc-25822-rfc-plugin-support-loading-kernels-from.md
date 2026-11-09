# RFC #25822: [RFC][Plugin]: support loading kernels from other place

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25822](https://github.com/vllm-project/vllm/issues/25822) |
| 作者 | ILikeIneine |
| 创建日期 | 2025-09-28 |
| 状态 | ⚪ closed(关闭于 2026-01-02) |
| labels | RFC, stale |
| 热度 | 👍 2 · 💬 6 |
| 分类 | Attention / Kernel |

## 一句话结论
提议在 Platform 上增加内核导入接口(import_general_kernels / import_moe_kernels),让编译内核的加载位置可插拔,插件无需把内核拷进 vllm 目录。

## 提案动机
当前所有编译内核都在 `_custom_ops.py` 中通过 `import vllm._C`、`import vllm._moe_C` 等直接加载。这带来两个限制:动态库必须放在 vllm 目录下(通常在 site-package 内);插件或 out-of-tree 平台若想复用该文件、适配或注册同样的内核,必须在安装阶段把自己的内核从插件目录拷贝到 vllm 目录里。

## 提出的方案
- 在 `Platform` 上提供平台级接口,支持加载平台特定的 kernel 操作:`import_general_kernels()`(导入平台特定 C 内核,默认实现导入 `vllm._C`)与 `import_moe_kernels()`(导入平台特定 MoE 内核,默认实现导入 `vllm._moe_C`)。
- `_custom_ops.py` 侧通过 `current_platform.import_general_kernels()` 加载,失败时捕获 `ImportError` 并 `logger.warning`;MoE 内核用 `contextlib.suppress(ImportError)` 包裹,成功则置 `supports_moe_ops = True`。
- 作者已提交 draft PR #25823,征求评论与帮助。

## 优化点与收益
正文未给出量化数字。收益为打包与插件生态层面:内核动态库不再被强制放进 vllm 安装目录,插件/自定义平台可在不拷贝文件的情况下注册与加载自己的内核,消除安装期的文件拷贝 hack。

## 现状与落地
已关闭(2026-01-02)。正文附有 draft PR #25823,反馈期 9.28 ~ 10.8;最终合入情况需查关联 PR。

## 关联
- Draft PR: https://github.com/vllm-project/vllm/pull/25823

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
