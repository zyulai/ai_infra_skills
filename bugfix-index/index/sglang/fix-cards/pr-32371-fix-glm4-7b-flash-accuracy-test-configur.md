# PR #32371: Fix GLM4-7B-Flash accuracy test configuration, tune Qwen3.6-27B/35B performance test parameters, and harden Ascend NPU multi-node E2E test utilities against pod name format errors.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#32371](https://github.com/sgl-project/sglang/pull/32371) |
| 作者 | pllimax |
| 合并日期 | 2026-07-29 |
| labels | npu, run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
NPU 测试基建修三处:E2E 工具对 pod 名非数字后缀崩溃加防护,GLM4-7B-Flash AIME25 测试配置纠错,Qwen3.6-27B/35B 性能测试参数调优,全部为测试侧改动。

## 问题现象
Ascend NPU 多节点 E2E 测试工具在 pod 名/ConfigMap 条目不符合"以数字结尾"预期格式时崩溃(`int()` 直接转换非数字后缀);GLM4-7B-Flash 的 AIME25 精度测试配置错误(类名沿用 DeepSeek_V3_2、环境变量名用错、精度阈值不对)。正文未给出具体崩溃日志。

## 根因
正文未逐条披露根因分析, Modifications 描述的机制如下:
1. pod 名格式脆弱性:`int(host_name.rsplit("-", 1)[-1])` 等三处及 ConfigMap 遍历假定最后一段是数字,遇到意外格式即抛异常(由 diff 推断的失败机制);同时 pod 列表未按 label 过滤,会枚举到非目标 pod。
2. GLM4-7B-Flash 测试:类名 `TestNPUDeepSeek_V3_2_8P_AIME2025` 与模型不符、环境变量误用 `HCCL_BUFFSIZE`(应为 `DEEPEP_HCCL_BUFFSIZE`)、精度阈值 0.916 需恢复为 0.7667(正文称 "Restore accuracy threshold",未解释 0.916 从何而来)。
3. Qwen3.6-35B-A3B 性能问题与 `HCCL_BUFFSIZE: "1"` 配置相关(正文将其归入 "performance fix",未进一步说明)。

## 修复方案
1. `run_npu_e2e_test.py`:`list_namespaced_pod` 加 `label_selector="app=sgl-ascend"` 只列目标 pod;`test_npu_multi_node_utils.py` 三处 hostname 解析先校验后缀 `isdigit()` 再 `int()`,异常格式抛 `RuntimeError`;ConfigMap 遍历跳过非数字后缀条目并记日志。
2. `test_npu_glm4_7_flash_1p_aime25.py`:类名改为 `TestNPUGLM_4_7_FLASH_1P_AIME25`,`HCCL_BUFFSIZE`→`DEEPEP_HCCL_BUFFSIZE`,accuracy 阈值 0.916→0.7667。
3. `test_npu_qwen3_6_27b_1p_gpqa.py`:补 `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`。
4. `test_npu_qwen3_6_27b_w8a8_2p_in16k_out1k_50ms.py`:加 `SGLANG_ENABLE_SPEC_V2=1`,调 max-running-requests 29→37、mem-fraction 0.68→0.70、扩充 cuda-graph-bs 列表、speculative 参数 steps 3→4/draft 4→5、并发与 prompt 数调整。
5. `test_npu_qwen3_6_35b_a3b_...py`:`HCCL_BUFFSIZE "1"→"100"`。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:仅测试与测试基建文件(6 files, +56/-16),不涉及推理运行时代码;多节点 E2E 工具对异常格式从崩溃改为显式报错/跳过;GLM4-7B-Flash 精度门限放低至 0.7667(阈值合理性依赖作者判断);Qwen3.6 性能测试的通过基线随参数变化。CI:Base 绿、Extra 红(原帖标注)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
