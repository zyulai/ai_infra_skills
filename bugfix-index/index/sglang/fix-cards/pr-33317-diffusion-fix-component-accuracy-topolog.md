# PR #33317: [diffusion] Fix component accuracy topology reuse

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33317](https://github.com/sgl-project/sglang/pull/33317) |
| 作者 | BBuf |
| 合并日期 | 2026-08-04 |
| labels | bug, ci, diffusion, npu, ready-for-review, run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
组件精度 harness 仅按 TP/SP/DP/CFG 总量复用并行组,Ulysses/Ring 分解变化被跳过重建,Ring 案例跑在陈旧 Ulysses 拓扑上;补齐检查并清理组与 IPC 状态。

## 问题现象
失败测试套件中,`fsdp-inference` 先以 SP=2 初始化为 Ulysses-2/Ring-1,紧接着 Qwen Ring 案例请求 Ulysses-1/Ring-2;因 SP 总量仍为 2,harness 跳过重建,Qwen ring canary 在陈旧的 Ulysses 拓扑下运行,测试失败(正文给出相关 CI 失败链接)。另有一个独立的阈值边界问题:Z-Image 2-GPU 精度案例的 H100 运行产出 0.993382,低于 0.9935 阈值;H200 在 CUDA-IPC 开/关两种情况下均产出 0.993504。修复后验证:H200 同进程拓扑切换 Qwen Image Ring-2 得 0.998509、Z-Image Ulysses-2 得 0.993504(2 passed, 38 deselected)。

## 根因
正文 Root cause 段完整披露:组件精度 harness 基于 TP/SP/DP/CFG 大小判断是否复用 model-parallel 组;Ulysses/Ring 分解变化而 SP 总量不变时被误判为可复用,Ring 案例继承了陈旧拓扑。连锁问题:拓扑检查修正后,harness 的合成 all-true 文本掩码会进入不支持的 masked-ring 路径——该掩码无语义作用,Ring 案例现将其省略。Z-Image 失败则是另一回事:H100/H200 FlashAttention 后端方差(0.99338–0.99350)使阈值 0.9935 落在边缘。

## 修复方案
- `test/single_test_file/component_accuracy/utils.py` 的 `initialize_parallel_runtime`:已初始化时的复用判断补充 `current_ulysses == ulysses_degree` 与 `current_ring == ring_degree`,分解变化即 `destroy_model_parallel()` 后重建。
- `hooks.py` 的 `_build_transformer_hook_inputs`:`uses_ring_parallel = (case.server_args.ring_degree or 1) > 1`,Ring 并行时省略合成的 all-true attention mask(注释:"This synthetic mask is all True and has no semantic effect. Omit it for ring-parallel cases, whose masked attention path is unsupported.")。
- `device_communicators/ipc_a2a.py`:`IpcA2AState` 记录 `group` 并新增 `reset()`;`ipc_a2a_ready` 在 group 被替换时先 reset 再按需重 init。
- `parallel_state.py` 的 `destroy_model_parallel`:销毁前 `IPC_A2A.reset()`;显式 `torch.distributed.destroy_process_group` 销毁 `PROCESS_GROUP.ULYSSES_PG`/`RING_PG` 并置 None——正文/diff 注释指出这两组由 SP coordinator 单独创建,`GroupCoordinator.destroy()` 不负责释放,否则反复 Ulysses/Ring 切换会泄漏 NCCL communicator 及其 CUDA 显存。
- `config.py`:Z-Image 两案例 transformer 阈值 0.9935→0.993(注释:H100/H200 FlashAttention 波动约 0.99338–0.99350),覆盖后端方差而不掩盖实质性回归。
- 新增单测:`test_component_accuracy_inputs.py`、`test_component_accuracy_parallel_runtime.py`、`test_ipc_a2a_lifecycle.py`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:组件精度 harness 的并行初始化/复用判定与 hook 输入构造;扩散运行时 `destroy_model_parallel` 的组销毁路径与 IPC-A2A 传输生命周期(组替换时重置);Z-Image 2-GPU CI 阈值放宽 0.0005,理论上可能放过 0.993–0.9935 之间的轻微劣化,正文称不会掩盖实质性回归;其余模型与推理路径不受影响。

## 关联
- [相关 CI 失败](https://github.com/sgl-project/sglang/actions/runs/30753555033/job/91516481836?pr=33282)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
