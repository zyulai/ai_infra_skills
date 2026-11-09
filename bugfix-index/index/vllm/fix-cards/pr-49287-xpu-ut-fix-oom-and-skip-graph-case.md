# PR #49287: [XPU][UT] Fix OOM and skip graph case

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49287](https://github.com/vllm-project/vllm/pull/49287) |
| 作者 | mayuyuace |
| 合并日期 | 2026-08-18 |
| labels | intel-gpu, multi-modality, kv-connector, mistral |
| 热度 | 👍 2 · 💬 13 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
修复 XPU 单测 OOM 与死循环:显存等待机制泛化为跨平台,mamba_cache_cg_padding 无 graph 时 skip,XPU 默认利用率降为 0.9。

## 问题现象
XPU 上单测存在两类问题(正文列明):
1. `test_mamba_cache_cg_padding` 只能在 graph 开启时运行,否则 UT 会陷入无限循环;
2. 连续用例导致 VRAM 不足——engine 启动时空闲内存不够,触发 OOM。

## 根因
正文给出修复思路但未系统展开机制;由 diff 推断(代码注释):XPU/oneAPI runtime 会在 pytest 父进程中为整个生命周期常驻约 1 GiB context(随进程内 HfRunner 模型增长),分布式测试还会在 engine 子进程额外分配 CCL context;默认 `gpu_memory_utilization` 0.92 对两者余量不足。且 XPU(Level Zero)与 ROCm 一样,在上一个 engine 关闭后会延迟释放显存,而原有的显存等待函数 `wait_for_rocm_memory_to_settle` 只对 ROCm 生效,XPU 场景无人等待。graph 用例则因 cudagraph 被禁用时调度器 dispatch 行为导致 while 循环不终止(由 diff 推断)。

## 修复方案
多文件测试基建改动:
1. `tests/utils.py`:`wait_for_rocm_memory_to_settle` 更名为 `wait_for_memory_to_settle`,生效范围扩为 ROCm 或 XPU;显存采样新增 XPU 分支,经 `torch.accelerator.get_memory_info`(Level Zero)取 (free, total)。
2. `tests/conftest.py`:所有调用点同步更名;LLM 构造前新增 XPU 分支——调用方未显式指定时把 `gpu_memory_utilization` 默认降为 0.9,并以 `threshold_ratio=1.0 - gpu_memory_utilization` 等待显存回落。
3. `tests/models/language/generation/test_hybrid.py`:`test_mamba_cache_cg_padding` 在 `cudagraph_dispatcher.cudagraph_mode == CUDAGraphMode.NONE` 时 `pytest.skip`。
4. `test_colbert.py`、`test_voxtral_realtime.py`、`nixl_integration/run_accuracy_test.sh` 的调用点同步更名。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:全部为测试基建(tests/ 下),不影响 vllm 运行时;XPU 上未显式指定利用率的测试默认值 0.92→0.9,模型可用显存略降;`wait_for_memory_to_settle` 在 XPU 上会阻塞等待(默认 timeout 240s),若显存长期不释放将等到超时。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
