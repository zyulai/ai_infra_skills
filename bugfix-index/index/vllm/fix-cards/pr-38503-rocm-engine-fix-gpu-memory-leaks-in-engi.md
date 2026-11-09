# PR #38503: [ROCm][Engine] Fix GPU memory leaks in engine shutdown and test workaround for async KV prefix cache reset

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38503](https://github.com/vllm-project/vllm/pull/38503) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-04-25 |
| labels | rocm, ready, ci/build, v1, kv-connector |
| 热度 | 👍 1 · 💬 16 |
| 分类 | KV Cache / Memory |
| bug 类型 | resource |

## 一句话结论
修复 in-process 引擎关闭时三处 GPU 显存泄漏(lru_cache 钉住权重、gc.freeze 未解、无显式 teardown),另加 prefix cache 重置测试 workaround 与 ROCm UCX 泄漏修复。

## 问题现象
以 in-process 模式(`VLLM_ENABLE_V1_MULTIPROCESSING=0`)运行时,关闭引擎会泄漏 GPU 显存。另外,测试中 `reset_prefix_cache` 在异步 KV store 传输(CPU offloading)尚在途时可能一直返回 False——所有请求结束后 engine 不再 step,不会轮询 worker 获取已完成的传输。UCX 侧:UCX rocm_ipc 传输经 `hsa_amd_ipc_memory_create` 在 `ucp_mem_map` 时永久 pin 住 GPU 显存,引擎关闭后显存不可恢复(ROCm/ucx#33)。

## 根因
正文明确披露三点泄漏来源:
1. `supports_kw` 的 `@lru_cache` 以绑定方法(如 `model.forward`)为 key,对模型实例及其全部 GPU tensor 持有强引用,缓存存续期内无法释放;
2. `EngineCore.__init__` 调用 `gc.freeze()` 以降低 serving 期间 GC 开销,但 `shutdown()` 从不调用 `gc.unfreeze()`,引擎启动期分配的对象(模型权重、KV cache 等)对垃圾收集器不可见;
3. `GPUWorker.shutdown()` 从不释放模型权重、KV cache、rotary embedding 缓存(`_ROPE_DICT`)、编译上下文与 workspace 单例。
UCX 泄漏为 ROCm UCX 分支问题,通过升级 UCX 分支(`33b3b2a2`)与设置 `HSA_ENABLE_IPC_MODE_LEGACY=1` 规避。

## 修复方案
- `vllm/utils/func_utils.py`:`supports_kw` 拆为内部 `_supports_kw`(保留 lru_cache)与外层包装,缓存前把绑定方法解包为 `__func__`;
- `vllm/v1/engine/core.py`:`shutdown()` 末尾新增 `gc.unfreeze()`;
- `vllm/v1/worker/gpu_model_runner.py`:新增 `shutdown()`——清 `static_forward_context`、置 `model = None`、清空 `_ROPE_DICT`、`reset_workspace_manager()` 并调用 `_cleanup_profiling_kv_cache()`;`vllm/v1/worker/gpu_worker.py` 的 `shutdown()` 调用 `model_runner.shutdown()`;
- 测试:`_wait_for_prefix_cache_reset` 在重试间隔发送 dummy 单 token prefill(`TokensPrompt(prompt_token_ids=[0])`)强制 engine step,促使 scheduler 轮询 `get_finished()` 并释放被 in-flight async store 占用的 GPU block(按 reviewer 建议的临时 workaround);新增 ROCm 显存回收回归测试;
- `docker/Dockerfile.rocm`:升级 RIXL/UCX 分支、auditwheel 排除项调整、新增 `HSA_ENABLE_IPC_MODE_LEGACY=1`。

## 影响与回归风险
- 回归面:in-process(单进程)模式的引擎生命周期与资源回收;`supports_kw` 的缓存 key 语义变化(对所有调用方生效,绑定方法统一解包为底层函数);gpu_model_runner/gpu_worker 的 shutdown 路径;ROCm docker 镜像构建(UCX/RIXL 版本与环境变量);kv offload 相关测试的等待逻辑。另有一条 shutdown 时长断言从 <2s 放宽到 <2.1s。

## 关联
正文未列出 issue/PR 链接;Motivation 为 AMD CI Buildkite 构建链接(buildkite.com/vllm/amd-ci/builds/7062/...)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
