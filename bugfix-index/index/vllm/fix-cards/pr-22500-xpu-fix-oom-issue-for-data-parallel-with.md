# PR #22500: [XPU] Fix OOM issue for data parallel with Ray backend

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22500](https://github.com/vllm-project/vllm/pull/22500) |
| 作者 | faaany |
| 合并日期 | 2025-08-27 |
| labels | ready, v1 |
| 热度 | 👍 1 · 💬 4 |
| 分类 | Parallelism / Communication |
| bug 类型 | resource |

## 一句话结论
Ray + XPU 数据并行时两个 actor 挤在同一设备,actor 内设 env 太晚;改经 runtime_env 预设 ZE_AFFINITY_MASK。

## 问题现象
用 Ray 后端数据并行启动 vllm server 时启动即失败,报 KV cache 显存不足:正文照抄报错——`ValueError: To serve at least one request with the models's max seq len (131072), (3.50 GiB KV cache is needed, which is larger than the available KV cache memory (1.26 GiB). Based on the available memory, the estimated maximum model length is 47104.`(调用栈位于 `_initialize_kv_caches` → `check_enough_kv_cache_memory`)。实际是两个 Ray actor 都在同一设备上分配 KV cache 张量。

## 根因
正文披露:`DPEngineCoreActor.__init__` 里设置的环境变量不生效——Ray 在加载 Ray actor 之前就已初始化了 XPU 上下文(dpctl 提前初始化 GPU runtime),因此 actor `__init__` 中改环境变量对设备选择为时已晚。修复思路:必须在 Ray actor 初始化之前设置 XPU 环境变量;由于 `runtime_env` 在 Ray worker 进程启动时生效,故通过每个 actor 的 runtime_env 配置 `ZE_AFFINITY_MASK`。

## 修复方案
- `vllm/v1/engine/utils.py`:从 `set_device_control_env_var` 中抽出 `get_device_indices()`,返回给定 data parallel rank 的设备索引串(逻辑不变)。
- `vllm/v1/engine/core.py`:
  - `DPEngineCoreActor` 把 `_set_cuda_visible_devices` 拆为 `_set_visible_devices`:XPU 平台直接 pass(设备选择交给 runtime_env),其余平台走原逻辑并复用 `get_device_indices`。
  - `CoreEngineActorManager` 创建 actor 时,对 XPU 平台用 `get_device_indices` 算出 `ZE_AFFINITY_MASK`,连同 env_vars_dict 一起写入 `RuntimeEnv(env_vars=actor_env_vars)`,再传给 `ray.remote(...).options(...)`。

关键 diff 摘录(core.py):
```diff
+            if current_platform.is_xpu():
+                device_evar = current_platform.device_control_env_var
+                device_indices = get_device_indices(device_evar, local_index,
+                                                    world_size)
+                actor_env_vars = self.env_vars_dict.copy()
+                actor_env_vars[device_evar] = device_indices
+                runtime_env = RuntimeEnv(env_vars=actor_env_vars)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:XPU + Ray 数据并行路径(runtime_env 注入设备环境变量);`_set_cuda_visible_devices` 重构后非 XPU 平台行为等价(改经 `get_device_indices`),需关注 CUDA 等平台 DP 启动路径;XPU actor 内原有的环境变量设置被跳过。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
