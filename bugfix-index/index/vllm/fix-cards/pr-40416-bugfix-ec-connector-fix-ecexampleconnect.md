# PR #40416: [Bugfix][EC Connector] Fix ECExampleConnector load device under TP>1

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#40416](https://github.com/vllm-project/vllm/pull/40416) |
| 作者 | miyakido |
| 合并日期 | 2026-09-09 |
| labels | bug, ready, unstale, v1 |
| 热度 | 👍 5 · 💬 10 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
`ECExampleConnector.start_load_caches` 传裸 `"cuda"` 导致 safetensors 一律加载到 cuda:0;改为 pin 到当前 rank 的实际设备,修复 TP>1 下的失败。

## 问题现象
在 `tensor_parallel_size > 1` 时,非 rank-0 的 worker 会把 encoder cache 加载到 `cuda:0`,之后与本地设备上的张量合并时失败。作者在 vLLM v0.17.0 上用 Qwen3-VL-4B-Instruct、TP=4、4 x L4 GPU 复现。影响面:TP>1 且使用 EC(Encoder Cache)Connector 示例连接器加载缓存的场景。

## 根因
正文披露:`ECExampleConnector.start_load_caches` 把裸设备字符串 `"cuda"` 传给了 `safetensors.torch.load_file`;safetensors 会把 `"cuda"` 解析为 `cuda:0`,即使该 TP worker 实际运行在其他被选中的设备上,于是非 rank-0 worker 的缓存落在了错误的设备上。

## 修复方案
改 `vllm/distributed/ec_transfer/ec_connector/example_connector.py` 的 `start_load_caches`:用 vLLM 平台抽象把 CUDA 类平台的加载设备 pin 到当前选中的设备;非 CUDA 平台保持原 `device_type` 不变:

```diff
+        # A bare "cuda" makes safetensors load onto cuda:0. Pin the load to the
+        # selected device so tensor-parallel workers load the cache locally.
+        device = current_platform.device_type
+        if current_platform.is_cuda_alike():
+            device = f"{device}:{current_platform.current_device()}"
         # Load the EC for each mm data
         for mm_data in metadata.mm_datas:
...
-            ec_cache = safetensors.torch.load_file(
-                filename, device=current_platform.device_type
-            )["ec_cache"]
+            ec_cache = safetensors.torch.load_file(filename, device=device)["ec_cache"]
```

新增两个回归测试:CUDA 类平台加载到选中设备(如 `cuda:2`)、非 CUDA 平台保持裸 `device_type`(如 `cpu`)。pre-commit、Ruff、mypy 3.12 通过;focused pytest 因作者在 macOS ARM 无兼容 wheel 未本地重跑,已纳入 Linux/GPU CI(Buildkite #82780)。

## 影响与回归风险
- 影响窗口:正文提及复现于 vLLM v0.17.0;未给出受影响的更早版本范围。
- 回归面:仅 `ECExampleConnector` 的缓存加载设备选择逻辑及其单测;其他 EC/KV connector 与核心推理路径不受影响。CUDA 类平台设备字符串从 `cuda` 变为 `cuda:<current_device>`,依赖裸字符串行为的调用方需注意。

## 关联
- Buildkite CI run #82780(正文链接的验证构建)
- 正文未列出其他 issue/PR

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
