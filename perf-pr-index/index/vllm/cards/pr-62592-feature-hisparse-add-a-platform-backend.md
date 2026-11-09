# PR #62592: [Feature][HiSparse] Add a platform backend for device operations

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zhenwenqi2024 |
| 合并 | 2026-10-27 |
| 分类 | Model-specific Fast Path \| tier: major |
| Label | documentation, rocm, ready, deepseek, kv-connector, nvidia, verified, kv-cache-manager, glm, hisparse |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 6 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/62592 |

## 一句话结论

为 HiSparse 增加平台可选 backend 接口，让硬件插件复用 runtime/connector（不宣称 CUDA 性能提升）。

## 动机

HiSparse 的 runtime 与 connector 此前与 CUDA 绑定，硬件插件无法复用。PR 增加一个平台选择的 backend，使各平台可通过 `@classmethod` hook 指定自己的 HiSparse backend，而 CUDA 沿用相同的 cache 操作与 kernel。

## 优化点

新增 `HiSparseBackend` 承载 cache 操作、host 注册、IPC handles、resident-row payload、capture 查询、DMA 与 statistics dtype。各平台通过 `Platform.get_hisparse_backend_cls()` `@classmethod` 选择 backend，基类返回 `None`，CUDA 在 `platforms/cuda.py` override。分配前校验 cache format，device view 在既有初始化 rollback 作用域内准备。通用设备操作直接用 `torch.Stream`/`torch.Event`/`torch.accelerator`，仅 IPC event 需 backend 特定的创建/导入/导出；FP8 prefill 保留原有 custom-op 调用。Speculative group prefetch 不在范围内，Follower staging 保留原 `decode_batch` 条件，PR 明确 No CUDA performance improvement is claimed。

## 关键改动

```text
vllm/platforms/interface.py: 基类新增 get_hisparse_backend_cls() 返回 None
vllm/platforms/cuda.py: CUDA override 返回 CUDAHiSparseBackend
vllm/v1/hisparse/backend.py: 新增 HiSparseBackend 通用实现
vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py: connector 改用 backend/accelerator 入口
vllm/v1/attention/backends/mla/index_group.py、vllm/config/vllm.py: 派发与配置接线
```

```python
+    @classmethod
+    def get_hisparse_backend_cls(cls) -> str | None:
+        """Return the device implementation for the common HiSparse runtime."""
+        return None
...
+    @classmethod
+    def get_hisparse_backend_cls(cls) -> str | None:
+        return "vllm.v1.hisparse.backend.CUDAHiSparseBackend"
```

## 性能声明

PR 正文未给出定量性能数据，且明确声明 No CUDA performance improvement is claimed（本 PR 为平台接口改动）。PR 未注明测试条件。正文中的数字为测试通过数（如 CPU 验证 `22 failed, 5 passed` → `27 passed`），非性能基准。

## 验证与回归风险

验证以 CPU 为主：Linux/aarch64、PyTorch 2.10、显式 CPU target 下运行 test_kv_cache_planning.py / test_kv_cache_utils.py / test_config.py -k hisparse，由 22 failed/5 passed 改为 27 passed（默认 Qwen3-0.6B 配置缓存、未加载权重）。两处 checked-in resident-gather 用例用真实 CPU tensor 与真实 HiSparseBackend 通过；pre-commit（Ruff/mypy/typos/markdownlint/SPDX/import/test tethering）通过。作者明确：这是 CPU 验证而非 CUDA/NPU 执行，CUDA/NPU 模型、graph replay、精度与性能验证均 pending；Windows worker pytest collection 以 access violation（-1073741819）终止。回归风险：需与 Ascend 侧 resolver workspace ownership 对齐，device 回归未在本地运行。

## 关联

正文明确列出：Related #60072；Ascend 集成 vllm-project/vllm-ascend#19109；Speculative group prefetch 为 #63801 的独立工作；重叠检查涉及 HiSparse PR #57602、#60358。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
