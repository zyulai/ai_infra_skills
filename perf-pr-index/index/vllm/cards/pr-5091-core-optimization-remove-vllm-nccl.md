# PR #5091: [Core][Optimization] remove vllm-nccl

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @youkaichao |
| 合并 | 2024-05-29 |
| 分类 | Parallelism / Communication（tier: major） |
| Label | 无 |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 显示改动 7 个文件：`.buildkite/test-pipeline.yaml`、`requirements-cuda.txt`、`setup.py`、`tests/distributed/test_pynccl_library.py`（删除）、`vllm/distributed/device_communicators/pynccl_wrapper.py`、`vllm/utils.py`、`vllm/worker/worker_base.py` |
| 链接 | https://github.com/vllm-project/vllm/pull/5091 |

## 一句话结论

去掉 vLLM 自带的 vllm-nccl 库依赖，改用 PyTorch 自带 NCCL 并全局设 `NCCL_CUMEM_ENABLE=0`，解决 NCCL 2.19 cuMem 在 cudagraph capture 时额外占内存的问题。

## 动机

作者经数月排查定位到根因：NCCL 2.19 起默认开启 cuMem（`NCCL_CUMEM_ENABLE` 自动启用，2.18 默认关闭），该特性在 cudagraph capture 期间消耗内存；vLLM 升级到 PyTorch 2.2 时恰好踩中此问题。此前 vLLM 在 requirements 里锁定 `vllm-nccl-cu12>=2.18,<2.19` 并单独分发 NCCL 库来规避。解决方案很简单：加一个环境变量关闭 cuMem 即可，之后可以直接使用 PyTorch 自带的 NCCL 库，不再需要维护 `vllm-nccl` 分发渠道。

## 优化点

三条线同时收口。第一，依赖裁剪：从 `requirements-cuda.txt` 与 `setup.py` 移除 `vllm-nccl-cu12` 及其按 CUDA 大版本改写包名的逻辑。第二，库查找简化：`find_nccl_library` 不再搜索 vllm-managed 的 `${VLLM_CONFIG_ROOT}/vllm/nccl/cu*/libnccl.so.*`，直接返回 `libnccl.so.2`（CUDA）或 `librccl.so.1`（ROCm），依赖导入 torch 后 `ctypes` 可自动解析；`pynccl_wrapper` 删除子进程 `ldd` 完整性预检（`nccl_integrity_check`），改为直接 `ctypes.CDLL` 加载并按路径缓存。第三，问题规避：在 `WorkerBase.init_worker` 初始化 worker 之前统一注入 `os.environ['NCCL_CUMEM_ENABLE'] = '0'`，关闭 NCCL 的 cuMem 分配以恢复 cudagraph capture 期间的内存表现。

## 关键改动

```text
vllm/worker/worker_base.py: init_worker 前注入 NCCL_CUMEM_ENABLE=0（核心规避手段）
vllm/utils.py: 删除 nccl_integrity_check 与 vllm-managed NCCL 查找，find_nccl_library 改用 torch 自带库
vllm/distributed/device_communicators/pynccl_wrapper.py: 移除子进程完整性检查，直接 CDLL 加载
requirements-cuda.txt / setup.py: 移除 vllm-nccl-cu12 依赖
tests/distributed/test_pynccl_library.py: 删除（连同 CI step）
```

```python
def init_worker(self, *args, **kwargs):
    enable_trace_function_call_for_thread()
    # see https://github.com/NVIDIA/nccl/issues/1234
    os.environ['NCCL_CUMEM_ENABLE'] = '0'
```

## 性能声明

PR 正文未给出量化 benchmark 数字，属内存问题修复：NCCL 2.19 的 cuMem 默认开启会在 cudagraph capture 期间占用内存，本 PR 通过关闭该选项消除这一开销并移除独立 NCCL 分发（性能收益体现为内存占用恢复，PR 未注明具体节省量）。

## 验证与回归风险

删除 `tests/distributed/test_pynccl_library.py`（其完整性检查逻辑已随实现移除），并在 `.buildkite/test-pipeline.yaml` 移除对应 CI step；分布式正确性测试（`test_basic_distributed_correctness`、`test_chunked_prefill_distributed`）保留作为回归防线。风险面：全局设置 `NCCL_CUMEM_ENABLE=0` 影响所有 NCCL 使用路径（pyNCCL 与 torch.distributed 共享进程环境），若某些 NCCL 版本/拓扑依赖 cuMem 带来的地址窗口优势可能受影响（PR 未讨论）；用户仍可通过 `VLLM_NCCL_SO_PATH` 指定自备库。

## 关联

代码注释引用 NVIDIA/nccl#1234；触发背景是 PyTorch 2.2/2.3 升级。PR 正文与 diff 未链接其他 vLLM PR，无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
