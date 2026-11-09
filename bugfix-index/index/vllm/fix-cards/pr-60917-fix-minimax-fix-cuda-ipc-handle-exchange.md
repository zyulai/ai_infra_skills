# PR #60917: [Fix] MiniMax: fix CUDA IPC handle exchange on cuda-bindings >= 13.4

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#60917](https://github.com/vllm-project/vllm/pull/60917) |
| 作者 | laithsakka |
| 合并日期 | 2026-10-12 |
| labels | ready, nvidia, minimax |
| 热度 | 👍 0 · 💬 10 |
| 分类 | Model-specific Fast Path |
| bug 类型 | hang |

## 一句话结论
cuda-bindings ≥13.4 移除 cudaIpcMemHandle_t.reserved 致 handle 交换失败回退;改经 getPtr() 复制 64 字节。

## 问题现象
cuda-bindings 13.4(cuda-python)下报错:
```
AttributeError: 'cuda.bindings.runtime.cudaIpcMemHandle_t' object has no attribute 'reserved'
```
vLLM 随即回退到较慢的非 fused 路径并打印 "Falling back to the eager allreduce",`test_minimax_reduce_rms_qk` 失败。该问题出现在 torch-nightly CI lane,因为该环境安装 cuda-bindings 13.4.x。修复后 `pytest tests/kernels/core/test_minimax_reduce_rms.py::test_minimax_reduce_rms_qk -k "2-"` 为 6 passed, 12 skipped(只有 2 块 GPU,4/8-GPU 用例跳过),fallback 消息消失,fused 路径正常运行。

## 根因
正文披露:MiniMax fused allreduce+RMSNorm 内核需要每块 GPU 直接读其他 GPU 的内存。为完成设置,每块 GPU 分配 buffer、取得 64 字节 IPC handle 并发给其他 GPU 打开。vLLM 通过 `.reserved` 字段把 handle 转成 bytes 再转回;cuda-bindings 13.4 移除了 cudaIpcMemHandle_t 的该字段,代码随之失败。

## 修复方案
`vllm/model_executor/layers/minimax_rms_norm/lamport_workspace.py`:新增 `_IPC_HANDLE_SIZE = 64` 与两个函数,改为经 `getPtr()`(handle 的内存地址)复制字节:
```python
def _ipc_handle_to_bytes(handle) -> bytes:
    return ctypes.string_at(handle.getPtr(), _IPC_HANDLE_SIZE)

def _ipc_handle_from_bytes(raw: bytes):
    assert len(raw) == _IPC_HANDLE_SIZE, len(raw)
    handle = cudart.cudaIpcMemHandle_t()
    ctypes.memmove(handle.getPtr(), raw, _IPC_HANDLE_SIZE)
    return handle
```
`all_gather_object` 改为收集 `_ipc_handle_to_bytes(local_handle)`;对端不再 `handle.reserved = all_handles[r]`,改用 `_ipc_handle_from_bytes(raw)` 构造 handle 后 `cudaIpcOpenMemHandle`。所有 cuda-bindings 版本都支持 `getPtr()`,因此无需版本检查(CUDA ABI 把 handle 定为不透明 64 字节结构 `char reserved[64]`)。改动仅限该单文件。

## 影响与回归风险
- 回归面:仅 MiniMax lamport workspace 的 IPC handle 序列化/反序列化路径;旧版 cuda-bindings 上 `getPtr()` 同样可用,行为等价,不改变 fused allreduce 本身逻辑。

## 关联
Fixes https://github.com/vllm-project/vllm/issues/58599、https://github.com/pytorch/test-infra/issues/8931

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
