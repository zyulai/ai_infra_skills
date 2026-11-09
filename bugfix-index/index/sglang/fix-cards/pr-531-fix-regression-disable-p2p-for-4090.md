# PR #531: Fix Regression: Disable p2p for 4090

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#531](https://github.com/sgl-project/sglang/pull/531) |
| 作者 | ZX-ModelCloud |
| 合并日期 | 2024-06-12 |
| labels | (none) |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
RTX 40 系无 P2P,「假设 P2P 恒可用」的 monkey patch 致 4090 双卡初始化崩溃;按设备名跳过该 patch。

## 问题现象
正文给出 4090 双卡启动时的 stacktrace:custom_all_reduce.cuh:307 报 `Cuda error ... 'peer access is not supported between these two devices'`,随后 producer 进程提前终止(CudaIPCTypes 警告)、rpyc 连接被对端关闭(`EOFError: connection closed by peer`),router 初始化失败(`Initialization failed. router_init_state`),detokenizer 初始化正常。

## 根因
正文未披露根因分析,仅贴出 stacktrace。(由 diff 推断)sglang 此前用 `monkey_patch_vllm_p2p_access_check()` 把 vllm 的 `gpu_p2p_access_check` 无条件替换为恒返回 True,函数注释自述「假设 p2p access 总是允许,这在某些环境下可能出错」;RTX 40 系不支持 P2P,该假设失效后 custom all reduce 尝试启用 peer access 即报 `peer access is not supported`。本 PR 让该 patch 接收 gpu_id、查询设备名,检测到 "RTX 40" 时不打补丁,交由 vllm 原生 p2p 检查处理。

## 修复方案
- `python/sglang/srt/utils.py`:`monkey_patch_vllm_p2p_access_check` 增加 `gpu_id` 参数,仅当设备名不含 "RTX 40" 时才把 vllm 的 `gpu_p2p_access_check` 替换为恒 True;
- `python/sglang/srt/managers/controller/model_runner.py`:调用处改为传入 `self.gpu_id`。

```diff
-    def monkey_patch_vllm_p2p_access_check():
+    def monkey_patch_vllm_p2p_access_check(gpu_id: int):
+        device_name = torch.cuda.get_device_name(gpu_id)
+        if "RTX 40" not in device_name:
+            import vllm.distributed.device_communicators.custom_all_reduce_utils as tgt
+            setattr(tgt, "gpu_p2p_access_check", lambda *arg, **kwargs: True)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:TP 初始化时的 p2p monkey patch 路径(controller/model_runner);RTX 40 系多卡将走 vllm 真实的 p2p 检查(检查可能变慢,但不再因错误假设崩溃);非 RTX 40 设备行为不变;diff 注释自述设备名匹配「need a better check than just dev str name match」,按名称前缀匹配可能误伤同前缀设备。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
