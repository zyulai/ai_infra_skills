# PR #5778: fix bug that gpu0 occupies more memory when hicache is turned on

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5778](https://github.com/sgl-project/sglang/pull/5778) |
| 作者 | huangtingwei9988 |
| 合并日期 | 2025-05-09 |
| labels | (none) |
| 热度 | 👍 7 · 💬 2 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

## 一句话结论
开启 hicache 后 GPU0 显存明显偏高:异步线程默认设备落在 device0;线程启动时切换到对应 scheduler 设备修复。

## 问题现象
开启 hierarchical cache(hicache)后,GPU0 设备的显存占用显著高于其他设备。作者在 tp=8 DeepSeek R1 上观察到该现象(截图);用 pynvml 脚本在 tp4 DeepSeek-V2-Lite 上复现,并量化为 H20 上 GPU0 需多占 320M * (tp - 1) 显存(附测量脚本)。

## 根因
正文披露:HiCacheController 使用异步多线程(write_thread 与 load_thread)和多 stream(write_stream 与 load_stream)。异步多线程会丢失部分 cuda context 信息,导致这些异步线程的默认 current_device 全是 device0;对 GPU0 以外的设备,执行 `with torch.cuda.stream(write_stream or load_stream)` 时 current_device 需要从 device0 切到其他设备、结束后再切回 device0,这一反复切换使 GPU0 占用更多显存。

## 修复方案
正文:多线程启动时,把 current_device 切换到对应 scheduler 进程所在的设备。按 diff,`python/sglang/srt/managers/cache_controller.py` 中 5 个线程函数(write_thread_func_direct、load_thread_func_direct、load_thread_func_layer_by_layer、write_aux_func、load_thread_func_buffer)把 `with torch.cuda.stream(...):` 上下文包裹改为线程入口处直接 `torch.cuda.set_stream(...)`,线程循环体整体去嵌套:

```diff
-        with torch.cuda.stream(self.write_stream):
-            while not self.stop_event.is_set():
-                try:
-                    operation = self.write_queue.get(block=True, timeout=1)
+        torch.cuda.set_stream(self.write_stream)
+        while not self.stop_event.is_set():
+            try:
+                operation = self.write_queue.get(block=True, timeout=1)
```

## 影响与回归风险
- 影响窗口:正文未给出版本/提交范围。
- 回归面:按 diff,仅 `cache_controller.py` 中 hicache 写/读线程的 stream 绑定方式变化(直写、layer-by-layer、buffer 三种读写模式及 aux 辅助线程);stream 由 with 块改为线程生命周期内一次性 set,不再随块退出恢复,影响这些线程内的 CUDA stream 语义;其余模块未动。

## 关联
- [#5372](https://github.com/sgl-project/sglang/issues/5372)(正文 Related issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
