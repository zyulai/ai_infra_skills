# PR #18124: [Perf] Use safetensors `load_file` in multithread loader

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mmangkad |
| 合并 | 2026-02-03 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | run-ci |
| Diff | 素材未提供 diffstat;diff 为单文件小改动(1 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/18124 |

## 一句话结论

多线程加载器改用 Rust 后端的 `safetensors.load_file`,Kimi-K2.5 分片加载快 7.2x、端到端权重加载快 4x。

## 动机

`multi_thread_safetensors_weights_iterator` 原先用 `safetensors.safe_open` 打开文件后,在 Python 层逐 key 迭代提取 tensor,per-key 的 Python/元数据开销在超大模型(几十个 10 GB 级分片)上非常可观。改走 `safetensors.torch.load_file(..., device="cpu")` 可把"一次性加载全部 tensor"的 eager 路径交由 Rust 实现的 safetensors 加载器完成。这部分开销削减后,模型加载往往最终受限于存储 I/O——mmap 式访问在网络挂载上因大量小而随机的读会特别慢。

## 优化点

改动极小且聚焦:把 `_load_file` 中非 disable_mmap 分支的手动循环 `with safetensors.safe_open(...): result = {k: f.get_tensor(k) for k in f.keys()}` 替换为一行 `safetensors.torch.load_file(st_file, device="cpu")`,由 Rust 后端批量加载。`disable_mmap` 分支(读全量字节后 `safetensors.torch.load`)行为不变。多线程分发的 `ThreadPoolExecutor` 结构不动,每个分片仍由独立线程加载。

## 关键改动

```text
python/sglang/srt/model_loader/weight_utils.py: _load_file 的 mmap 分支改用 safetensors.torch.load_file,消除 Python 层逐 key 提取
```

```python
def _load_file(st_file: str):
    if disable_mmap:
        with open(st_file, "rb") as f:
            return safetensors.torch.load(f.read())
    return safetensors.torch.load_file(st_file, device="cpu")
```

## 性能声明

测试条件:`--model-loader-extra-config '{"enable_multithread_load": true,"num_threads": 64}'`,模型 Kimi-K2.5(64 个分片、每片约 10 GB,bf16,TP8),同一台机器与存储、无缓存:

- 分片加载阶段:18:24 → 02:33,**7.2x faster**(日志:`Loading weights took` 由 1456.72 s 降至 365.50 s)。
- 端到端权重加载总时长:约 24 分 46 秒 → 约 6 分 11 秒,**4x speedup**,整体等待时间减少 **75%**。

## 验证与回归风险

PR 未新增单元测试(checklist 的 format/tests/docs/benchmark 项均为未勾选),验证以同机同存储的加载日志对比为主。影响面仅限 `enable_multithread_load` 的多线程加载路径;非 mmap 分支从"逐 key 惰性提取"改为"整文件一次性加载",与改动前实际构建完整 dict 的行为等价,但 CPU 侧峰值内存时序可能略有不同;`device="cpu"` 固定,后续 GPU 直载不在此路径。回归风险总体很低。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
