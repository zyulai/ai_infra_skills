# PR #17344: [model-gateway] Optimize WASM cache lookups using SHA-256

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ppraneth |
| 合并 | 2026-01-26 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | model-gateway |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用(2 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/17344 |

## 一句话结论

sgl-model-gateway 的 WASM 组件缓存键从完整二进制换成 SHA-256 哈希,热查找约 2.7x(1.16 ms → 429 µs)。

## 动机

`WasmThreadPool` 的内部 `component_cache` 用原始 WASM 二进制(`Vec<u8>`)作为 LRU 缓存键,有两个问题:一是内存开销,大 WASM 组件的完整二进制作为键存进每个 worker 线程的 LRU;二是 CPU 开销,每次请求查找都要对整个二进制做哈希并可能进行逐字节 O(N) 比较来确认命中。模块注册时元数据里其实已经算好并保存了 SHA-256 哈希,可以直接作为唯一标识。

## 优化点

把组件缓存的键从 `LruCache<Vec<u8>, Component>` 改为 `LruCache<[u8; 32], Component>`:`WasmTask::ExecuteComponent` 增加 32 字节 `wasm_hash` 字段;`module_manager.rs` 的 `execute_module_interface` 从 `WasmModuleMeta` 取出预计算的 `sha256_hash` 随执行请求传给 runtime;worker 的 `execute_component_in_worker` 用定长 32 字节哈希做常数时间查找,原始 `wasm_bytes` 只在 cache miss 时用于编译,之后立即丢弃。附带把编译失败的错误包装修正为 `WasmError::Runtime(...)`。

## 关键改动

```text
sgl-model-gateway/src/wasm/runtime.rs: WasmTask 增加 wasm_hash;component_cache 键改为 [u8; 32]
sgl-model-gateway/src/wasm/module_manager.rs: 执行时从 WasmModuleMeta 取 sha256_hash 传给 runtime
```

```rust
let component = if let Some(comp) = cache.get(&wasm_hash) {
    comp.clone() // Component is just a handle (cheap clone)
} else {
    let comp = Component::new(engine, &wasm_bytes)...;
    cache.push(wasm_hash, comp.clone());
    comp
};
```

## 性能声明

PR 正文:用约 1MB 的 WASM 组件模拟大二进制比较的开销。优化前基线约 1.16 ms/次热查找,优化后(哈希键)约 429 µs/次热查找,缓存查找热路径约 **2.7x speedup**,并显著降低线程池的内存压力。PR 未注明测试硬件与更完整的负载条件。

## 验证与回归风险

diff 未新增测试,PR 的 Accuracy Tests 一节为空(Checklist 四项均未勾选)。风险面:改动限于 sgl-model-gateway 的 WASM 扩展执行路径,不影响模型推理;正确性依赖注册时 `sha256_hash` 与二进制内容一一对应(哈希碰撞或元数据哈希与实际 bytes 不一致时会取错缓存组件,PR 未讨论);`LruCache` 键类型变化是 Rust 编译期约束,风险主要在语义而非类型。

## 关联

正文未列出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
