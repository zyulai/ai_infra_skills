# PR #17400: [BUGFIX] fix value oom in radix tree

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17400](https://github.com/sgl-project/sglang/pull/17400) |
| 作者 | yizhang2077 |
| 合并日期 | 2026-01-21 |
| labels | run-ci |
| 热度 | 👍 4 · 💬 2 |
| 分类 | KV Cache / Memory |
| bug 类型 | resource |

## 一句话结论

radix tree 插入节点时直接引用外部 value 张量导致 value OOM;在三处插入点改为 `.clone()` 复制后修复。

## 问题现象

radix tree 的 value 出现 OOM(标题:"fix value oom in radix tree")。正文 Motivation 仅写 "fix missing part in #17191",未描述具体报错、触发场景或影响面;从分类(KV Cache / Memory)与 bug 类型(resource)看属内存资源类问题。

## 根因

正文未披露根因,只说明这是 #17191 漏改的部分。从 diff 看,HiRadixCache 与 MambaRadixCache 在插入新节点时把调用方传入的 `value` / `host_value` 张量直接赋给节点(`new_node.value = value` 等),节点与外部张量共享存储;树节点因此长期持有一份外部缓冲的引用(可能连带更大的底层 storage 无法释放或被外部改写),#17191 已在别处补了 `.clone()`,但漏掉了这些插入点(由 diff 推断)。

## 修复方案

给三处插入点的 value 赋值补上 `.clone()`,切断与外部张量的共享:

- `python/sglang/srt/mem_cache/hiradix_cache.py` `_insert_helper_host`:

  ```diff
  -            new_node.host_value = host_value
  +            new_node.host_value = host_value.clone()
  ```

- `hiradix_cache.py` `insert`:被驱逐节点重新插入 `value[:prefix_len]` 与全新节点 `value` 两处均改为克隆:

  ```diff
  -                    new_node.value = value[:prefix_len]
  +                    new_node.value = value[:prefix_len].clone()
  ...
  -            new_node.value = value
  +            new_node.value = value.clone()
  ```

- `python/sglang/srt/mem_cache/mamba_radix_cache.py` `_insert_helper`:`new_node.value = value.clone()`。

`mamba_value` 与其余字段赋值保持不变。

## 影响与回归风险

- 影响窗口:正文未给出版本范围;问题与 #17191 相关的改动窗口一致。
- 回归面:HiRadixCache(host value 路径与 insert 的两个分支)和 MambaRadixCache 的节点插入。克隆会带来一次张量复制,插入路径内存占用与耗时略增,但树节点持有独立副本后,外部缓冲可正常释放/复用,消除 OOM 的来源;查询命中语义不变。

## 关联

- #17191(本 PR 补齐其遗漏部分)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
