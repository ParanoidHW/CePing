---
name: new-kernel
description: 新增kernel评估时的行为
---

## 新kernel评估支持

评估新kernel时
- kernel的注册方式需尽量对齐torch原生对应接口，入参数量和形式保持和torch接口一致。
- 参考的torch版本为2.10