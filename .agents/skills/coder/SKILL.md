---
name: coder
description: 负责实际开发
---



## 开发规范

### Python Python 
- 使用 Python 3.14+，需保持向后兼容（到3.10版本即可）
- 使用 ruff 进行代码格式化和 lint
- 使用 pyright 进行类型检查
- 测试使用 pytest
- 依赖管理使用 uv


代码风格：
- 行长度限制 100 字符
- 使用类型注解
- 公开函数需要 docstring
- 独立函数尽可能补充独立测试用例，并确保自验证通过


### 其他规范

- 禁止随意生成解释，think得到的数据，需要有外部信息（paper、code、白皮书等）作为数据和信息支撑；
- 开发过程中，如果涉及外部检索到的信息数据，需要刷新到``docs/data_sources_wiki.md``文件中作为参考，将变更日志刷新到md最开始；
- 每次新增特性、进行重构或解决bug，需整理本次prompt&开发过程的摘要，刷新到本地DEVELOP_LOG.md文件（需提交到仓库）和review.log文件（不提交到仓库，仅用于本次开发的临时记录）；


### 自验证规范

本地conda环境base可以用于程序调试，每次增加新特性，要求自验证通过。

自验证方式：
```bash
cd tests
python run_tests.py

# 或使用 pytest
pytest tests/
```

- 如果有测试样例是web服务相关的，无法通过python脚本直接验证，可以启动web服务进行验证。确保web服务功能ok。