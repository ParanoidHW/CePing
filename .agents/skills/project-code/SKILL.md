---
name: project-code
description: 项目开发规范，包括代码风格、测试和依赖管理。编码规范需要遵守``开发规范``小节
---

## 开发规范

### Python Python 
- 使用 Python 3.14+，需保持向后兼容（到3.10版本即可）
- 使用 ruff 进行代码格式化和 lint
- 使用 pyright 进行类型检查
- 测试使用 pytest
- 依赖管理使用 uv
- 本地conda环境base可以用于程序调试，每次增加新特性，要求自验证通过

代码风格：
- 行长度限制 100 字符
- 使用类型注解
- 公开函数需要 docstring
- 独立函数尽可能补充独立测试用例，并确保自验证通过

自验证方式：
```bash
cd tests
python run_tests.py

# 或使用 pytest
pytest tests/
```

### Git 提交规范

如果没有特殊申明，你可以在修改并自验证完成后提交到本地（甚至是远程仓库，如果有权限的话），但需要自验证用例（包括新增的）全部通过后才可以提交。

使用 Conventional Commits 格式：

\[类型(范围)\]: 描述

允许的类型：feat, fix, docs, style, refactor, test, chore

示例：
- \[feat(auth)\]: 添加 OAuth 登录支持
- \[fix(api)\]: 修复用户查询返回空值的问题
- \[docs(readme)\]: 更新安装说明


### 其他规范

- 禁止随意生成解释，所有信息输入或者think得到的信息，需要有外部信息（paper、code、白皮书等）作为数据和信息支撑；
- 开发过程中，如果涉及外部检索到的信息数据，需要刷新到``docs/data_sources_wiki.md``文件中作为参考；
- 每次新增特性、进行重构或解决bug，需整理本次prompt&开发过程的摘要，刷新到本地DEVELOP_LOG.md文件；


## 新增特性需求原则

- 新增模型评估时，需通过subagent调用``new-model``技能，对新模型进行评估支持；
- 新增kernel评估时，需通过subagent调用``new-kernel``技能，对新kernel进行评估支持；