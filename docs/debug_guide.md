# Debug Guide (维测指南)

本文档介绍如何收集和提供调试信息，以便快速定位和解决问题。

---

## 一、遇到问题时如何收集信息

### 1. 使用前端调试面板（推荐）

**触发方式**：
- 快捷键：`Ctrl + Shift + D`
- 或点击页面右上角 "🔧 Debug" 按钮

**使用步骤**：
1. 打开调试面板
2. 在 "State" Tab 检查当前配置
3. 在 "Requests" Tab 查看最近的 API 请求
4. 在 "Errors" Tab 查看 Console 错误
5. 点击 "Export JSON" 下载调试信息

**导出内容**：
- 当前状态（workload、model、hardware、strategy、params）
- 最近 20 条 API 请求/响应
- 最近 50 条 Console 错误
- 环境信息（浏览器、URL、前端 commit）

### 2. 查看后端日志

后端日志位于 `logs/` 目录：

```bash
# 查看评估日志（详细）
cat logs/evaluate.log

# 查看错误日志
cat logs/error.log

# 查看所有日志（实时）
tail -f logs/*.log

# 查看最近的错误
grep "ERROR" logs/evaluate.log | tail -20
```

**日志文件说明**：
- `api.log`：API 请求日志（INFO 级别）
- `evaluate.log`：评估过程详细日志（DEBUG 级别）
- `error.log`：错误日志（ERROR 级别）

**日志格式**（JSON 结构化）：
```json
{
  "timestamp": "2026-05-07T12:30:00",
  "level": "INFO",
  "step": "evaluate_start",
  "data": {
    "request": {...}
  }
}
```

### 3. 使用浏览器 DevTools

**打开 DevTools**：
- Windows/Linux: `F12` 或 `Ctrl + Shift + I`
- Mac: `Cmd + Option + I`

**Console 标签**：
- 查看错误日志
- 查看网络请求失败原因

**Network 标签**：
- 查看所有网络请求
- 检查请求和响应内容
- 查看请求耗时

**Application 标签**：
- 查看 LocalStorage
- 查看 SessionStorage
- 查看浏览器缓存

---

## 二、调试信息 JSON 格式说明

### 完整格式示例

```json
{
  "debug_info": {
    "timestamp": "2026-05-07T12:30:00",
    "frontend": {
      "state": {
        "workload": "inference/autoregressive",
        "model": "llama-7b",
        "hardware": {
          "device_preset": "H100-SXM-80GB",
          "num_devices": 8,
          "topology_type": "mesh"
        },
        "strategy": {
          "tp_degree": 8,
          "pp_degree": 1,
          "dp_degree": 1,
          "ep_degree": 1,
          "sp_degree": 1,
          "activation_checkpointing": false,
          "zero_stage": 0
        },
        "params": {
          "batch_size": 32,
          "seq_len": 4096
        }
      },
      "console_errors": [
        {
          "timestamp": "2026-05-07T12:28:15",
          "type": "error",
          "message": "TypeError: Cannot read property 'get' of undefined",
          "stack": "Error: ...\n    at Object.get (...)",
          "filename": "http://localhost:5173/src/api/evaluate.ts",
          "lineno": 15,
          "colno": 10
        }
      ],
      "network": {
        "requests": [
          {
            "timestamp": "2026-05-07T12:29:30",
            "method": "POST",
            "url": "/api/evaluate",
            "status": 500,
            "request_body": {
              "workload_name": "inference/autoregressive",
              "model_name": "llama-7b",
              "hardware": {...},
              "strategy": {...},
              "params": {...}
            },
            "response_body": {
              "success": false,
              "error": "AttributeError: 'str' object has no attribute 'get'"
            },
            "error": "Request failed with status code 500"
          }
        ]
      }
    },
    "environment": {
      "frontend_commit": "08a38e7",
      "browser": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
      "url": "http://localhost:5173/"
    }
  }
}
```

### 字段说明

#### 1. frontend.state
- `workload`: 当前选择的 workload 类型
- `model`: 当前选择的模型名称
- `hardware`: 硬件配置（设备、数量、拓扑）
- `strategy`: 并行策略（TP/PP/DP/EP/SP、activation checkpointing、zero stage）
- `params`: workload 参数（batch_size、seq_len 等）

#### 2. frontend.console_errors
- `timestamp`: 错误发生时间
- `type`: 错误类型（error / unhandledrejection / console_error）
- `message`: 错误信息
- `stack`: 调用栈（可选）
- `filename`: 发生错误的文件名（可选）
- `lineno` / `colno`: 行号和列号（可选）

#### 3. frontend.network.requests
- `timestamp`: 请求时间
- `method`: HTTP 方法（GET/POST/PUT/DELETE）
- `url`: 请求 URL
- `status`: HTTP 状态码（可选）
- `request_body`: 请求体（可选）
- `response_body`: 响应体（可选）
- `error`: 错误信息（可选）

#### 4. environment
- `frontend_commit`: 前端 Git commit hash
- `browser`: 浏览器 User Agent
- `url`: 当前页面 URL

---

## 三、提供信息给开发者的方式

### 推荐方式

1. **前端调试信息**：
   - 使用调试面板导出 JSON 文件
   - 或复制 JSON 内容到剪贴板

2. **后端日志**：
   - 提供 `logs/evaluate.log` 文件内容
   - 或提供 `logs/error.log` 文件内容

3. **环境信息**：
   - 操作系统版本（如 Windows 11、Ubuntu 22.04）
   - 浏览器版本（如 Chrome 120.0.6099.130）
   - Python 版本（如 3.10.12）
   - Node.js 版本（如 20.10.0）

4. **重现步骤**：
   - 详细描述操作步骤
   - 提供截图或录屏（可选）

### Issue 模板

```markdown
## 问题描述
简要描述遇到的问题

## 重现步骤
1. 选择 workload: inference/autoregressive
2. 选择 model: llama-7b
3. 配置 hardware: ...
4. 点击 "Run Evaluation"
5. 出现错误

## 环境信息
- 操作系统: Windows 11
- 浏览器: Chrome 120.0.6099.130
- Python: 3.10.12
- Node.js: 20.10.0
- 前端 commit: 08a38e7

## 调试信息
[粘贴调试信息 JSON]

## 后端日志
[粘贴 logs/evaluate.log 相关内容]

## 截图
[可选：粘贴截图]
```

---

## 四、常见问题排查

### 问题 1：评估失败，返回 500 错误

**排查步骤**：
1. 查看前端调试面板 "Errors" Tab
2. 查看 "Requests" Tab 中的响应内容
3. 查看后端 `logs/evaluate.log` 日志
4. 检查后端日志中的 `step` 字段，定位失败步骤

**常见原因**：
- 配置参数错误（如 `tp_degree > num_devices`）
- 模型配置文件错误
- 后端代码异常

### 问题 2：前端白屏或无响应

**排查步骤**：
1. 打开浏览器 DevTools Console 标签
2. 查看是否有 JavaScript 错误
3. 使用前端调试面板导出错误信息
4. 检查 Network 标签，查看 API 请求是否成功

**常见原因**：
- JavaScript 异常
- API 请求超时
- 前端代码 bug

### 问题 3：配置参数错误

**排查步骤**：
1. 查看前端调试面板 "State" Tab
2. 检查 workload、model、hardware、strategy 配置是否正确
3. 查看后端日志中的 validation 步骤

**常见原因**：
- 必填字段未填写
- 参数类型错误（如字符串填数字）
- 参数范围错误（如 tp_degree = 0）

### 问题 4：网络请求失败

**排查步骤**：
1. 查看前端调试面板 "Requests" Tab
2. 查看浏览器 DevTools Network 标签
3. 检查后端是否启动（`flask run`）
4. 检查后端日志 `logs/api.log`

**常见原因**：
- 后端未启动
- 端口被占用
- CORS 配置错误
- 网络超时

---

## 五、高级调试技巧

### 5.1 查看后端详细日志

```bash
# 查看所有 INFO 及以上级别的日志
cat logs/evaluate.log | grep '"level": "INFO"'

# 查看特定步骤的日志
cat logs/evaluate.log | grep '"step": "evaluate_start"'

# 查看错误日志
cat logs/error.log

# 实时查看日志
tail -f logs/evaluate.log
```

### 5.2 前端断点调试

1. 打开浏览器 DevTools
2. 进入 Sources 标签
3. 找到对应的源文件（如 `App.tsx`）
4. 点击行号设置断点
5. 刷新页面，触发断点
6. 使用 Step Over / Step Into / Step Out 调试

### 5.3 后端断点调试

**使用 VS Code 调试**：
1. 创建 `.vscode/launch.json` 配置
2. 设置断点
3. 启动调试（F5）

**使用 pdb 调试**：
```python
import pdb; pdb.set_trace()
```

### 5.4 检查模型配置

```bash
# 查看模型配置文件
cat configs/models/llama-7b.yaml

# 查看 workload 配置文件
cat configs/workloads/inference/autoregressive.yaml
```

---

## 六、性能监控（P2 功能，计划实现）

### 计划功能
- [ ] 前端性能面板（加载时间、渲染性能）
- [ ] 后端性能日志（评估耗时分解）
- [ ] 性能趋势图表
- [ ] 性能告警

---

**如有其他问题，请提供调试信息并提交 Issue。**