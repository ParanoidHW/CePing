# 阶段3设计方案：前端参数界面动态渲染

## 1. param_schema 字段定义格式

### 1.1 Schema 结构

```python
param_schema = {
    "training": [  # 训练模式参数列表
        {
            "name": "seq_len",           # 参数名（用于发送到后端）
            "label": "Sequence Length",  # 前端显示标签
            "type": "number",            # 输入类型: number, select
            "default": 4096,             # 默认值
            "min": 1,                    # 最小值（可选）
            "max": 32768,                # 最大值（可选）
        },
        ...
    ],
    "inference": [  # 推理模式参数列表
        {
            "name": "prompt_len",
            "label": "Prompt Length",
            "type": "number",
            "default": 1024,
        },
        ...
    ]
}
```

### 1.2 参数分组

为简化实现，参数分为两组：
- `training`: 训练模式所需参数
- `inference`: 推理模式所需参数

## 2. 各模型类型的 param_schema 定义

### 2.1 LLM 类模型（llama, mixtral, deepseek）

```python
"param_schema": {
    "training": [
        {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 32},
        {"name": "seq_len", "label": "Sequence Length", "type": "number", "default": 4096},
    ],
    "inference": [
        {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 8},
        {"name": "prompt_len", "label": "Prompt Length", "type": "number", "default": 1024},
        {"name": "generation_len", "label": "Generation Length", "type": "number", "default": 128},
    ]
}
```

### 2.2 视频生成类模型（wan-t2v-14b）

```python
"param_schema": {
    "training": [
        {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 1},
        {"name": "num_frames", "label": "Num Frames", "type": "number", "default": 81},
        {"name": "height", "label": "Height", "type": "number", "default": 720},
        {"name": "width", "label": "Width", "type": "number", "default": 1280},
    ],
    "inference": [
        {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 1},
        {"name": "num_frames", "label": "Num Frames", "type": "number", "default": 81},
        {"name": "height", "label": "Height", "type": "number", "default": 720},
        {"name": "width", "label": "Width", "type": "number", "default": 1280},
        {"name": "num_steps", "label": "Inference Steps", "type": "number", "default": 50},
        {"name": "use_cfg", "label": "Use CFG", "type": "select", "default": "true", "options": ["true", "false"]},
    ]
}
```

### 2.3 wan-dit 单独模型

```python
"param_schema": {
    "training": [
        {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 1},
        {"name": "num_frames", "label": "Num Frames", "type": "number", "default": 81},
        {"name": "height", "label": "Height", "type": "number", "default": 720},
        {"name": "width", "label": "Width", "type": "number", "default": 1280},
    ],
    "inference": [
        {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 1},
        {"name": "num_frames", "label": "Num Frames", "type": "number", "default": 81},
        {"name": "height", "label": "Height", "type": "number", "default": 720},
        {"name": "width", "label": "Width", "type": "number", "default": 1280},
        {"name": "num_steps", "label": "Inference Steps", "type": "number", "default": 50},
        {"name": "use_cfg", "label": "Use CFG", "type": "select", "default": "true", "options": ["true", "false"]},
    ]
}
```

### 2.4 ResNet/VAE 类模型

```python
# ResNet
"param_schema": {
    "training": [
        {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 32},
        {"name": "height", "label": "Image Height", "type": "number", "default": 224},
        {"name": "width", "label": "Image Width", "type": "number", "default": 224},
    ],
    "inference": [
        {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 32},
        {"name": "height", "label": "Image Height", "type": "number", "default": 224},
        {"name": "width", "label": "Image Width", "type": "number", "default": 224},
    ]
}

# VAE
"param_schema": {
    "training": [
        {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 8},
        {"name": "num_frames", "label": "Num Frames", "type": "number", "default": 17},
        {"name": "height", "label": "Height", "type": "number", "default": 256},
        {"name": "width", "label": "Width", "type": "number", "default": 256},
    ],
    "inference": [
        {"name": "batch_size", "label": "Batch Size", "type": "number", "default": 8},
        {"name": "num_frames", "label": "Num Frames", "type": "number", "default": 17},
        {"name": "height", "label": "Height", "type": "number", "default": 256},
        {"name": "width", "label": "Width", "type": "number", "default": 256},
    ]
}
```

## 3. 前端动态渲染逻辑

### 3.1 数据流

```
用户选择模型预设 → 加载 preset.param_schema → 渲染参数输入框 → 用户切换模式 → 更新显示的参数
```

### 3.2 核心函数

1. **loadModelPreset()**: 修改为读取 param_schema
2. **renderParamInputs()**: 新增，根据 param_schema 渲染参数区域
3. **collectConfig()**: 修改为根据 param_schema 收集参数

### 3.3 HTML 容器结构

保持现有 `training-params` 和 `inference-params` 容器，改为动态填充：

```html
<div id="training-params">
    <!-- 动态渲染训练参数 -->
</div>
<div id="inference-params" style="display: none;">
    <!-- 动态渲染推理参数 -->
</div>
```

## 4. 改动方案

### 4.1 后端改动 (registry.py)

- 为所有 preset 添加 `param_schema` 字段
- 保持向后兼容，旧字段（如 `max_seq_len`）保留

### 4.2 前端改动 (app.js)

主要改动点：

1. `loadModelPreset()` 中读取 `preset.param_schema`
2. 新增 `renderParamInputs(schema)` 函数
3. 修改 `collectConfig()` 移除硬编码 videoParams 逻辑
4. 修改 `updateModeUI()` 调用参数渲染

### 4.3 前端改动 (index.html)

- 无需修改 HTML 结构，使用现有容器
- 或可添加一个动态容器用于特殊参数

## 5. 向后兼容

- 如果 preset 没有 param_schema，使用默认 LLM 参数
- 保持 `state.currentPipeline` 逻辑用于后端路由选择