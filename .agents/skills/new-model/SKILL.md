---
name: new-model
description: 新增模型时行为
---

## 新模型评估支持

- 优先从[Huggingface](https://huggingface.co/models)上检索和获取**官方模型**的结构文件；
- 严格根据结构配置文件的超参进行评估，不得随意修改其中任何一个模型配置参数；如有不明确的结构参数，需进行备注；
- 如果Hugginface上有多个模型配置文件，并且无法区分主要backbone，同时支持评估，并在DEVELOP_LOG里进行提示，提示我进行区分；
