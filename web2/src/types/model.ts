export interface ModelInfo {
  name: string
  display_name: string
  architecture: string
  sparse_type: string
  supported_workloads: string[]
}

export interface ModelSchema {
  name: string
  description: string
  architecture: string
  sparse_type: string
  attention_features: string[]
  supported_workloads: string[]
  config: ModelConfig
  param_schema: Record<string, ParamSchema[]>
}

export interface ModelConfig {
  hidden_size: number
  num_layers: number
  num_heads: number
  vocab_size: number
  intermediate_size?: number
  head_dim?: number
  max_position_embeddings?: number
}

export interface ParamSchema {
  name: string
  label: string
  type: 'number' | 'string' | 'boolean'
  default: number | string | boolean | null
  min?: number
  max?: number
  required: boolean
  description: string
}