export interface ModelInfo {
  name: string
  display_name: string
  architecture: string
  sparse_type: string
}

export interface ModelSchema {
  name: string
  description: string
  architecture: string
  sparse_type: string
  attention_features: string[]
  supported_workloads: string[]
  config: ModelConfig
  param_schema: Record<string, ParamField[]>
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

export interface ParamField {
  key: string
  type: 'number' | 'string' | 'boolean' | 'select'
  default: number | string | boolean
  label: string
  description?: string
  min?: number
  max?: number
  options?: string[]
}