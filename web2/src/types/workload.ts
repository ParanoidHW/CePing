export interface WorkloadCategory {
  category: string
  workloads: WorkloadInfo[]
}

export interface WorkloadInfo {
  name: string
  category: string
  description: string
  workload_type: string
}

export interface WorkloadSchema {
  name: string
  display_name: string
  workload_name: string
  description: string
  category: string
  workload_type: string
  compute_mode: string
  stages: StageSchema[]
  parameters: Record<string, ParamSchema>
  throughput_metric: string
  supported_models: string[]
}

export interface StageSchema {
  name: string
  compute_type: string
  component: string
  repeat: number | string
  compute_pattern?: string
  extra_params?: Record<string, unknown>
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