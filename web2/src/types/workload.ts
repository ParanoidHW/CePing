export interface WorkloadCategory {
  category: string
  workloads: WorkloadInfo[]
}

export interface WorkloadInfo {
  name: string
  display_name: string
  description: string
}

export interface WorkloadSchema {
  name: string
  display_name: string
  description: string
  workload_type: string
  compute_mode: string
  stages: StageSchema[]
  parameters: Record<string, ParamSchema>
  supported_models: string[]
  throughput_metric?: string
}

export interface StageSchema {
  name: string
  compute_type: string
  component: string
  repeat: number
}

export interface ParamSchema {
  type: 'number' | 'string' | 'boolean' | 'select'
  default: number | string | boolean
  label: string
  description?: string
  min?: number
  max?: number
  options?: string[]
  required?: boolean
}