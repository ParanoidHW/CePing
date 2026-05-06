export interface EvaluationRequest {
  workload_name: string
  model_name: string
  hardware: HardwareSchema
  strategy: StrategySchema
  params: Record<string, unknown>
}

export interface EvaluationResult {
  success: boolean
  validation?: ValidationResult
  result?: PerformanceResult
  error?: string
}

export interface ValidationResult {
  is_valid: boolean
  errors: string[]
  warnings: string[]
  memory_warning?: string
}

export interface PerformanceResult {
  workload_type: string
  total_time_sec: number
  peak_memory_gb: number
  throughput: number
  throughput_metric: string
  metrics: Record<string, number>
  stages: StageResult[]
  breakdown: BreakdownResult
}

export interface StageResult {
  name: string
  time_sec: number
  memory_gb: number
  throughput?: number
}

export interface BreakdownResult {
  by_stage?: Record<string, StageBreakdown>
  by_phase?: Record<string, PhaseBreakdown>
  by_submodule?: Record<string, SubmoduleBreakdown>
  memory?: MemoryBreakdown
  communication?: CommunicationBreakdown
}

export interface StageBreakdown {
  time_sec: number
  memory_gb: number
  percentage: number
}

export interface PhaseBreakdown {
  name: string
  time_sec: number
  memory_gb: number
}

export interface SubmoduleBreakdown {
  name: string
  type: string
  time_sec: number
  memory_gb: number
  flops?: number
}

export interface MemoryBreakdown {
  parameters_gb: number
  activations_gb: number
  gradients_gb?: number
  optimizer_gb?: number
  kv_cache_gb?: number
}

export interface CommunicationBreakdown {
  all_reduce_gb: number
  all_gather_gb: number
  reduce_scatter_gb: number
  all_to_all_gb?: number
}