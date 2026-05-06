export interface HardwareInfo {
  name: string
  vendor: string
  memory_gb: number
  tflops: number
  bandwidth_gb_s: number
}

export interface TopologyInfo {
  name: string
  description: string
  bandwidth_gb_s: number
}

export interface HardwareSchema {
  device_preset: string
  num_devices: number
  topology_type: string
  custom_topology?: Record<string, unknown>
}

export interface StrategySchema {
  tp_degree: number
  pp_degree: number
  dp_degree: number
  ep_degree: number
  sp_degree: number
  activation_checkpointing: boolean
  zero_stage: number
}

export type { WorkloadCategory, WorkloadInfo, WorkloadSchema, StageSchema } from './workload'
export type { ModelInfo, ModelSchema, ModelConfig, ParamSchema } from './model'
export type {
  EvaluationRequest,
  EvaluationResult,
  ValidationResult,
  PerformanceResult,
  StageResult,
  BreakdownResult,
  StageBreakdown,
  PhaseBreakdown,
  SubmoduleBreakdown,
  MemoryBreakdown,
  CommunicationBreakdown
} from './result'