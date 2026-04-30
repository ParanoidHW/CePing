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
  device: string
  num_devices: number
  topology?: string
}

export interface StrategySchema {
  tp_degree: number
  pp_degree: number
  dp_degree: number
  sp_degree?: number
}

export * from './workload'
export * from './model'
export * from './result'