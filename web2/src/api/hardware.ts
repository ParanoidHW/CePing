import client from './client'
import type { HardwareInfo, TopologyInfo } from '@/types'

export async function listHardware(): Promise<Record<string, HardwareInfo[]>> {
  const response = await client.get('/hardware')
  const { devices, device_details } = response.data
  
  const result: Record<string, HardwareInfo[]> = {}
  for (const [vendor, names] of Object.entries(devices) as [string, string[]][]) {
    result[vendor] = names.map((name: string) => {
      const details = device_details[name] || {}
      return {
        name,
        vendor,
        memory_gb: details.memory_gb || 0,
        tflops: details.fp16_tflops_cube || details.bf16_tflops_cube || 0,
        bandwidth_gb_s: details.memory_bandwidth_gbps || 0,
      }
    })
  }
  return result
}

export async function listTopologies(): Promise<TopologyInfo[]> {
  const response = await client.get('/hardware/topologies')
  return response.data.topologies
}

export async function getResources(): Promise<{
  devices: Record<string, HardwareInfo[]>
  topologies: TopologyInfo[]
}> {
  const response = await client.get('/resources')
  return response.data
}