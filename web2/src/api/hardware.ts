import client from './client'
import type { HardwareInfo, TopologyInfo } from '@/types'

export async function listHardware(): Promise<Record<string, HardwareInfo[]>> {
  const response = await client.get('/hardware')
  return response.data.devices
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