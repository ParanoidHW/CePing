import client from './client'
import type { WorkloadSchema } from '@/types'

export async function listWorkloads(): Promise<Record<string, string[]>> {
  const response = await client.get('/workloads')
  return response.data.categories
}

export async function getWorkloadSchema(name: string): Promise<WorkloadSchema> {
  const response = await client.get(`/workload/${name}`)
  return response.data
}