import client from './client'
import type { ModelInfo, ModelSchema } from '@/types'

export async function listModels(workload?: string): Promise<ModelInfo[]> {
  const response = await client.get('/models', {
    params: workload ? { workload } : undefined
  })
  return response.data.models
}

export async function getModelSchema(name: string): Promise<ModelSchema> {
  const response = await client.get(`/model/${name}`)
  return response.data
}