import client from './client'
import type { EvaluationRequest, EvaluationResult } from '@/types'

export async function evaluate(request: EvaluationRequest): Promise<EvaluationResult> {
  const response = await client.post('/evaluate', request)
  return response.data
}