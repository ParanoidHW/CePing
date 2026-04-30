import { useState } from 'react'
import { evaluate } from '@/api'
import type { EvaluationRequest, EvaluationResult } from '@/types'

export function useEvaluate() {
  const [result, setResult] = useState<EvaluationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runEvaluate = async (request: EvaluationRequest) => {
    setLoading(true)
    setError(null)
    setResult(null)
    
    try {
      const data = await evaluate(request)
      setResult(data)
      return data
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Evaluation failed')
      throw err
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setResult(null)
    setError(null)
  }

  return { result, loading, error, runEvaluate, reset }
}