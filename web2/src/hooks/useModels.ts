import { useState, useEffect } from 'react'
import { listModels, getModelSchema } from '@/api'
import type { ModelInfo, ModelSchema } from '@/types'

export function useModels(workload?: string) {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    listModels(workload)
      .then(setModels)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [workload])

  return { models, loading, error }
}

export function useModelSchema(name: string | null) {
  const [schema, setSchema] = useState<ModelSchema | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!name) {
      setSchema(null)
      return
    }
    setLoading(true)
    getModelSchema(name)
      .then(setSchema)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [name])

  return { schema, loading, error }
}