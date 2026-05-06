import { useState, useEffect } from 'react'
import { listWorkloads, getWorkloadSchema } from '@/api'
import type { WorkloadSchema } from '@/types'

export function useWorkloads() {
  const [categories, setCategories] = useState<Record<string, string[]>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    listWorkloads()
      .then(setCategories)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return { categories, loading, error }
}

export function useWorkloadSchema(name: string | null) {
  const [schema, setSchema] = useState<WorkloadSchema | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!name) {
      setSchema(null)
      return
    }
    setLoading(true)
    getWorkloadSchema(name)
      .then(setSchema)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [name])

  return { schema, loading, error }
}