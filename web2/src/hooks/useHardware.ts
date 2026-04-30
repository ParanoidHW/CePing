import { useState, useEffect } from 'react'
import { listHardware, listTopologies, getResources } from '@/api'
import type { HardwareInfo, TopologyInfo } from '@/types'

export function useHardware() {
  const [devices, setDevices] = useState<Record<string, HardwareInfo[]>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    listHardware()
      .then(setDevices)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return { devices, loading, error }
}

export function useTopologies() {
  const [topologies, setTopologies] = useState<TopologyInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    listTopologies()
      .then(setTopologies)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return { topologies, loading, error }
}

export function useResources() {
  const [devices, setDevices] = useState<Record<string, HardwareInfo[]>>({})
  const [topologies, setTopologies] = useState<TopologyInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    getResources()
      .then((data) => {
        setDevices(data.devices)
        setTopologies(data.topologies)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return { devices, topologies, loading, error }
}