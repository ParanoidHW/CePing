/**
 * Debug info export utility.
 * 
 * Generates, copies, and downloads debug information in JSON format.
 */

import { errorCapture } from './errorCapture'
import { requestCapture } from './requestCapture'
import type { HardwareSchema, StrategySchema } from '@/types'

export interface DebugInfo {
  debug_info: {
    timestamp: string
    frontend: {
      state: {
        workload: string | null
        model: string | null
        hardware: Record<string, unknown>
        strategy: Record<string, unknown>
        params: Record<string, unknown>
      }
      console_errors: unknown[]
      network: {
        requests: unknown[]
      }
    }
    environment: {
      frontend_commit: string
      browser: string
      url: string
    }
  }
}

export function generateDebugInfo(
  workload: string | null,
  model: string | null,
  hardware: HardwareSchema,
  strategy: StrategySchema,
  params: Record<string, unknown>
): DebugInfo {
  const commit = (import.meta as any).env?.VITE_GIT_COMMIT_HASH || 'unknown'
  
  return {
    debug_info: {
      timestamp: new Date().toISOString(),
      frontend: {
        state: {
          workload,
          model,
          hardware: hardware as unknown as Record<string, unknown>,
          strategy: strategy as unknown as Record<string, unknown>,
          params,
        },
        console_errors: errorCapture.getErrors(),
        network: {
          requests: requestCapture.getRequests(),
        },
      },
      environment: {
        frontend_commit: commit,
        browser: navigator.userAgent,
        url: window.location.href,
      },
    },
  }
}

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    console.error('Failed to copy to clipboard')
    return false
  }
}

export function downloadAsFile(data: unknown, filename: string): void {
  const json = JSON.stringify(data, null, 2)
  const blob = new Blob([json], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}