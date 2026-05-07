/**
 * Request capture utility for debugging.
 * 
 * Intercepts axios requests/responses and stores up to 20 recent requests.
 */

import axios from 'axios'

export interface CapturedRequest {
  timestamp: string
  method: string
  url: string
  status?: number
  request_body?: unknown
  response_body?: unknown
  error?: string
}

class RequestCapture {
  private requests: CapturedRequest[] = []
  private maxRequests = 20

  constructor() {
    this.setup()
  }

  private setup(): void {
    axios.interceptors.request.use(
      (config) => {
        const request: CapturedRequest = {
          timestamp: new Date().toISOString(),
          method: config.method?.toUpperCase() || 'GET',
          url: config.url || '',
          request_body: config.data,
        }
        this.requests.push(request)
        if (this.requests.length > this.maxRequests) {
          this.requests.shift()
        }
        return config
      },
      (error) => {
        return Promise.reject(error)
      }
    )

    axios.interceptors.response.use(
      (response) => {
        const lastRequest = this.requests[this.requests.length - 1]
        if (lastRequest) {
          lastRequest.status = response.status
          lastRequest.response_body = response.data
        }
        return response
      },
      (error) => {
        const lastRequest = this.requests[this.requests.length - 1]
        if (lastRequest) {
          lastRequest.status = error.response?.status
          lastRequest.response_body = error.response?.data
          lastRequest.error = error.message
        }
        return Promise.reject(error)
      }
    )
  }

  getRequests(): CapturedRequest[] {
    return [...this.requests]
  }

  clearRequests(): void {
    this.requests = []
  }
}

export const requestCapture = new RequestCapture()