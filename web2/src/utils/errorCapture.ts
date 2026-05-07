/**
 * Error capture utility for debugging.
 * 
 * Captures global errors, console errors, and unhandled promise rejections.
 * Stores up to 50 recent errors.
 */

export interface CapturedError {
  timestamp: string
  type: 'error' | 'unhandledrejection' | 'console_error'
  message: string
  stack?: string
  filename?: string
  lineno?: number
  colno?: number
}

class ErrorCapture {
  private errors: CapturedError[] = []
  private maxErrors = 50
  private originalConsoleError: typeof console.error

  constructor() {
    this.originalConsoleError = console.error.bind(console)
    this.setup()
  }

  private setup(): void {
    window.onerror = (message, filename, lineno, colno, error) => {
      this.addError({
        timestamp: new Date().toISOString(),
        type: 'error',
        message: String(message),
        stack: error?.stack,
        filename: filename || undefined,
        lineno: lineno || undefined,
        colno: colno || undefined,
      })
    }

    window.addEventListener('unhandledrejection', (event) => {
      this.addError({
        timestamp: new Date().toISOString(),
        type: 'unhandledrejection',
        message: String(event.reason),
        stack: event.reason?.stack,
      })
    })

    console.error = (...args: unknown[]) => {
      this.originalConsoleError(...args)
      this.addError({
        timestamp: new Date().toISOString(),
        type: 'console_error',
        message: args.map(arg => String(arg)).join(' '),
      })
    }
  }

  private addError(error: CapturedError): void {
    this.errors.push(error)
    if (this.errors.length > this.maxErrors) {
      this.errors.shift()
    }
  }

  getErrors(): CapturedError[] {
    return [...this.errors]
  }

  clearErrors(): void {
    this.errors = []
  }
}

export const errorCapture = new ErrorCapture()