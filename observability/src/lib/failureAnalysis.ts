export type FailureSeverity = 'LOW' | 'MEDIUM' | 'HIGH'

export interface FailureType {
  code: string
  severity: FailureSeverity
  description: string
  step_ids: string[]
}

export interface ObservationFailureSummary {
  max_severity: FailureSeverity
  codes: string[]
}

export interface FailureSummary {
  trace_id: string
  has_failure: boolean
  failure_types: FailureType[]
  behavioural_signals: Record<string, unknown>
  duration_seconds: number | null
  user_query: string | null
  per_observation_failures: Record<string, ObservationFailureSummary>
}

function getFailureApiBaseUrl(): string {
  const fromEnv = import.meta.env.VITE_FAILURE_ANALYSIS_URL as string | undefined
  // Default to a local dev server if not configured.
  return fromEnv && fromEnv.trim() ? fromEnv.trim().replace(/\/$/, '') : 'http://localhost:8000'
}

export async function fetchFailureSummary(traceId: string): Promise<FailureSummary | null> {
  const baseUrl = getFailureApiBaseUrl()
  const url = `${baseUrl}/traces/${encodeURIComponent(traceId)}/failure-summary`

  try {
    const response = await fetch(url, {
      headers: {
        Accept: 'application/json',
      },
    })

    if (!response.ok) {
      console.warn('[FailureAnalysis] Request failed', response.status, response.statusText)
      return null
    }

    const payload = (await response.json()) as FailureSummary
    return payload
  } catch (error) {
    console.error('[FailureAnalysis] Failed to fetch failure summary', error)
    return null
  }
}
