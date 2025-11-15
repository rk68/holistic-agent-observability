export interface LangfuseTrace {
  id: string
  name: string
  environment: string | null
  timestamp: string
  latency: number | null
  url: string
  tags: string[]
}

export interface LangfuseObservation {
  id: string
  type: string
  name: string | null
  parentObservationId: string | null
  startTime: string
  endTime: string | null
  metadata: Record<string, unknown>
  input: unknown
  output: unknown
}

export interface LangfuseTraceDetail extends LangfuseTrace {
  observations: LangfuseObservation[]
}

interface LangfuseTraceResponse {
  id: string
  name: string
  environment: string | null
  timestamp: string
  latency?: number | null
  htmlPath: string
  tags?: string[]
}

interface PublicTraceListResponse {
  data: LangfuseTraceResponse[]
  meta?: {
    next?: string | null
  }
}

interface LangfuseObservationResponse {
  id: string
  type: string
  name: string | null
  parentObservationId: string | null
  startTime: string
  endTime: string | null
  metadata?: Record<string, unknown>
  input?: unknown
  output?: unknown
}

interface TraceDetailResponse {
  id: string
  name: string
  environment: string | null
  timestamp: string
  latency?: number | null
  htmlPath: string
  tags?: string[]
  observations: LangfuseObservationResponse[]
}

const REQUIRED_ENV_VARS = [
  'VITE_LANGFUSE_HOST',
  'VITE_LANGFUSE_PUBLIC_KEY',
  'VITE_LANGFUSE_SECRET_KEY',
] as const

type RequiredEnvVar = (typeof REQUIRED_ENV_VARS)[number]

function ensureEnv(name: RequiredEnvVar): string {
  const value = import.meta.env[name]
  if (!value) {
    throw new Error(`Missing ${name}. Update your environment or .env file.`)
  }
  return value
}

function encodeBasicAuth(username: string, password: string): string {
  const credentials = `${username}:${password}`

  if (typeof globalThis.btoa === 'function') {
    return globalThis.btoa(credentials)
  }

  const nodeBuffer = (globalThis as { Buffer?: { from(data: string, encoding: string): { toString(encoding: string): string } } }).Buffer
  if (typeof nodeBuffer !== 'undefined') {
    return nodeBuffer.from(credentials, 'utf-8').toString('base64')
  }

  throw new Error('Unable to encode credentials: no base64 encoder available in this environment')
}

function normaliseHost(host: string): string {
  return host.endsWith('/') ? host.slice(0, -1) : host
}

export async function fetchRecentTraces(limit = 50, maxPages = 20): Promise<LangfuseTrace[]> {
  const host = normaliseHost(ensureEnv('VITE_LANGFUSE_HOST'))
  const publicKey = ensureEnv('VITE_LANGFUSE_PUBLIC_KEY')
  const secretKey = ensureEnv('VITE_LANGFUSE_SECRET_KEY')

  const aggregated: LangfuseTrace[] = []
  const seen = new Set<string>()
  let cursor: string | null = null

  for (let page = 0; page < maxPages; page += 1) {
    const requestUrl = new URL(`${host}/api/public/traces`)
    requestUrl.searchParams.set('limit', String(limit))
    if (cursor) {
      requestUrl.searchParams.set('cursor', cursor)
    }

    const response = await fetch(requestUrl, {
      headers: {
        Authorization: `Basic ${encodeBasicAuth(publicKey, secretKey)}`,
        Accept: 'application/json',
      },
    })

    if (!response.ok) {
      const body = await response.text()
      throw new Error(`Langfuse request failed: ${response.status} ${response.statusText}\n${body}`)
    }

    const payload = (await response.json()) as PublicTraceListResponse
    const pageTraces = (payload.data ?? []).map((trace) => ({
      id: trace.id,
      name: trace.name,
      environment: trace.environment ?? 'default',
      timestamp: trace.timestamp,
      latency: trace.latency ?? null,
      url: `${host}${trace.htmlPath}`,
      tags: trace.tags ?? [],
    }))

    pageTraces.forEach((trace) => {
      if (!seen.has(trace.id)) {
        seen.add(trace.id)
        aggregated.push(trace)
      }
    })

    const nextCursor = payload.meta?.next ?? null
    if (!nextCursor || nextCursor === cursor) {
      break
    }
    cursor = nextCursor
  }

  return aggregated
}

export async function fetchTraceDetail(traceId: string): Promise<LangfuseTraceDetail> {
  const host = normaliseHost(ensureEnv('VITE_LANGFUSE_HOST'))
  const publicKey = ensureEnv('VITE_LANGFUSE_PUBLIC_KEY')
  const secretKey = ensureEnv('VITE_LANGFUSE_SECRET_KEY')

  const response = await fetch(`${host}/api/public/traces/${traceId}`, {
    headers: {
      Authorization: `Basic ${encodeBasicAuth(publicKey, secretKey)}`,
      Accept: 'application/json',
    },
  })

  if (!response.ok) {
    const body = await response.text()
    throw new Error(`Failed to load trace detail: ${response.status} ${response.statusText}\n${body}`)
  }

  const payload = (await response.json()) as TraceDetailResponse
  return {
    id: payload.id,
    name: payload.name,
    environment: payload.environment ?? 'default',
    timestamp: payload.timestamp,
    latency: payload.latency ?? null,
    url: `${host}${payload.htmlPath}`,
    tags: payload.tags ?? [],
    observations: (payload.observations ?? []).map((observation) => ({
      id: observation.id,
      type: observation.type,
      name: observation.name,
      parentObservationId: observation.parentObservationId,
      startTime: observation.startTime,
      endTime: observation.endTime,
      metadata: observation.metadata ?? {},
      input: observation.input ?? null,
      output: observation.output ?? null,
    })),
  }
}
