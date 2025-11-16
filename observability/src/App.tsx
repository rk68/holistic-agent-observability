import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  Position,
  type Edge as ReactFlowEdge,
  type Node as ReactFlowNode,
  type ReactFlowInstance,
} from 'reactflow'
import 'reactflow/dist/style.css'

import TraceTable from './components/TraceTable'
import {
  fetchRecentTraces,
  fetchTraceDetail,
  type LangfuseTrace,
  type LangfuseTraceDetail,
  type LangfuseObservation,
} from './lib/langfuse'
import './App.css'

type ConnectionStatus = 'idle' | 'connecting' | 'connected' | 'error'
type DetailStatus = 'idle' | 'loading' | 'loaded' | 'error'

type NarrativeStatus = 'idle' | 'processing' | 'ready'

type ExecutionNarrative = {
  userQuestion: string | null
  planningTools: string[]
  toolsExecuted: { name: string; summary: string }[]
  finalAnswer: string | null
}

function formatMgSig(kg: number | null | undefined, sig = 3): string {
  const mg = (Number(kg) || 0) * 1_000_000
  const abs = Math.abs(mg)
  if (!isFinite(mg) || abs === 0) {
    return '0'
  }
  try {
    const fmt = new Intl.NumberFormat(undefined, {
      maximumSignificantDigits: sig,
      minimumSignificantDigits: 1,
    })
    return fmt.format(mg)
  } catch {
    // Fallback if Intl fails in some envs
    return Number(mg).toPrecision(sig)
  }
}

type ReasoningSummary = {
  goal: string | null
  plan: string[]
  observations: string[]
  result: string | null
}

type ObservationInsight = {
  observationId: string
  stage: string | null
  summary: string | null
  bullets: string[]
}

type ObservationMetric = {
  observationId: string
  metric: string
  subject: string
  entailment: number
  contradiction: number
  neutral: number
  label: string
}

type CarbonPhase = {
  kg: number
  duration_s: number
}

type CarbonSummary = {
  total_kg: number
  duration_seconds: number
  breakdown: {
    reasoning: CarbonPhase
    rest: CarbonPhase
  }
  source?: string
}

type StoredTraceSummary = {
  narrative: ExecutionNarrative | null
  reasoningSummary: ReasoningSummary | null
  observationInsights: ObservationInsight[]
  observationMetrics: ObservationMetric[]
  rootCauseObservationId: string | null
  carbonSummary?: CarbonSummary | null
  timestamp: number
}

const normaliseStoredId = (value: string | null): string | null => {
  if (!value || value === 'null') {
    return null
  }
  const trimmed = value.trim()
  return trimmed ? trimmed : null
}

const loadStoredApiKeys = (): ApiKeyMap => {
  const raw = safeReadStorage(STORAGE_KEYS.apiKey)
  if (!raw) {
    return {}
  }
  try {
    const parsed = JSON.parse(raw) as ApiKeyMap
    if (parsed && typeof parsed === 'object') {
      return parsed
    }
  } catch (error) {
    console.warn('[Observability] Failed to parse stored API keys', error)
  }
  return {}
}

type GraphNodeData = {
  label: ReactNode
  kind: string
  step?: number
}

type GraphNode = ReactFlowNode<GraphNodeData>
type GraphEdge = ReactFlowEdge

const ROOT_OFFSET_Y = 80
const VERTICAL_SPACING = 120
const HORIZONTAL_SPACING = 260

const TRACE_SUMMARY_URL =
  (import.meta as any).env?.VITE_TRACE_SUMMARY_URL ?? 'http://localhost:8000/trace-summary'

const STORAGE_KEYS = {
  selectedTraceId: 'observability:selectedTraceId',
  selectedObservationId: 'observability:selectedObservationId',
  pendingTraceId: 'observability:pendingTraceId',
  provider: 'observability:provider',
  apiKey: 'observability:apiKey',
  nliModel: 'observability:nliModel',
  llmModel: 'observability:llmModel',
}

const NLI_MODEL_OPTIONS: { id: string; label: string }[] = [
  { id: 'cross-encoder/nli-deberta-v3-small', label: 'DeBERTa v3 Small (NLI)' },
  { id: 'cross-encoder/nli-deberta-v3-base', label: 'DeBERTa v3 Base (NLI)' },
  { id: 'cross-encoder/nli-roberta-base', label: 'RoBERTa Base (NLI)' },
]

const LLM_MODEL_OPTIONS: { id: string; label: string }[] = [
  { id: 'qwen3:4b', label: 'Qwen3 4B (Ollama)' },
  { id: 'gpt-oss:20b', label: 'GPT‑OSS 20B (Ollama)' },
  { id: 'llama3.1:8b', label: 'Llama 3.1 8B (Ollama)' },
]

type ObservabilityProvider = 'langfuse' | 'langsmith'

const PROVIDER_OPTIONS: { id: ObservabilityProvider; label: string; helper: string }[] = [
  { id: 'langfuse', label: 'Langfuse', helper: 'Analyze LangGraph traces stored in Langfuse.' },
  { id: 'langsmith', label: 'LangSmith', helper: 'Connect to LangChain’s LangSmith observability suite.' },
]

type ApiKeyMap = Partial<Record<ObservabilityProvider, string>>

const SUMMARY_KEY_PREFIX = 'observability:traceSummary:'
const STORED_TRACE_KEY_PREFIX = 'observability:processedTrace:'

const safeReadStorage = (key: string): string | null => {
  if (typeof window === 'undefined') {
    return null
  }
  try {
    return window.localStorage.getItem(key)
  } catch (error) {
    console.warn('[Observability] Unable to read storage key', key, error)
    return null
  }
}

const safeWriteStorage = (key: string, value: string | null): void => {
  if (typeof window === 'undefined') {
    return
  }
  try {
    if (value === null) {
      window.localStorage.removeItem(key)
    } else {
      window.localStorage.setItem(key, value)
    }
  } catch (error) {
    console.warn('[Observability] Unable to write storage key', key, error)
  }
}

const loadPersistedNarrative = (traceId: string | null): ExecutionNarrative | null => {
  if (!traceId) {
    return null
  }
  const raw = safeReadStorage(SUMMARY_KEY_PREFIX + traceId)
  if (!raw) {
    return null
  }
  try {
    const parsed = JSON.parse(raw) as ExecutionNarrative
    const planningTools = Array.isArray(parsed.planningTools)
      ? parsed.planningTools.map((item) => String(item)).filter((item) => Boolean(item))
      : []
    const toolsExecuted = Array.isArray(parsed.toolsExecuted)
      ? parsed.toolsExecuted
          .map((entry) => {
            if (!entry || typeof entry !== 'object') {
              return null
            }
            const candidate = entry as { name?: unknown; summary?: unknown }
            const name = typeof candidate.name === 'string' ? candidate.name : null
            if (!name) {
              return null
            }
            const summary =
              typeof candidate.summary === 'string' && candidate.summary
                ? candidate.summary
                : name
            return { name, summary }
          })
          .filter((item): item is { name: string; summary: string } => item !== null)
      : []
    return {
      userQuestion: typeof parsed.userQuestion === 'string' ? parsed.userQuestion : null,
      planningTools,
      toolsExecuted,
      finalAnswer: typeof parsed.finalAnswer === 'string' ? parsed.finalAnswer : null,
    }
  } catch (error) {
    console.warn('[Observability] Failed to parse stored narrative for trace', traceId, error)
    safeWriteStorage(SUMMARY_KEY_PREFIX + traceId, null)
    return null
  }
}

const persistNarrativeForTrace = (traceId: string, narrative: ExecutionNarrative | null): void => {
  if (!traceId) {
    return
  }
  if (!narrative) {
    safeWriteStorage(SUMMARY_KEY_PREFIX + traceId, null)
    return
  }
  try {
    safeWriteStorage(SUMMARY_KEY_PREFIX + traceId, JSON.stringify(narrative))
  } catch (error) {
    console.warn('[Observability] Failed to persist narrative for trace', traceId, error)
  }
}

const loadStoredTraceSummary = (traceId: string | null): StoredTraceSummary | null => {
  if (!traceId) {
    return null
  }
  const raw = safeReadStorage(STORED_TRACE_KEY_PREFIX + traceId)
  if (!raw) {
    return null
  }
  try {
    const parsed = JSON.parse(raw) as StoredTraceSummary
    if (parsed && typeof parsed === 'object') {
      return {
        narrative: parsed.narrative ?? null,
        reasoningSummary: parsed.reasoningSummary ?? null,
        observationInsights: Array.isArray(parsed.observationInsights) ? parsed.observationInsights : [],
        observationMetrics: Array.isArray(parsed.observationMetrics) ? parsed.observationMetrics : [],
        rootCauseObservationId: typeof parsed.rootCauseObservationId === 'string' ? parsed.rootCauseObservationId : null,
        carbonSummary: parsed.carbonSummary ?? null,
        timestamp: typeof parsed.timestamp === 'number' ? parsed.timestamp : Date.now(),
      }
    }
  } catch (error) {
    console.warn('[Observability] Failed to parse stored processed trace', traceId, error)
    safeWriteStorage(STORED_TRACE_KEY_PREFIX + traceId, null)
  }
  return null
}

const persistStoredTraceSummary = (traceId: string, summary: StoredTraceSummary | null): void => {
  if (!traceId) {
    return
  }
  if (!summary) {
    safeWriteStorage(STORED_TRACE_KEY_PREFIX + traceId, null)
    return
  }
  try {
    safeWriteStorage(STORED_TRACE_KEY_PREFIX + traceId, JSON.stringify(summary))
  } catch (error) {
    console.warn('[Observability] Failed to persist processed trace', traceId, error)
  }
}

type PendingTraceRequest = {
  traceId: string
  timestamp: number
}

const readPendingTrace = (): PendingTraceRequest | null => {
  const raw = safeReadStorage(STORAGE_KEYS.pendingTraceId)
  if (!raw) {
    return null
  }
  try {
    const parsed = JSON.parse(raw) as PendingTraceRequest
    if (parsed && typeof parsed.traceId === 'string') {
      return parsed
    }
  } catch (error) {
    console.warn('[Observability] Failed to parse pending trace record', error)
  }
  safeWriteStorage(STORAGE_KEYS.pendingTraceId, null)
  return null
}

const persistPendingTrace = (traceId: string | null): PendingTraceRequest | null => {
  if (!traceId) {
    safeWriteStorage(STORAGE_KEYS.pendingTraceId, null)
    return null
  }
  const record: PendingTraceRequest = { traceId, timestamp: Date.now() }
  safeWriteStorage(STORAGE_KEYS.pendingTraceId, JSON.stringify(record))
  return record
}

function formatDateTime(timestamp: string): string {
  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: 'medium',
      timeStyle: 'medium',
    }).format(new Date(timestamp))
  } catch {
    return timestamp
  }
}

function truncate(value: string | null | undefined, max = 180): string | null {
  if (!value) {
    return null
  }
  const trimmed = value.trim()
  if (!trimmed) {
    return null
  }
  return trimmed.length > max ? `${trimmed.slice(0, max - 1).trim()}…` : trimmed
}

function toNodeLabel(observation: LangfuseObservation, step: number): string {
  const metadata = observation.metadata ?? {}
  const parts: string[] = [`#${step}`, observation.type.toLowerCase()]

  if (observation.name) {
    parts.push(`• ${observation.name}`)
  } else if (typeof metadata.langgraph_node === 'string') {
    parts.push(`• ${metadata.langgraph_node}`)
  }

  return parts.join(' ')
}

const NODE_THEMES: Record<
  string,
  { border: string; background: string; accent: string; text: string; badge: string; badgeText: string }
> = {
  GENERATION: {
    border: '#4f46e5',
    background: '#eef2ff',
    accent: '#3730a3',
    text: '#1e1b4b',
    badge: 'rgba(79,70,229,0.12)',
    badgeText: '#312e81',
  },
  TOOL: {
    border: '#0f766e',
    background: '#ecfdf5',
    accent: '#0f766e',
    text: '#064e3b',
    badge: 'rgba(15,118,110,0.12)',
    badgeText: '#065f46',
  },
  AGENT: {
    border: '#9333ea',
    background: '#f3e8ff',
    accent: '#7e22ce',
    text: '#581c87',
    badge: 'rgba(147,51,234,0.12)',
    badgeText: '#6b21a8',
  },
  SPAN: {
    border: '#0891b2',
    background: '#ecfeff',
    accent: '#0e7490',
    text: '#0f172a',
    badge: 'rgba(8,145,178,0.12)',
    badgeText: '#0e7490',
  },
  CHAIN: {
    border: '#a16207',
    background: '#fffbeb',
    accent: '#854d0e',
    text: '#713f12',
    badge: 'rgba(161,98,7,0.12)',
    badgeText: '#854d0e',
  },
  default: {
    border: '#475569',
    background: '#f8fafc',
    accent: '#334155',
    text: '#0f172a',
    badge: 'rgba(71,85,105,0.12)',
    badgeText: '#0f172a',
  },
}

const NODE_ICONS: Record<string, string> = {
  GENERATION: '🤖',
  TOOL: '🛠️',
  AGENT: '🧠',
  SPAN: '🧩',
  CHAIN: '⛓️',
  default: '📦',
}

type NodeVisualOptions = {
  kind: string
  isSelected: boolean
  hasIssue: boolean
  borderline: boolean
}

const ISSUE_THEME = {
  border: '#dc2626',
  background: '#fef2f2',
  accent: '#b91c1c',
  text: '#7f1d1d',
  badge: 'rgba(220,38,38,0.12)',
  badgeText: '#b91c1c',
}

const BORDERLINE_THEME = {
  border: '#ea580c',
  background: '#fff7ed',
  accent: '#c2410c',
  text: '#7c2d12',
  badge: 'rgba(234,88,12,0.12)',
  badgeText: '#9a3412',
}

const getNodeVisuals = ({ kind, isSelected, hasIssue, borderline }: NodeVisualOptions) => {
  let theme = NODE_THEMES[kind as keyof typeof NODE_THEMES] ?? NODE_THEMES.default
  if (hasIssue) {
    theme = ISSUE_THEME
  } else if (borderline) {
    theme = BORDERLINE_THEME
  }

  const border = isSelected ? '#1d4ed8' : theme.border
  const background = isSelected ? '#eef2ff' : theme.background
  const boxShadow = isSelected
    ? '0 15px 35px rgba(37,99,235,0.25)'
    : '0 8px 20px rgba(15,23,42,0.08)'

  return {
    style: {
      borderRadius: 18,
      padding: '18px 22px',
      border: `3px solid ${border}`,
      background,
      boxShadow,
      minWidth: 260,
      maxWidth: 420,
      color: theme.text,
      lineHeight: 1.4,
    },
    badgeStyles: {
      backgroundColor: theme.badge,
      color: isSelected ? '#1e1b4b' : theme.badgeText,
    },
    accent: theme.accent,
    text: theme.text,
  }
}

function extractFirstUserMessage(observation: LangfuseObservation): string | null {
  const input = observation.input

  const inspectMessages = (messages: unknown): string | null => {
    if (!Array.isArray(messages)) {
      return null
    }
    for (const item of messages) {
      if (!item || typeof item !== 'object') {
        continue
      }
      if ('role' in item && (item as { role?: string }).role === 'user' && 'content' in item) {
        const content = (item as { content?: unknown }).content
        if (typeof content === 'string') {
          return content
        }
      }
    }
    return null
  }

  if (input && typeof input === 'object' && 'messages' in input) {
    const candidate = inspectMessages((input as { messages?: unknown[] }).messages)
    if (candidate) {
      return candidate
    }
  }

  return inspectMessages(input)
}

function extractModelOutput(observation: LangfuseObservation): string | null {
  const output = observation.output

  const inspectMessages = (messages: unknown): string | null => {
    if (!Array.isArray(messages)) {
      return null
    }
    let fallback: string | null = null

    for (const item of messages) {
      if (!item || typeof item !== 'object') {
        continue
      }

      const record = item as { content?: unknown; additional_kwargs?: unknown }

      // Prefer explicit reasoning content when emitted by ChatOllama with
      // reasoning mode enabled. This is stored under
      // additional_kwargs.reasoning_content and does not include the final
      // answer text.
      const additional = record.additional_kwargs
      if (additional && typeof additional === 'object') {
        const reasoning = (additional as { reasoning_content?: unknown }).reasoning_content
        if (typeof reasoning === 'string' && reasoning.trim()) {
          return reasoning
        }
      }

      if ('content' in record && typeof record.content === 'string') {
        if (!fallback) {
          fallback = record.content
        }
      }
    }
    return fallback
  }

  if (output && typeof output === 'object' && 'messages' in output) {
    const candidate = inspectMessages((output as { messages?: unknown[] }).messages)
    if (candidate) {
      return candidate
    }
  }

  if (Array.isArray(output)) {
    const candidate = inspectMessages(output)
    if (candidate) {
      return candidate
    }
  }

  if (output && typeof output === 'object') {
    const candidate = inspectMessages([output])
    if (candidate) {
      return candidate
    }
  }

  if (typeof output === 'string') {
    return output
  }

  return null
}

function extractAssistantContent(observation: LangfuseObservation): string | null {
  const output = observation.output

  const inspectMessages = (messages: unknown): string | null => {
    if (!Array.isArray(messages)) {
      return null
    }
    for (const item of messages) {
      if (!item || typeof item !== 'object') {
        continue
      }
      const record = item as { content?: unknown }
      if (typeof record.content === 'string' && record.content.trim()) {
        return record.content
      }
    }
    return null
  }

  if (output && typeof output === 'object' && 'messages' in output) {
    const candidate = inspectMessages((output as { messages?: unknown[] }).messages)
    if (candidate) {
      return candidate
    }
  }

  if (Array.isArray(output)) {
    const candidate = inspectMessages(output)
    if (candidate) {
      return candidate
    }
  }

  if (output && typeof output === 'object') {
    const candidate = inspectMessages([output])
    if (candidate) {
      return candidate
    }
  }

  if (typeof output === 'string' && output.trim()) {
    return output
  }

  return null
}

function extractPlannedToolCallsFromReasoning(
  reasoning: string | null,
  availableToolNames: string[],
): string[] {
  if (!reasoning || !availableToolNames.length) {
    return []
  }

  const known = new Set(availableToolNames)
  const planned: string[] = []
  const lines = reasoning.split(/\r?\n/)
  const toolPattern = /([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)/g

  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (!line) {
      continue
    }

    let match: RegExpExecArray | null
    while ((match = toolPattern.exec(line)) !== null) {
      const name = match[1]
      if (known.has(name) && !planned.includes(name)) {
        planned.push(name)
      }
    }
  }

  return planned
}

function formatPayload(payload: unknown, fallback = '—'): string {
  if (payload == null) {
    return fallback
  }
  if (typeof payload === 'string') {
    const trimmed = payload.trim()
    if (!trimmed) {
      return fallback
    }
    try {
      return JSON.stringify(JSON.parse(trimmed), null, 2)
    } catch {
      return trimmed
    }
  }

  try {
    return JSON.stringify(payload, null, 2)
  } catch (error) {
    console.warn('Unable to serialise payload', error)
    return String(payload)
  }
}

function buildHighlight(observation: LangfuseObservation): { label: string; value: string | null } | null {
  switch (observation.type) {
    case 'SPAN':
      return { label: 'User prompt', value: extractFirstUserMessage(observation) }
    case 'AGENT':
      return { label: 'Agent reasoning', value: extractModelOutput(observation) }
    case 'GENERATION':
      return { label: 'LLM reasoning', value: extractModelOutput(observation) }
    case 'TOOL':
      return {
        label: observation.name ? `Tool • ${observation.name}` : 'Tool output',
        value: typeof observation.output === 'string'
          ? observation.output
          : formatPayload(observation.output),
      }
    default:
      return null
  }
}

function App() {
  const initialTraceId = normaliseStoredId(safeReadStorage(STORAGE_KEYS.selectedTraceId))
  const initialObservationId = normaliseStoredId(safeReadStorage(STORAGE_KEYS.selectedObservationId))
  const initialProvider = ((): ObservabilityProvider => {
    const stored = safeReadStorage(STORAGE_KEYS.provider)
    if (stored === 'langsmith') {
      return 'langsmith'
    }
    return 'langfuse'
  })()
  const initialApiKeys = loadStoredApiKeys()
  const [status, setStatus] = useState<ConnectionStatus>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [traces, setTraces] = useState<LangfuseTrace[]>([])
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(initialTraceId)
  const [traceDetail, setTraceDetail] = useState<LangfuseTraceDetail | null>(null)
  const [detailStatus, setDetailStatus] = useState<DetailStatus>('idle')
  const [detailError, setDetailError] = useState<string | null>(null)
  const [selectedObservationId, setSelectedObservationId] = useState<string | null>(initialObservationId)
  const [narrativeStatus, setNarrativeStatus] = useState<NarrativeStatus>('idle')
  const [remoteNarrative, setRemoteNarrative] = useState<ExecutionNarrative | null>(
    () => loadPersistedNarrative(initialTraceId),
  )
  const [pendingTrace, setPendingTraceState] = useState<PendingTraceRequest | null>(() => readPendingTrace())
  const [provider, setProvider] = useState<ObservabilityProvider>(initialProvider)
  const [apiKeys, setApiKeys] = useState<ApiKeyMap>(initialApiKeys)
  const [pendingApiKey, setPendingApiKey] = useState<string>(initialApiKeys[initialProvider] ?? '')
  const [reasoningSummary, setReasoningSummary] = useState<ReasoningSummary | null>(null)
  const [observationInsights, setObservationInsights] = useState<ObservationInsight[]>([])
  const [observationMetrics, setObservationMetrics] = useState<ObservationMetric[]>([])
  const [rootCauseObservationId, setRootCauseObservationId] = useState<string | null>(null)
  const [storedSummary, setStoredSummary] = useState<StoredTraceSummary | null>(null)
  const [carbonSummary, setCarbonSummary] = useState<CarbonSummary | null>(null)
  const initialNliModel = normaliseStoredId(safeReadStorage(STORAGE_KEYS.nliModel)) ||
    NLI_MODEL_OPTIONS[0]?.id || 'cross-encoder/nli-deberta-v3-small'
  const initialLlmModel = normaliseStoredId(safeReadStorage(STORAGE_KEYS.llmModel)) ||
    LLM_MODEL_OPTIONS[0]?.id || 'qwen3:4b'
  const [nliModel, setNliModel] = useState<string>(initialNliModel)
  const [llmModel, setLlmModel] = useState<string>(initialLlmModel)

  const metricOverview = useMemo(() => {
    const totals = { grounded: 0, neutral: 0, contradicted: 0 }
    for (const m of observationMetrics) {
      if (m.label === 'ENTAILED') totals.grounded += 1
      else if (m.label === 'NEUTRAL') totals.neutral += 1
      else if (m.label === 'CONTRADICTED') totals.contradicted += 1
    }
    return totals
  }, [observationMetrics])

  const metricDistribution = useMemo(() => {
    const total = observationMetrics.length || 1
    return [
      {
        key: 'grounded',
        label: 'Grounded',
        count: metricOverview.grounded,
        percent: Math.round((metricOverview.grounded / total) * 100),
      },
      {
        key: 'neutral',
        label: 'Needs review',
        count: metricOverview.neutral,
        percent: Math.round((metricOverview.neutral / total) * 100),
      },
      {
        key: 'contradiction',
        label: 'Contradiction',
        count: metricOverview.contradicted,
        percent: Math.round((metricOverview.contradicted / total) * 100),
      },
    ]
  }, [metricOverview, observationMetrics.length])

  const rawMetricSamples = useMemo(() => {
    const items = observationMetrics.slice(0, 6).map((m) => ({
      key: `${m.observationId}:${m.subject}:${m.metric}`,
      title: `${m.subject} • ${m.metric.replaceAll('_', ' ')}`,
      entailment: Math.max(0, Math.min(1, Number(m.entailment) || 0)),
      neutral: Math.max(0, Math.min(1, Number(m.neutral) || 0)),
      contradiction: Math.max(0, Math.min(1, Number(m.contradiction) || 0)),
    }))
    return items
  }, [observationMetrics])

  const storedSummaryTimestamp = useMemo(() => {
    if (!storedSummary) {
      return null
    }
    try {
      return formatDateTime(new Date(storedSummary.timestamp).toISOString())
    } catch {
      return null
    }
  }, [storedSummary])

  const applyStoredSummary = useCallback(
    (summary: StoredTraceSummary) => {
      setRemoteNarrative(summary.narrative)
      setReasoningSummary(summary.reasoningSummary)
      setObservationInsights(summary.observationInsights)
      setObservationMetrics(summary.observationMetrics)
      setRootCauseObservationId(summary.rootCauseObservationId)
      setCarbonSummary(summary.carbonSummary ?? null)
      setNarrativeStatus('ready')
    },
    [],
  )

  useEffect(() => {
    if (!selectedTraceId) {
      setStoredSummary(null)
      return
    }
    const stored = loadStoredTraceSummary(selectedTraceId)
    setStoredSummary(stored)
    if (stored) {
      applyStoredSummary(stored)
    } else {
      // Clear previous trace's metrics so panels don't show stale data
      setReasoningSummary(null)
      setObservationInsights([])
      setObservationMetrics([])
      setRootCauseObservationId(null)
      setCarbonSummary(null)
      setNarrativeStatus('idle')
    }
  }, [selectedTraceId, applyStoredSummary])

  const handleLoadStoredSummary = useCallback(() => {
    if (!selectedTraceId) {
      return
    }
    const stored = loadStoredTraceSummary(selectedTraceId)
    if (stored) {
      setStoredSummary(stored)
      applyStoredSummary(stored)
    }
  }, [selectedTraceId, applyStoredSummary])

  const updatePendingTrace = useCallback((traceId: string | null) => {
    const record = persistPendingTrace(traceId)
    setPendingTraceState(record)
    return record
  }, [])

  useEffect(() => {
    safeWriteStorage(STORAGE_KEYS.provider, provider)
  }, [provider])

  useEffect(() => {
    safeWriteStorage(STORAGE_KEYS.nliModel, nliModel)
  }, [nliModel])

  useEffect(() => {
    safeWriteStorage(STORAGE_KEYS.llmModel, llmModel)
  }, [llmModel])

  useEffect(() => {
    if (!Object.keys(apiKeys).length) {
      safeWriteStorage(STORAGE_KEYS.apiKey, null)
      return
    }
    try {
      safeWriteStorage(STORAGE_KEYS.apiKey, JSON.stringify(apiKeys))
    } catch (error) {
      console.warn('[Observability] Failed to persist API keys', error)
    }
  }, [apiKeys])

  useEffect(() => {
    setPendingApiKey(apiKeys[provider] ?? '')
  }, [provider])

  const statusLabel = useMemo(() => {
    switch (status) {
      case 'connecting':
        return 'Connecting to Langfuse…'
      case 'connected':
        return traces.length
          ? `Loaded ${traces.length} trace${traces.length === 1 ? '' : 's'}`
          : 'Connected — no traces returned'
      case 'error':
        return 'Unable to load traces'
      default:
        return 'Not connected'
    }
  }, [status, traces])

  const handleConnect = useCallback(async () => {
    setStatus('connecting')
    setErrorMessage(null)

    console.info('[Observability] Connecting to Langfuse for recent traces…')

    try {
      const results = await fetchRecentTraces()
      console.info('[Observability] Received traces', { count: results.length })
      setTraces(results)
      setStatus('connected')

      if (results.length === 0) {
        setSelectedTraceId(null)
        setTraceDetail(null)
        setDetailStatus('idle')
        setSelectedObservationId(null)
        return
      }

      if (selectedTraceId && !results.some((trace) => trace.id === selectedTraceId)) {
        console.info('[Observability] Previously selected trace no longer in recent results')
        setSelectedTraceId(null)
        setTraceDetail(null)
        setDetailStatus('idle')
        setSelectedObservationId(null)
      }
    } catch (error) {
      setStatus('error')
      setErrorMessage(error instanceof Error ? error.message : 'Unknown error occurred')
      console.error('[Observability] Failed to load recent traces', error)
    }
  }, [selectedTraceId])

  const handleSelectTrace = useCallback(async (trace: LangfuseTrace) => {
    console.info('[Observability] Loading trace detail', { traceId: trace.id, name: trace.name })
    setSelectedTraceId(trace.id)
    setDetailStatus('loading')
    setDetailError(null)
    setSelectedObservationId(null)
    setNarrativeStatus('idle')
    setRemoteNarrative(null)

    try {
      const detail = await fetchTraceDetail(trace.id)
      console.info('[Observability] Loaded trace detail', {
        traceId: trace.id,
        observations: detail.observations.length,
      })
      setTraceDetail(detail)
      setDetailStatus('loaded')
      setSelectedObservationId(detail.observations.length > 0 ? detail.observations[0].id : null)
    } catch (error) {
      setTraceDetail(null)
      setDetailStatus('error')
      setDetailError(error instanceof Error ? error.message : 'Unknown error occurred')
      console.error('[Observability] Failed to load trace detail', error)
    }
  }, [])

  useEffect(() => {
    if (status === 'connected' && traces.length > 0 && !selectedTraceId) {
      void handleSelectTrace(traces[0])
    }
  }, [status, traces, selectedTraceId, handleSelectTrace])

  useEffect(() => {
    if (!selectedTraceId || traces.length === 0) {
      return
    }
    const next = traces.find((trace) => trace.id === selectedTraceId)
    if (!next) {
      return
    }
    if (traceDetail && traceDetail.id === next.id) {
      return
    }
    if (detailStatus === 'loading') {
      return
    }
    void handleSelectTrace(next)
  }, [selectedTraceId, traces, traceDetail, detailStatus, handleSelectTrace])

  useEffect(() => {
    safeWriteStorage(STORAGE_KEYS.selectedTraceId, selectedTraceId)
  }, [selectedTraceId])

  useEffect(() => {
    safeWriteStorage(STORAGE_KEYS.selectedObservationId, selectedObservationId)
  }, [selectedObservationId])

  useEffect(() => {
    const narrative = loadPersistedNarrative(selectedTraceId)
    setRemoteNarrative(narrative)
  }, [selectedTraceId])

  const handleProviderSelect = useCallback(
    (next: ObservabilityProvider) => {
      setProvider(next)
    },
    [],
  )

  const handleSaveApiKey = useCallback(() => {
    setApiKeys((current) => {
      const next = { ...current }
      const trimmed = pendingApiKey.trim()
      if (trimmed) {
        next[provider] = trimmed
      } else {
        delete next[provider]
      }
      return next
    })
  }, [pendingApiKey, provider])

  const activeApiKey = apiKeys[provider] ?? ''
  const apiKeyStatusLabel = activeApiKey ? 'API key saved locally' : 'Using .env credentials'

  useEffect(() => {
    if (detailStatus !== 'loaded' || !traceDetail) {
      return
    }

    if (traceDetail.observations.length === 0) {
      setSelectedObservationId(null)
      return
    }

    setSelectedObservationId((current) => {
      if (current && traceDetail.observations.some((observation) => observation.id === current)) {
        return current
      }
      return traceDetail.observations[0].id
    })
  }, [detailStatus, traceDetail])

  const hasTraces = traces.length > 0
  const selectedTrace = useMemo(
    () => traces.find((trace) => trace.id === selectedTraceId) ?? null,
    [traces, selectedTraceId],
  )

  const orderedObservations = useMemo(() => {
    if (!traceDetail) {
      return [] as LangfuseObservation[]
    }
    const ordered = [...traceDetail.observations].sort(
      (a, b) => new Date(a.startTime).getTime() - new Date(b.startTime).getTime(),
    )
    console.info('[Observability] Ordered observations', { count: ordered.length })
    return ordered
  }, [traceDetail])

  const observationById = useMemo(() => {
    if (!traceDetail) {
      return new Map<string, LangfuseObservation>()
    }
    return new Map(traceDetail.observations.map((observation) => [observation.id, observation]))
  }, [traceDetail])

  const selectedObservation = useMemo(() => {
    if (!orderedObservations.length) {
      return null
    }
    if (!selectedObservationId) {
      return orderedObservations[0]
    }
    return (
      orderedObservations.find((observation) => observation.id === selectedObservationId) ??
      orderedObservations[0]
    )
  }, [orderedObservations, selectedObservationId])

  const selectedObservationIndex = useMemo(() => {
    if (!selectedObservation) {
      return -1
    }
    return orderedObservations.findIndex((item) => item.id === selectedObservation.id)
  }, [orderedObservations, selectedObservation])

  const executionNarrative = useMemo(() => {
    if (!orderedObservations.length) {
      return null
    }

    // 1. User question (first user message we can find).
    let userQuestion: string | null = null
    for (const obs of orderedObservations) {
      const candidate = extractFirstUserMessage(obs)
      if (candidate) {
        userQuestion = candidate
        break
      }
    }

    // 2. Tool observations in this trace.
    const toolObservations = orderedObservations.filter(
      (obs) => obs.type === 'TOOL' && typeof obs.name === 'string' && obs.name,
    )
    const toolNames = Array.from(new Set(toolObservations.map((obs) => obs.name as string)))

    // 3. First reasoning step that mentions any of the known tools.
    let planningTools: string[] = []
    for (const obs of orderedObservations) {
      if (obs.type !== 'AGENT' && obs.type !== 'GENERATION' && obs.type !== 'SPAN') {
        continue
      }
      const reasoning = extractModelOutput(obs)
      if (!reasoning) {
        continue
      }
      const names = extractPlannedToolCallsFromReasoning(reasoning, toolNames)
      if (names.length) {
        planningTools = names
        break
      }
    }

    // 4. Executed tool summaries (one short line per distinct tool name).
    const toolsExecuted: { name: string; summary: string }[] = []
    const seenTools = new Set<string>()
    for (const obs of toolObservations) {
      const name = obs.name as string
      if (seenTools.has(name)) {
        continue
      }
      seenTools.add(name)

      let rawSummary: string
      if (typeof obs.output === 'string') {
        rawSummary = obs.output
      } else {
        rawSummary = formatPayload(obs.output)
      }
      const firstLine = rawSummary.split(/\r?\n/)[0] ?? ''
      const summary =
        firstLine.length > 180 ? `${firstLine.slice(0, 177).trimEnd()}…` : firstLine

      toolsExecuted.push({ name, summary })
    }

    // 5. Final assistant answer from the last generation/agent node.
    let finalAnswer: string | null = null
    for (let index = orderedObservations.length - 1; index >= 0; index -= 1) {
      const obs = orderedObservations[index]
      if (obs.type === 'GENERATION' || obs.type === 'AGENT') {
        const candidate = extractAssistantContent(obs)
        if (candidate) {
          finalAnswer = candidate
          break
        }
      }
    }

    if (!userQuestion && !planningTools.length && !toolsExecuted.length && !finalAnswer) {
      return null
    }

    return {
      userQuestion,
      planningTools,
      toolsExecuted,
      finalAnswer,
    }
  }, [orderedObservations])

  const effectiveNarrative: ExecutionNarrative | null = remoteNarrative ?? executionNarrative

  const selectedReasoning = useMemo(() => {
    if (!selectedObservation) {
      return null
    }
    return extractModelOutput(selectedObservation)
  }, [selectedObservation])

  const planVsExecution = useMemo(() => {
    if (!selectedObservation) {
      return null
    }

    // Only attempt plan/execution extraction for model-bearing observations
    // (agent/generation/model-chain nodes).
    if (
      selectedObservation.type !== 'AGENT' &&
      selectedObservation.type !== 'GENERATION' &&
      selectedObservation.type !== 'SPAN'
    ) {
      return null
    }

    const reasoning = selectedReasoning
    if (!reasoning) {
      return null
    }

    const toolNames = Array.from(
      new Set(
        orderedObservations
          .filter((obs) => obs.type === 'TOOL' && typeof obs.name === 'string' && obs.name)
          .map((obs) => obs.name as string),
      ),
    )

    const plannedNames = extractPlannedToolCallsFromReasoning(reasoning, toolNames)
    if (!plannedNames.length) {
      return null
    }

    const executed = new Set(
      orderedObservations
        .filter((obs) => obs.type === 'TOOL' && typeof obs.name === 'string')
        .map((obs) => obs.name as string)
        .filter((name) => plannedNames.includes(name)),
    )

    return {
      planned: plannedNames,
      executed,
      reasoning,
    }
  }, [orderedObservations, selectedObservation, selectedReasoning])

  const observationMetricsById = useMemo(() => {
    const map = new Map<string, ObservationMetric>()
    const severityValue = (metric: ObservationMetric) => {
      if (metric.label === 'CONTRADICTED') return 2 + metric.contradiction
      if (metric.label === 'NEUTRAL') return 1 + metric.contradiction
      return metric.entailment
    }
    observationMetrics.forEach((metric) => {
      const current = map.get(metric.observationId)
      if (!current || severityValue(metric) > severityValue(current)) {
        map.set(metric.observationId, metric)
      }
    })
    return map
  }, [observationMetrics])

  const flowGraph = useMemo(() => {
    if (!traceDetail) {
      return { nodes: [] as GraphNode[], edges: [] as GraphEdge[] }
    }

    const nodes: GraphNode[] = [
      {
        id: traceDetail.id,
        data: {
          label: traceDetail.name || 'Trace',
          kind: 'trace',
        },
        position: { x: 0, y: 0 },
        type: 'input',
        draggable: false,
        selectable: false,
        sourcePosition: Position.Right,
        style: {
          borderRadius: 10,
          padding: 10,
          fontWeight: 600,
          background: '#1d4ed8',
          color: '#ffffff',
          border: '3px solid #1e3a8a',
          boxShadow: '0 8px 20px rgba(15,23,42,0.12)',
        },
      },
    ]

    const observationMap = new Map(traceDetail.observations.map((observation) => [observation.id, observation]))
    const resolvedParentById = new Map<string, string | null>()

    const metadataParentKeys = [
      'parentObservationId',
      'parent_observation_id',
      'parent_run_id',
      'parentRunId',
      'trigger_observation_id',
      'triggerObservationId',
      'root_observation_id',
      'rootObservationId',
    ] as const

    const extractMetadataParentId = (observation: LangfuseObservation): string | null => {
      const metadata = observation.metadata ?? {}
      for (const key of metadataParentKeys) {
        const value = (metadata as Record<string, unknown>)[key]
        if (typeof value === 'string' && value) {
          return value
        }
      }
      return null
    }

    const isToolsSpan = (observation: LangfuseObservation | undefined): boolean => {
      if (!observation) {
        return false
      }

      // Many LangGraph orchestrator nodes show up as type "CHAIN" with
      // metadata.langgraph_node === "tools". We treat SPAN / AGENT / CHAIN as
      // eligible container nodes for tool calls.
      const type = String(observation.type ?? '').toLowerCase()
      if (!['span', 'agent', 'chain'].includes(type)) {
        return false
      }

      const name = (observation.name ?? '').toLowerCase()
      if (name.includes('tool')) {
        return true
      }

      const metadata = observation.metadata ?? {}
      const node =
        typeof (metadata as { langgraph_node?: unknown }).langgraph_node === 'string'
          ? ((metadata as { langgraph_node?: string }).langgraph_node ?? '').toLowerCase()
          : ''

      return node === 'tools' || node === 'tool'
    }

    const isModelSpan = (observation: LangfuseObservation | undefined): boolean => {
      if (!observation) {
        return false
      }

      // CHAIN observations that correspond to the "model" LangGraph node.
      const type = String(observation.type ?? '').toLowerCase()
      if (type !== 'chain') {
        return false
      }

      const name = (observation.name ?? '').toLowerCase()
      if (name.includes('model')) {
        return true
      }

      const metadata = observation.metadata ?? {}
      const node =
        typeof (metadata as { langgraph_node?: unknown }).langgraph_node === 'string'
          ? ((metadata as { langgraph_node?: string }).langgraph_node ?? '').toLowerCase()
          : ''

      return node === 'model'
    }

    const extractToolCallNamesFromOutput = (observation: LangfuseObservation): string[] => {
      const names = new Set<string>()
      const visit = (value: unknown): void => {
        if (!value) {
          return
        }
        if (Array.isArray(value)) {
          for (const item of value) {
            visit(item)
          }
          return
        }
        if (typeof value === 'object') {
          const record = value as Record<string, unknown>

          // OpenAI / tools-style schema: tool_calls array on the assistant message.
          if (Array.isArray(record.tool_calls)) {
            for (const call of record.tool_calls) {
              if (call && typeof call === 'object' && 'name' in call) {
                const name = (call as { name?: unknown }).name
                if (typeof name === 'string' && name) {
                  names.add(name)
                }
              }
            }
          }

          // LangGraph / Bedrock-style schema: individual messages of type "tool" that
          // carry the invoked tool name directly. These often appear inside an
          // output.messages array and look like:
          // { type: "tool", name: "banking.get_account_balance", ... }.
          const typeValue = record.type
          const nameValue = record.name
          if (
            typeof typeValue === 'string' &&
            typeValue.toLowerCase() === 'tool' &&
            typeof nameValue === 'string' &&
            nameValue
          ) {
            names.add(nameValue)
          }

          for (const nested of Object.values(record)) {
            visit(nested)
          }
        }
      }

      visit(observation.output)
      return Array.from(names)
    }

    const generationToolNamesById = new Map<string, string[]>()
    orderedObservations.forEach((observation) => {
      if (observation.type === 'GENERATION') {
        generationToolNamesById.set(observation.id, extractToolCallNamesFromOutput(observation))
      }
    })

    const resolveParentId = (observation: LangfuseObservation, index: number): string | null => {
      const directParent = observation.parentObservationId
      const metadataParent = extractMetadataParentId(observation)
      const directParentObs = directParent ? observationMap.get(directParent) : undefined
      const metadataParentObs = metadataParent ? observationMap.get(metadataParent) : undefined

      if (observation.type === 'TOOL') {
        const toolName = observation.name

        // 0. If a tools span directly wraps this tool, keep that structure.
        if (directParent && directParentObs && isToolsSpan(directParentObs)) {
          return directParent
        }
        if (metadataParent && metadataParentObs && isToolsSpan(metadataParentObs)) {
          return metadataParent
        }

        // 1. Prefer the GENERATION whose output contains a matching tool_call name.
        if (toolName) {
          for (let pointer = index - 1; pointer >= 0; pointer -= 1) {
            const candidate = orderedObservations[pointer]
            if (candidate.type !== 'GENERATION') {
              continue
            }
            const toolNames = generationToolNamesById.get(candidate.id)
            if (toolNames && toolNames.includes(toolName)) {
              return candidate.id
            }
          }
        }

        // 2. If direct/metadata parent is a GENERATION, keep that.
        if (directParent && directParentObs && directParentObs.type === 'GENERATION') {
          return directParent
        }

        if (metadataParent && metadataParentObs && metadataParentObs.type === 'GENERATION') {
          return metadataParent
        }

        // 3. Otherwise, prefer the nearest preceding GENERATION step as the branch anchor.
        for (let pointer = index - 1; pointer >= 0; pointer -= 1) {
          const candidate = orderedObservations[pointer]
          if (candidate.type === 'GENERATION') {
            return candidate.id
          }
        }

        // 4. Finally, attach to the nearest preceding non-TOOL step.
        for (let pointer = index - 1; pointer >= 0; pointer -= 1) {
          const candidate = orderedObservations[pointer]
          if (candidate.type !== 'TOOL') {
            return candidate.id
          }
        }
      }

      if (isToolsSpan(observation)) {
        // Try to hang this tools span under the generation that produced the tools.
        let toolNameFromChild: string | null = null
        for (let pointer = index + 1; pointer < orderedObservations.length; pointer += 1) {
          const candidate = orderedObservations[pointer]
          if (candidate.parentObservationId !== observation.id) {
            continue
          }
          if (candidate.type === 'TOOL' && candidate.name) {
            toolNameFromChild = candidate.name
            break
          }
        }

        if (toolNameFromChild) {
          for (let pointer = index - 1; pointer >= 0; pointer -= 1) {
            const candidate = orderedObservations[pointer]
            if (candidate.type !== 'GENERATION') {
              continue
            }
            const toolNames = generationToolNamesById.get(candidate.id)
            if (toolNames && toolNames.includes(toolNameFromChild)) {
              return candidate.id
            }
          }
        }

        // Even if we cannot match a specific tool name back to a generation,
        // it is usually more coherent for the diagram if the "tools" span is
        // visually anchored under the most recent GENERATION rather than
        // directly under the agent/root.
        for (let pointer = index - 1; pointer >= 0; pointer -= 1) {
          const candidate = orderedObservations[pointer]
          if (candidate.type === 'GENERATION') {
            return candidate.id
          }
        }
      }

      if (isModelSpan(observation)) {
        // Model chain steps typically consume tool outputs, so we prefer to
        // visually attach them under tools/tool spans rather than directly
        // under the agent/root.

        if (
          directParent &&
          directParentObs &&
          (isToolsSpan(directParentObs) || directParentObs.type === 'TOOL')
        ) {
          return directParent
        }

        if (
          metadataParent &&
          metadataParentObs &&
          (isToolsSpan(metadataParentObs) || metadataParentObs.type === 'TOOL')
        ) {
          return metadataParent
        }

        // 1. Prefer the nearest preceding TOOL as the visual parent.
        for (let pointer = index - 1; pointer >= 0; pointer -= 1) {
          const candidate = orderedObservations[pointer]
          if (candidate.type === 'TOOL') {
            return candidate.id
          }
        }

        // 2. Otherwise, prefer the nearest preceding tools span.
        for (let pointer = index - 1; pointer >= 0; pointer -= 1) {
          const candidate = orderedObservations[pointer]
          if (isToolsSpan(candidate)) {
            return candidate.id
          }
        }

        // 3. As a final preference before generic fallbacks, hang under the
        // most recent GENERATION step.
        for (let pointer = index - 1; pointer >= 0; pointer -= 1) {
          const candidate = orderedObservations[pointer]
          if (candidate.type === 'GENERATION') {
            return candidate.id
          }
        }
      }

      // Non-TOOL observations (and tools spans that we couldn't re-anchor) keep the
      // physical parent relationships when present.
      if (directParent && (directParent === traceDetail.id || observationMap.has(directParent))) {
        return directParent
      }

      if (metadataParent && (metadataParent === traceDetail.id || observationMap.has(metadataParent))) {
        return metadataParent
      }

      if (index > 0) {
        return orderedObservations[index - 1]?.id ?? traceDetail.id
      }

      return traceDetail.id
    }

    orderedObservations.forEach((observation, index) => {
      const resolvedParent = resolveParentId(observation, index)
      resolvedParentById.set(observation.id, resolvedParent)
    })

    const depthCache = new Map<string, number>()
    const computeDepth = (nodeId: string): number => {
      const cached = depthCache.get(nodeId)
      if (cached !== undefined) {
        return cached
      }

      const parentId = resolvedParentById.get(nodeId)
      if (!parentId || parentId === traceDetail.id) {
        depthCache.set(nodeId, 1)
        return 1
      }

      const depth = computeDepth(parentId) + 1
      depthCache.set(nodeId, depth)
      return depth
    }

    orderedObservations.forEach((observation, index) => {
      const depth = computeDepth(observation.id)
      const metric = observationMetricsById.get(observation.id)
      const isSelected = selectedObservation ? observation.id === selectedObservation.id : index === 0
      const hasIssue = metric?.label === 'CONTRADICTED'
      const borderline = metric?.label === 'NEUTRAL'
      const { style: nodeStyle, badgeStyles } = getNodeVisuals({
        kind: observation.type,
        isSelected,
        hasIssue: Boolean(hasIssue),
        borderline: Boolean(borderline),
      })
      const metadata = observation.metadata ?? {}
      const nodeTitle =
        observation.name ||
        (typeof metadata.langgraph_node === 'string' && metadata.langgraph_node) ||
        observation.type.toLowerCase()

      const metricPill = metric ? (
        <span className={`flow-node__metric flow-node__metric--${metric.label.toLowerCase()}`}>
          {metric.label === 'CONTRADICTED'
            ? 'Contradiction'
            : metric.label === 'NEUTRAL'
            ? 'Needs review'
            : 'Grounded'}
        </span>
      ) : null

      const rootPill = rootCauseObservationId === observation.id ? (
        <span className="flow-node__metric flow-node__metric--root">Root cause</span>
      ) : null

      const icon = NODE_ICONS[observation.type as keyof typeof NODE_ICONS] ?? NODE_ICONS.default
      const nodeLabel = (
        <div className="flow-node">
          <div className="flow-node__top">
            <span className="flow-node__badge" style={badgeStyles}>
              {observation.type.toLowerCase()}
            </span>
            <span className="flow-node__step">Step {index + 1}</span>
          </div>
          <div className="flow-node__title-row">
            <div className="flow-node__icon">{icon}</div>
            <div className="flow-node__title">{nodeTitle}</div>
          </div>
          {metricPill || rootPill ? (
            <div className="flow-node__meta">
              {metricPill}
              {rootPill}
            </div>
          ) : null}
        </div>
      )

      nodes.push({
        id: observation.id,
        position: {
          x: depth * HORIZONTAL_SPACING,
          y: ROOT_OFFSET_Y + index * VERTICAL_SPACING,
        },
        data: {
          label: nodeLabel,
          kind: observation.type,
          step: index + 1,
        },
        draggable: false,
        selectable: true,
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        style: nodeStyle,
      })
    })

    const edges: GraphEdge[] = []
    const edgeIds = new Set<string>()

    orderedObservations.forEach((observation, index) => {
      const primarySource = resolvedParentById.get(observation.id) ?? traceDetail.id
      const primaryEdgeId = `${primarySource}→${observation.id}`

      if (!edgeIds.has(primaryEdgeId)) {
        edges.push({
          id: primaryEdgeId,
          source: primarySource,
          target: observation.id,
          animated: observation.type === 'TOOL',
        })
        edgeIds.add(primaryEdgeId)
      }

      // For model chain steps, also draw fan-in edges from any TOOL
      // observations that occurred between the last GENERATION and this
      // model node. This makes the flow 7→9 and 8→9 (before 9→10) more
      // visually explicit.
      if (isModelSpan(observation)) {
        let lastGenerationIndex = -1
        for (let pointer = index - 1; pointer >= 0; pointer -= 1) {
          if (orderedObservations[pointer].type === 'GENERATION') {
            lastGenerationIndex = pointer
            break
          }
        }

        if (lastGenerationIndex >= 0) {
          for (let pointer = lastGenerationIndex + 1; pointer < index; pointer += 1) {
            const candidate = orderedObservations[pointer]
            if (candidate.type !== 'TOOL') {
              continue
            }
            const extraEdgeId = `${candidate.id}→${observation.id}`
            if (edgeIds.has(extraEdgeId)) {
              continue
            }
            edges.push({
              id: extraEdgeId,
              source: candidate.id,
              target: observation.id,
              animated: true,
            })
            edgeIds.add(extraEdgeId)
          }
        }
      }
    })

    console.info('[Observability] Graph built', { nodes: nodes.length, edges: edges.length })
    return { nodes, edges }
  }, [traceDetail, orderedObservations, selectedObservation, observationMetricsById, rootCauseObservationId])

  const graphSubtitle = useMemo(() => {
    if (!selectedTrace) {
      return 'Select a trace from the table to inspect its execution flow.'
    }

    if (detailStatus === 'loading') {
      return 'Loading observations…'
    }

    if (detailStatus === 'error') {
      return 'We hit a snag loading observations.'
    }

    const observationCount = orderedObservations.length
    const observationLabel = `${observationCount} observation${observationCount === 1 ? '' : 's'}`
    return `Started ${formatDateTime(selectedTrace.timestamp)} • ${observationLabel}`
  }, [selectedTrace, detailStatus, orderedObservations])

  const highlight = selectedObservation ? buildHighlight(selectedObservation) : null
  const detailMetadata = selectedObservation?.metadata as Record<string, unknown> | undefined
  const detailInput = selectedObservation?.metadata?.inputs ?? selectedObservation?.input ?? null
  const detailOutput = selectedObservation?.metadata?.output ?? selectedObservation?.output ?? null
  const formattedInput = formatPayload(detailInput)
  const formattedOutput = formatPayload(detailOutput)
  const formattedMetadata = formatPayload(detailMetadata ?? {})
  const selectedMetric = selectedObservation ? observationMetricsById.get(selectedObservation.id) ?? null : null
  const selectedInsights = useMemo(() => {
    if (!selectedObservation) {
      return [] as ObservationInsight[]
    }
    return observationInsights.filter((insight) => insight.observationId === selectedObservation.id)
  }, [selectedObservation, observationInsights])

  const loadingTraceId = detailStatus === 'loading' ? selectedTraceId : null

  const handleNodeClick = useCallback(
    (_: unknown, node: GraphNode) => {
      if (!traceDetail || node.id === traceDetail.id) {
        return
      }
      const observation = observationById.get(node.id)
      if (observation) {
        setSelectedObservationId(observation.id)
      }
    },
    [traceDetail, observationById],
  )

  const handleFlowInit = useCallback((instance: ReactFlowInstance) => {
    window.requestAnimationFrame(() => {
      instance.fitView({ padding: 0.2 })
    })
  }, [])

  const handleProcessTrace = useCallback(async () => {
      if (!traceDetail || orderedObservations.length === 0 || narrativeStatus === 'processing') {
        return
      }
      setNarrativeStatus('processing')
      updatePendingTrace(traceDetail.id)

      const payload = {
        trace_id: traceDetail.id,
        trace_name: traceDetail.name ?? null,
        observations: orderedObservations,
        nli_model: nliModel,
        llm_model: llmModel,
      }

      try {
        const response = await fetch(TRACE_SUMMARY_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        })
        if (!response.ok) {
          const text = await response.text().catch(() => '')
          throw new Error(
            `Trace summary request failed: ${response.status} ${response.statusText} ${text}`,
          )
        }
        const data = (await response.json()) as Record<string, unknown>
        const value = data as Record<string, unknown>
        const toolsRaw = Array.isArray(value.toolsExecuted) ? value.toolsExecuted : []
        const tools: { name: string; summary: string }[] = toolsRaw
          .map((item) => {
            if (!item || typeof item !== 'object') {
              return null
            }
            const record = item as { name?: unknown; summary?: unknown }
            const name = typeof record.name === 'string' ? record.name.trim() : ''
            if (!name) {
              return null
            }
            const summary =
              typeof record.summary === 'string' && record.summary.trim()
                ? record.summary.trim()
                : name
            return { name, summary }
          })
          .filter((item): item is { name: string; summary: string } => item !== null)

        const planningRaw = Array.isArray(value.planningTools) ? value.planningTools : []
        const planningTools = planningRaw.map((item) => String(item))

        const userQuestion =
          typeof value.userQuestion === 'string' && value.userQuestion.trim()
            ? value.userQuestion
            : null

        const finalAnswer =
          typeof value.finalAnswer === 'string' && value.finalAnswer.trim()
            ? value.finalAnswer
            : null

        const narrative: ExecutionNarrative = {
          userQuestion,
          planningTools,
          toolsExecuted: tools,
          finalAnswer,
        }

        const isEmptyNarrative =
          !userQuestion && !finalAnswer && planningTools.length === 0 && tools.length === 0

        // Only override the local, trace-derived executionNarrative when the
        // backend actually returns some content. If the LLM is unavailable or
        // returns an effectively empty object, we keep the existing narrative
        // so the UI still shows a useful story.
        const nextNarrative = isEmptyNarrative ? null : narrative
        setRemoteNarrative(nextNarrative)
        if (traceDetail) {
          persistNarrativeForTrace(traceDetail.id, nextNarrative)
        }

        const summaryRaw = value.reasoningSummary
        let nextReasoningSummary: ReasoningSummary | null = null
        if (summaryRaw && typeof summaryRaw === 'object') {
          const goalValue =
            typeof (summaryRaw as { goal?: unknown }).goal === 'string'
              ? ((summaryRaw as { goal?: string }).goal ?? '').trim()
              : ''
          const planValue = Array.isArray((summaryRaw as { plan?: unknown[] }).plan)
            ? ((summaryRaw as { plan?: unknown[] }).plan ?? [])
                .map((item) => String(item).trim())
                .filter((item) => item.length > 0)
            : []
          const observationsValue = Array.isArray((summaryRaw as { observations?: unknown[] }).observations)
            ? ((summaryRaw as { observations?: unknown[] }).observations ?? [])
                .map((item) => String(item).trim())
                .filter((item) => item.length > 0)
            : []
          const resultValue =
            typeof (summaryRaw as { result?: unknown }).result === 'string'
              ? ((summaryRaw as { result?: string }).result ?? '').trim()
              : ''
          nextReasoningSummary = {
            goal: goalValue || null,
            plan: planValue,
            observations: observationsValue,
            result: resultValue || null,
          }
        }
        setReasoningSummary(nextReasoningSummary)

        const insightsRaw = Array.isArray(value.observationInsights) ? value.observationInsights : []
        const parsedInsights = insightsRaw
          .map((entry) => {
            if (!entry || typeof entry !== 'object') {
              return null
            }
            const record = entry as {
              observationId?: unknown
              stage?: unknown
              summary?: unknown
              bullets?: unknown
            }
            const observationId = typeof record.observationId === 'string' ? record.observationId : null
            if (!observationId) {
              return null
            }
            const stage = typeof record.stage === 'string' ? record.stage : null
            const summary = typeof record.summary === 'string' ? record.summary : null
            const bullets = Array.isArray(record.bullets)
              ? record.bullets
                  .map((item) => String(item).trim())
                  .filter((item) => item.length > 0)
                  .slice(0, 4)
              : []
            return { observationId, stage, summary, bullets }
          })
          .filter((item): item is ObservationInsight => item !== null)
        setObservationInsights(parsedInsights)

        const metricsRaw = Array.isArray(value.observationMetrics) ? value.observationMetrics : []
        const parsedMetrics = metricsRaw
          .map((entry) => {
            if (!entry || typeof entry !== 'object') {
              return null
            }
            const record = entry as ObservationMetric
            if (typeof record.observationId !== 'string') {
              return null
            }
            return {
              observationId: record.observationId,
              metric: typeof record.metric === 'string' ? record.metric : 'groundedness',
              subject: typeof record.subject === 'string' ? record.subject : 'response',
              entailment: Number(record.entailment) || 0,
              contradiction: Number(record.contradiction) || 0,
              neutral: Number(record.neutral) || 0,
              label: typeof record.label === 'string' ? record.label : 'NEUTRAL',
            }
          })
          .filter((item): item is ObservationMetric => item !== null)
        setObservationMetrics(parsedMetrics)

        const rootCause =
          typeof value.rootCauseObservationId === 'string' && value.rootCauseObservationId
            ? value.rootCauseObservationId
            : null
        setRootCauseObservationId(rootCause)

        const carbonRaw = value.carbonSummary
        const nextCarbonSummary: CarbonSummary | null =
          carbonRaw && typeof carbonRaw === 'object'
            ? (carbonRaw as CarbonSummary)
            : null
        setCarbonSummary(nextCarbonSummary)
        setNarrativeStatus('ready')

        if (traceDetail) {
          const storedPayload: StoredTraceSummary = {
            narrative: nextNarrative,
            reasoningSummary: nextReasoningSummary,
            observationInsights: parsedInsights,
            observationMetrics: parsedMetrics,
            rootCauseObservationId: rootCause,
            carbonSummary: nextCarbonSummary,
            timestamp: Date.now(),
          }
          persistStoredTraceSummary(traceDetail.id, storedPayload)
          setStoredSummary(storedPayload)
        }
      } catch (error) {
        console.error('[Observability] Failed to summarise trace', error)
        setRemoteNarrative(null)
        if (traceDetail) {
          persistNarrativeForTrace(traceDetail.id, null)
        }
        setReasoningSummary(null)
        setObservationInsights([])
        setObservationMetrics([])
        setRootCauseObservationId(null)
        setCarbonSummary(null)
        setNarrativeStatus('ready')
      } finally {
        updatePendingTrace(null)
      }
    },
    [traceDetail, orderedObservations, narrativeStatus, updatePendingTrace, nliModel, llmModel],
  )

  useEffect(() => {
    if (!pendingTrace || !traceDetail || detailStatus !== 'loaded') {
      return
    }
    if (pendingTrace.traceId !== traceDetail.id) {
      return
    }
    if (narrativeStatus !== 'idle') {
      return
    }
    void handleProcessTrace()
  }, [pendingTrace, traceDetail, detailStatus, narrativeStatus, handleProcessTrace])

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>AgentGlass: Observability for LLM Agents</h1>
        <p>
          Visualise the full flow of tool calls, break down model thoughts into digestible story beats, and score
          how well each step aligns with the agent’s objectives—all in one workspace.
        </p>
      </header>

      <section className="app-card connection-card">
        <div className="connection-card__heading">
          <div>
            <p className="connection-card__eyebrow">Observability platform</p>
            <h2>Connect your trace source</h2>
            <p>
              Switch between Langfuse or LangSmith and manage API keys without touching env files. Empty fields
              fall back to your local configuration.
            </p>
          </div>
          <span className={`status-chip ${activeApiKey ? 'status-chip--success' : 'status-chip--muted'}`}>
            {apiKeyStatusLabel}
          </span>
        </div>

        <div className="connection-card__body">
          <div className="provider-options">
            {PROVIDER_OPTIONS.map((option) => (
              <label
                key={option.id}
                className={
                  provider === option.id
                    ? 'provider-option provider-option--active'
                    : 'provider-option'
                }
              >
                <input
                  type="radio"
                  name="observability-provider"
                  value={option.id}
                  checked={provider === option.id}
                  onChange={() => handleProviderSelect(option.id)}
                />
                <span className="provider-option__label">{option.label}</span>
                <span className="provider-option__helper">{option.helper}</span>
              </label>
            ))}
          </div>

          <div className="api-key-field">
            <label htmlFor="observability-api-key">API key</label>
            <div className="api-key-input-group">
              <input
                id="observability-api-key"
                type="password"
                placeholder={`Paste your ${provider === 'langsmith' ? 'LangSmith' : 'Langfuse'} API key`}
                value={pendingApiKey}
                onChange={(event) => setPendingApiKey(event.target.value)}
                autoComplete="off"
                spellCheck={false}
              />
              <button type="button" className="secondary-button" onClick={handleSaveApiKey}>
                Save key
              </button>
            </div>
            <p className="field-hint">
              Keys are stored locally in your browser for convenience. Leave blank to continue using your .env
              configuration.
            </p>
          </div>
        </div>
      </section>

      <section className="app-card">
          <div className="panel-header">
            <button
              type="button"
              className="primary-button"
              onClick={handleConnect}
              disabled={status === 'connecting'}
            >
              {status === 'connecting'
                ? 'Connecting…'
                : status === 'connected'
                ? 'Refresh traces'
                : 'Connect to Langfuse'}
            </button>
            <span className={`status-badge status-${status}`}>{statusLabel}</span>
          </div>

          {status === 'error' && errorMessage ? (
            <div className="error-banner" role="alert">
              {errorMessage}
            </div>
          ) : null}

          {hasTraces ? (
            <div className="trace-table-scroll">
              <TraceTable
                traces={traces}
                onSelect={handleSelectTrace}
                onProcess={handleProcessTrace}
                selectedTraceId={selectedTraceId}
                loadingTraceId={loadingTraceId}
                processingTraceId={narrativeStatus === 'processing' ? pendingTrace?.traceId ?? traceDetail?.id ?? null : null}
              />
            </div>
          ) : (
            <div className="empty-state">
              <h2>Preview your Langfuse traces</h2>
              <p>
                Click the button above to import recent traces. Store credentials in
                <code>.env</code> (see <code>.env.example</code>) so this interface can connect on your behalf.
              </p>
            </div>
          )}
      </section>

      <section className="app-card settings-panel">
        <div className="panel-header">
          <div>
            <h2>Model settings</h2>
            <p>Select models for groundedness scoring and narrative generation. These do not affect the agent.</p>
          </div>
        </div>
        <div className="settings-grid">
          <div className="settings-field">
            <label htmlFor="nli-model">NLI model</label>
            <select
              id="nli-model"
              value={nliModel}
              onChange={(e) => setNliModel(e.target.value)}
            >
              {NLI_MODEL_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <div className="settings-field">
            <label htmlFor="llm-model">Narrative LLM</label>
            <select
              id="llm-model"
              value={llmModel}
              onChange={(e) => setLlmModel(e.target.value)}
            >
              {LLM_MODEL_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </section>

      <section className="app-card graph-card">
          <div className="graph-header">
            <div className="graph-header__title">
              <h2>{selectedTrace ? selectedTrace.name || 'Trace flow' : 'Trace flow'}</h2>
              <p>{graphSubtitle}</p>
            </div>
            {(storedSummaryTimestamp || selectedTrace) ? (
              <div className="graph-header__meta">
                {storedSummaryTimestamp ? (
                  <div className="stored-summary-chip">Cached {storedSummaryTimestamp}</div>
                ) : null}
                {selectedTrace ? (
                  <div className="graph-header-actions">
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={handleProcessTrace}
                      disabled={
                        detailStatus !== 'loaded' ||
                        orderedObservations.length === 0 ||
                        narrativeStatus === 'processing'
                      }
                    >
                      {narrativeStatus === 'idle'
                        ? 'Process trace'
                        : narrativeStatus === 'processing'
                        ? 'Processing…'
                        : 'Reprocess trace'}
                    </button>
                    {storedSummary ? (
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={handleLoadStoredSummary}
                        disabled={narrativeStatus === 'processing'}
                      >
                        Load cached summary
                      </button>
                    ) : null}
                    <a
                      className="secondary-button"
                      href={selectedTrace.url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open in Langfuse
                    </a>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>

          {detailStatus === 'loading' ? (
            <div className="info-banner">Loading trace observations…</div>
          ) : null}

          {detailStatus === 'error' && detailError ? (
            <div className="error-banner" role="alert">
              {detailError}
            </div>
          ) : null}

          {detailStatus === 'loaded' && orderedObservations.length === 0 ? (
            <div className="empty-state graph-empty-state">No observations recorded for this trace yet.</div>
          ) : null}

          {detailStatus !== 'loading' && !selectedTrace ? (
            <div className="empty-state graph-empty-state">
              Select a trace to render its flow graph.
            </div>
          ) : null}

          {detailStatus === 'loaded' && orderedObservations.length > 0 ? (
            <>
              <div className="graph-metrics">
                <div className="graph-metric-card graph-metric-card--counts">
                  <div className="graph-metric-card__header">
                    <span className="graph-metric-card__label">Groundedness overview</span>
                    <span className="graph-metric-card__stat">
                      {observationMetrics.length} sample{observationMetrics.length === 1 ? '' : 's'}
                    </span>
                  </div>
                  <ul className="metric-distribution">
                    {metricDistribution.map((entry) => (
                      <li key={entry.key} className="metric-distribution__item">
                        <div className="metric-distribution__label">
                          <span className={`metric-dot metric-dot--${entry.key}`} aria-hidden />
                          <div>
                            <strong>{entry.label}</strong>
                            <span>{entry.count} steps</span>
                          </div>
                        </div>
                        <div className="metric-distribution__bar" aria-hidden>
                          <div
                            className={`metric-distribution__fill metric-distribution__fill--${entry.key}`}
                            style={{ width: `${entry.percent}%` }}
                          />
                        </div>
                        <span className="metric-distribution__percent">{entry.percent}%</span>
                      </li>
                    ))}
                  </ul>
                </div>
                {carbonSummary ? (
                  <div className="graph-metric-card graph-metric-card--carbon">
                    <div className="graph-metric-card__header">
                      <span className="graph-metric-card__label">Sustainability impact</span>
                      <span className="graph-metric-card__stat">{formatMgSig(carbonSummary.total_kg)} mg CO₂e</span>
                    </div>
                    <ul className="metric-distribution">
                      <li className="metric-distribution__item">
                        <div className="metric-distribution__label">
                          <span className="metric-dot metric-dot--grounded" aria-hidden />
                          <div>
                            <strong>Reasoning</strong>
                            <span>{formatMgSig(carbonSummary.breakdown.reasoning.kg)} mg CO₂e</span>
                          </div>
                        </div>
                        <div className="metric-distribution__bar" aria-hidden>
                          <div
                            className="metric-distribution__fill metric-distribution__fill--grounded"
                            style={{
                              width: `${Math.min(100, Math.max(0, Math.round(
                                (carbonSummary.breakdown.reasoning.kg / Math.max(1e-9, carbonSummary.total_kg)) * 100
                              )))}%`,
                            }}
                          />
                        </div>
                        <span className="metric-distribution__percent">
                          {Math.min(100, Math.max(0, Math.round(
                            (carbonSummary.breakdown.reasoning.kg / Math.max(1e-9, carbonSummary.total_kg)) * 100
                          )))}%
                        </span>
                      </li>
                      <li className="metric-distribution__item">
                        <div className="metric-distribution__label">
                          <span className="metric-dot metric-dot--neutral" aria-hidden />
                          <div>
                            <strong>Rest</strong>
                            <span>{formatMgSig(carbonSummary.breakdown.rest.kg)} mg CO₂e</span>
                          </div>
                        </div>
                        <div className="metric-distribution__bar" aria-hidden>
                          <div
                            className="metric-distribution__fill metric-distribution__fill--neutral"
                            style={{
                              width: `${Math.min(100, Math.max(0, Math.round(
                                (carbonSummary.breakdown.rest.kg / Math.max(1e-9, carbonSummary.total_kg)) * 100
                              )))}%`,
                            }}
                          />
                        </div>
                        <span className="metric-distribution__percent">
                          {Math.min(100, Math.max(0, Math.round(
                            (carbonSummary.breakdown.rest.kg / Math.max(1e-9, carbonSummary.total_kg)) * 100
                          )))}%
                        </span>
                      </li>
                    </ul>
                    <div className="graph-metric-card__foot">source: {carbonSummary.source || 'CodeCarbon'}</div>
                  </div>
                ) : (
                  <div className="graph-metric-card graph-metric-card--carbon">
                    <div className="graph-metric-card__header">
                      <span className="graph-metric-card__label">Sustainability impact</span>
                      <span className="graph-metric-card__stat">—</span>
                    </div>
                    <p className="graph-metric-empty">Process this trace to generate sustainability metrics.</p>
                  </div>
                )}
                <div className="graph-metric-card graph-metric-card--raw">
                  <div className="graph-metric-card__header">
                    <span className="graph-metric-card__label">Recent scores</span>
                    <span className="graph-metric-card__stat">Sampled snapshots</span>
                  </div>
                  {rawMetricSamples.length > 0 ? (
                    <ul className="graph-metric-raw-list">
                      {rawMetricSamples.map((item) => (
                        <li key={item.key} className="graph-metric-raw">
                          <div className="graph-metric-raw__title">
                            <strong>{item.title}</strong>
                            <span>{Math.round(item.entailment * 100)}% grounded</span>
                          </div>
                          <div className="graph-metric-raw__stack" aria-hidden>
                            <span
                              className="graph-metric-raw__segment graph-metric-raw__segment--contradiction"
                              style={{ width: `${Math.max(0, Math.round(item.contradiction * 100))}%` }}
                            />
                            <span
                              className="graph-metric-raw__segment graph-metric-raw__segment--neutral"
                              style={{ width: `${Math.max(0, Math.round(item.neutral * 100))}%` }}
                            />
                            <span
                              className="graph-metric-raw__segment graph-metric-raw__segment--grounded"
                              style={{ width: `${Math.max(0, Math.round(item.entailment * 100))}%` }}
                            />
                          </div>
                          <div className="graph-metric-raw__legend">
                            <span className="legend-chip legend-chip--grounded">
                              <span className="legend-dot legend-dot--grounded" />
                              {Math.round(item.entailment * 100)}% grounded
                            </span>
                            <span className="legend-chip legend-chip--neutral">
                              <span className="legend-dot legend-dot--neutral" />
                              {Math.round(item.neutral * 100)}% neutral
                            </span>
                            <span className="legend-chip legend-chip--contradiction">
                              <span className="legend-dot legend-dot--contradiction" />
                              {Math.round(item.contradiction * 100)}% contradicted
                            </span>
                          </div>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="graph-metric-empty">Process this trace to see groundedness metrics here.</p>
                  )}
                </div>
                <div className="graph-metric-card graph-metric-card--explainer">
                  <div className="graph-metric-card__header">
                    <span className="graph-metric-card__label">About this metric</span>
                  </div>
                  <details className="graph-metric-explainer" open>
                    <summary>How groundedness is labeled</summary>
                    <ul className="metric-guidelines">
                      <li>Premise = user question + up to 6 recent tool summaries.</li>
                      <li>Hypotheses = reasoning snippet and final answer for each agent/model step.</li>
                      <li>Cross-encoder NLI buckets: contradiction ≥ 0.5, entailment ≥ 0.6, else neutral.</li>
                    </ul>
                  </details>
                </div>
              </div>
              <div className="graph-content">
              <div className="graph-container">
                <ReactFlowProvider>
                  <ReactFlow
                    nodes={flowGraph.nodes}
                    edges={flowGraph.edges}
                    style={{ width: '100%', height: '100%' }}
                    defaultViewport={{ x: 0, y: 0, zoom: 1 }}
                    nodesDraggable={false}
                    nodesConnectable={false}
                    elementsSelectable
                    panOnScroll
                    zoomOnScroll={false}
                    proOptions={{ hideAttribution: true }}
                    onNodeClick={handleNodeClick}
                    onInit={handleFlowInit}
                  >
                    <Background gap={24} />
                    <MiniMap zoomable pannable />
                    <Controls showInteractive={false} position="bottom-right" />
                  </ReactFlow>
                </ReactFlowProvider>
              </div>

              <aside className="observation-detail">
                {selectedObservation ? (
                  <div className="observation-detail__content">
                    <header className="observation-detail__header">
                      <span className="observation-step">Step {selectedObservationIndex + 1}</span>
                      <h3>{toNodeLabel(selectedObservation, selectedObservationIndex + 1)}</h3>
                      <p>{formatDateTime(selectedObservation.startTime)}</p>
                    </header>

                    {selectedMetric ? (
                      <div className="observation-metric-card">
                        <div className="observation-metric-card__header">
                          <span className="observation-metric-card__eyebrow">Reasoning credibility</span>
                          <span
                            className={`metric-label metric-label--${selectedMetric.label.toLowerCase()}`}
                          >
                            {selectedMetric.label === 'CONTRADICTED'
                              ? 'Contradicted'
                              : selectedMetric.label === 'NEUTRAL'
                              ? 'Needs review'
                              : 'Grounded'}
                          </span>
                        </div>
                        <h4>{selectedMetric.subject}</h4>
                        <p className="observation-metric-card__summary">
                          {selectedMetric.metric === 'groundedness'
                            ? 'LLM output checked against trace evidence.'
                            : selectedMetric.metric}
                        </p>
                        <div className="metric-bars">
                          <div className="metric-bar">
                            <span>Grounded</span>
                            <div className="metric-bar__track">
                              <div
                                className="metric-bar__fill metric-bar__fill--entailment"
                                style={{ width: `${Math.min(100, Math.max(0, selectedMetric.entailment * 100))}%` }}
                              />
                            </div>
                            <strong>{Math.round(selectedMetric.entailment * 100)}%</strong>
                          </div>
                          <div className="metric-bar">
                            <span>Needs review</span>
                            <div className="metric-bar__track">
                              <div
                                className="metric-bar__fill metric-bar__fill--neutral"
                                style={{ width: `${Math.min(100, Math.max(0, selectedMetric.neutral * 100))}%` }}
                              />
                            </div>
                            <strong>{Math.round(selectedMetric.neutral * 100)}%</strong>
                          </div>
                          <div className="metric-bar">
                            <span>Contradiction</span>
                            <div className="metric-bar__track">
                              <div
                                className="metric-bar__fill metric-bar__fill--contradiction"
                                style={{ width: `${Math.min(100, Math.max(0, selectedMetric.contradiction * 100))}%` }}
                              />
                            </div>
                            <strong>{Math.round(selectedMetric.contradiction * 100)}%</strong>
                          </div>
                        </div>
                        {rootCauseObservationId === selectedObservation.id ? (
                          <div className="observation-metric-card__root">
                            Flagged as likely root-cause for failed reasoning.
                          </div>
                        ) : null}
                      </div>
                    ) : null}

                    {reasoningSummary ? (
                      <div className="reasoning-summary-card">
                        <div className="reasoning-summary-card__header">
                          <span className="reasoning-summary-card__eyebrow">Reasoning timeline</span>
                          <p>LLM-decomposed view of the agent’s plan, observations, and conclusion.</p>
                        </div>
                        {reasoningSummary.goal ? (
                          <div className="reasoning-summary-card__section">
                            <span>Goal</span>
                            <p>{reasoningSummary.goal}</p>
                          </div>
                        ) : null}
                        {reasoningSummary.plan.length > 0 ? (
                          <div className="reasoning-summary-card__section">
                            <span>Plan</span>
                            <ul>
                              {reasoningSummary.plan.map((step, index) => (
                                <li key={`${step}-${index}`}>{step}</li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                        {reasoningSummary.observations.length > 0 ? (
                          <div className="reasoning-summary-card__section">
                            <span>Observations</span>
                            <ul>
                              {reasoningSummary.observations.map((item, index) => (
                                <li key={`${item}-${index}`}>{item}</li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                        {reasoningSummary.result ? (
                          <div className="reasoning-summary-card__section">
                            <span>Result</span>
                            <p>{reasoningSummary.result}</p>
                          </div>
                        ) : null}
                      </div>
                    ) : null}

                    {selectedInsights.length > 0 ? (
                      <div className="observation-insights-card">
                        <h4>Insight for this step</h4>
                        <div className="observation-insights-card__list">
                          {selectedInsights.map((insight) => (
                            <article key={`${insight.observationId}-${insight.stage}`}>
                              <header>
                                {insight.stage ? <span>{insight.stage}</span> : null}
                                {insight.summary ? <h5>{insight.summary}</h5> : null}
                              </header>
                              {insight.bullets.length > 0 ? (
                                <ul>
                                  {insight.bullets.map((bullet, index) => (
                                    <li key={`${insight.observationId}-${index}`}>{bullet}</li>
                                  ))}
                                </ul>
                              ) : null}
                            </article>
                          ))}
                        </div>
                      </div>
                    ) : null}

                    {narrativeStatus === 'processing' ? (
                      <div className="execution-narrative">
                        <h4>Execution narrative</h4>
                        <p>Processing trace…</p>
                      </div>
                    ) : null}

                    {narrativeStatus === 'ready' && effectiveNarrative ? (
                      <div className="execution-narrative">
                        <div className="timeline">
                          <div className="timeline__header">
                            <h4>Execution narrative</h4>
                            <p className="timeline__subhead">From user query to final answer</p>
                          </div>
                          <ol>
                            {effectiveNarrative.userQuestion ? (
                              <li className="timeline__item">
                                <span className="timeline__dot" />
                                <div className="timeline__card">
                                  <span className="timeline__label">User asked</span>
                                  <p>{truncate(effectiveNarrative.userQuestion, 200)}</p>
                                </div>
                              </li>
                            ) : null}
                            {effectiveNarrative.planningTools.length > 0 ? (
                              <li className="timeline__item">
                                <span className="timeline__dot timeline__dot--accent" />
                                <div className="timeline__card">
                                  <span className="timeline__label">Planned tool calls</span>
                                  <ul className="execution-narrative__tools-list">
                                    {effectiveNarrative.planningTools.map((tool) => (
                                      <li key={tool} className="execution-narrative__tool-chip">{tool}</li>
                                    ))}
                                  </ul>
                                </div>
                              </li>
                            ) : null}
                            {effectiveNarrative.toolsExecuted.length > 0 ? (
                              <li className="timeline__item">
                                <span className="timeline__dot" />
                                <div className="timeline__card">
                                  <span className="timeline__label">Tools executed</span>
                                  <ul className="execution-narrative__tools-list execution-narrative__tools-list--stacked">
                                    {effectiveNarrative.toolsExecuted.map((item) => (
                                      <li key={item.name}>
                                        <div className="execution-narrative__tool-chip">✅ {item.name}</div>
                                        <p className="execution-narrative__tool-summary">{truncate(item.summary, 200)}</p>
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              </li>
                            ) : null}
                            {effectiveNarrative.finalAnswer ? (
                              <li className="timeline__item">
                                <span className="timeline__dot timeline__dot--accent" />
                                <div className="timeline__card">
                                  <span className="timeline__label">Model concluded</span>
                                  <p>{truncate(effectiveNarrative.finalAnswer, 260)}</p>
                                </div>
                              </li>
                            ) : null}
                          </ol>
                        </div>
                      </div>
                    ) : null}

                    {highlight && highlight.value ? (
                      <div className="observation-highlight">
                        <h4>{highlight.label}</h4>
                        <p>{truncate(highlight.value, 240)}</p>
                      </div>
                    ) : null}

                    {planVsExecution ? (
                      <div className="observation-plan">
                        <h4>Plan vs execution</h4>
                        <ul>
                          {planVsExecution.planned.map((toolName) => {
                            const isExecuted = planVsExecution.executed.has(toolName)
                            const lowerReasoning = planVsExecution.reasoning.toLowerCase()
                            const idx = lowerReasoning.indexOf(toolName.toLowerCase())
                            let snippet: string | null = null
                            if (idx !== -1) {
                              const start = Math.max(0, idx - 80)
                              const end = Math.min(planVsExecution.reasoning.length, idx + 200)
                              snippet = planVsExecution.reasoning.slice(start, end).trim()
                            }
                            return (
                              <li
                                key={toolName}
                                className={
                                  isExecuted
                                    ? 'observation-plan__item observation-plan__item--executed'
                                    : 'observation-plan__item observation-plan__item--pending'
                                }
                              >
                                <div className="observation-plan__row">
                                  <span className="observation-plan__tool">{toolName}</span>
                                  <span className="observation-plan__status">
                                    {isExecuted ? '✅ Executed' : '⌛ Planned only'}
                                  </span>
                                </div>
                                {snippet ? (
                                  <div className="observation-plan__snippet-box">
                                    <span>Reasoning snippet</span>
                                    <p>{truncate(snippet, 180)}</p>
                                  </div>
                                ) : null}
                              </li>
                            )
                          })}
                        </ul>
                      </div>
                    ) : null}

                    <dl className="observation-meta">
                      {selectedObservation.name ? (
                        <div>
                          <dt>Name</dt>
                          <dd>{selectedObservation.name}</dd>
                        </div>
                      ) : null}
                      {detailMetadata?.langgraph_node ? (
                        <div>
                          <dt>LangGraph node</dt>
                          <dd>{String(detailMetadata.langgraph_node)}</dd>
                        </div>
                      ) : null}
                      {selectedObservation.endTime ? (
                        <div>
                          <dt>Finished</dt>
                          <dd>{formatDateTime(selectedObservation.endTime)}</dd>
                        </div>
                      ) : null}
                    </dl>

                    <div className="observation-payloads">
                      <details open>
                        <summary>Output / Reasoning</summary>
                        <pre>{formattedOutput}</pre>
                      </details>
                      {selectedReasoning ? (
                        <details>
                          <summary>Full reasoning (model thoughts)</summary>
                          <pre>{selectedReasoning}</pre>
                        </details>
                      ) : null}
                      <details>
                        <summary>Input</summary>
                        <pre>{formattedInput}</pre>
                      </details>
                      <details>
                        <summary>Metadata</summary>
                        <pre>{formattedMetadata}</pre>
                      </details>
                    </div>
                  </div>
                ) : (
                  <div className="observation-detail__empty">Select a node to view its payloads.</div>
                )}
              </aside>
              </div>
            </>
          ) : null}
      </section>

    </div>
  )
}

export default App
