import { useCallback, useEffect, useMemo, useState } from 'react'
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

type GraphNodeData = {
  label: string
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
  const [status, setStatus] = useState<ConnectionStatus>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [traces, setTraces] = useState<LangfuseTrace[]>([])
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null)
  const [traceDetail, setTraceDetail] = useState<LangfuseTraceDetail | null>(null)
  const [detailStatus, setDetailStatus] = useState<DetailStatus>('idle')
  const [detailError, setDetailError] = useState<string | null>(null)
  const [selectedObservationId, setSelectedObservationId] = useState<string | null>(null)
  const [narrativeStatus, setNarrativeStatus] = useState<NarrativeStatus>('idle')
  const [remoteNarrative, setRemoteNarrative] = useState<ExecutionNarrative | null>(null)

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
      const isSelected = selectedObservation ? observation.id === selectedObservation.id : index === 0

      nodes.push({
        id: observation.id,
        position: {
          x: depth * HORIZONTAL_SPACING,
          y: ROOT_OFFSET_Y + index * VERTICAL_SPACING,
        },
        data: {
          label: toNodeLabel(observation, index + 1),
          kind: observation.type,
          step: index + 1,
        },
        draggable: false,
        selectable: true,
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        style: {
          borderRadius: 10,
          padding: 10,
          border: isSelected ? '2px solid #1d4ed8' : '1px solid #cbd5f5',
          background: isSelected ? '#eef2ff' : '#ffffff',
          fontWeight: isSelected ? 600 : 500,
          color: '#0f172a',
        },
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
  }, [traceDetail, orderedObservations, selectedObservation])

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

  const handleProcessTrace = useCallback(() => {
    if (!traceDetail || orderedObservations.length === 0) {
      return
    }
    setNarrativeStatus('processing')

    const payload = {
      trace_id: traceDetail.id,
      trace_name: traceDetail.name ?? null,
      observations: orderedObservations,
    }

    fetch(TRACE_SUMMARY_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
      .then(async (response) => {
        if (!response.ok) {
          const text = await response.text().catch(() => '')
          throw new Error(
            `Trace summary request failed: ${response.status} ${response.statusText} ${text}`,
          )
        }
        return response.json() as Promise<unknown>
      })
      .then((data) => {
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
        setRemoteNarrative(isEmptyNarrative ? null : narrative)
        setNarrativeStatus('ready')
      })
      .catch((error) => {
        console.error('[Observability] Failed to summarise trace', error)
        setRemoteNarrative(null)
        setNarrativeStatus('ready')
      })
  }, [traceDetail, orderedObservations])

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Langfuse Observability</h1>
        <p>
          Browse Langfuse traces on the left and inspect chronological execution flow on the right.
          Click nodes for inputs, outputs, and reasoning.
        </p>
      </header>

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
                selectedTraceId={selectedTraceId}
                loadingTraceId={loadingTraceId}
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

      <section className="app-card graph-card">
          <div className="graph-header">
            <div>
              <h2>{selectedTrace ? selectedTrace.name || 'Trace flow' : 'Trace flow'}</h2>
              <p>{graphSubtitle}</p>
            </div>
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

                    {narrativeStatus === 'processing' ? (
                      <div className="execution-narrative">
                        <h4>Execution narrative</h4>
                        <p>Processing trace…</p>
                      </div>
                    ) : null}

                    {narrativeStatus === 'ready' && effectiveNarrative ? (
                      <div className="execution-narrative">
                        <h4>Execution narrative</h4>
                        <ol>
                          {effectiveNarrative.userQuestion ? (
                            <li>
                              <span className="execution-narrative__label">User asked</span>
                              <p>{truncate(effectiveNarrative.userQuestion, 200)}</p>
                            </li>
                          ) : null}
                          {effectiveNarrative.planningTools.length > 0 ? (
                            <li>
                              <span className="execution-narrative__label">Planned tool calls</span>
                              <ul className="execution-narrative__tools-list">
                                {effectiveNarrative.planningTools.map((tool) => (
                                  <li key={tool} className="execution-narrative__tool-chip">
                                    {tool}
                                  </li>
                                ))}
                              </ul>
                            </li>
                          ) : null}
                          {effectiveNarrative.toolsExecuted.length > 0 ? (
                            <li>
                              <span className="execution-narrative__label">Tools executed</span>
                              <ul className="execution-narrative__tools-list execution-narrative__tools-list--stacked">
                                {effectiveNarrative.toolsExecuted.map((item) => (
                                  <li key={item.name}>
                                    <div className="execution-narrative__tool-chip">{item.name}</div>
                                    <p className="execution-narrative__tool-summary">
                                      {truncate(item.summary, 200)}
                                    </p>
                                  </li>
                                ))}
                              </ul>
                            </li>
                          ) : null}
                          {effectiveNarrative.finalAnswer ? (
                            <li>
                              <span className="execution-narrative__label">Model concluded</span>
                              <p>{truncate(effectiveNarrative.finalAnswer, 260)}</p>
                            </li>
                          ) : null}
                        </ol>
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
                                    {isExecuted ? 'Executed' : 'Planned only'}
                                  </span>
                                </div>
                                {snippet ? (
                                  <p className="observation-plan__snippet">{truncate(snippet, 180)}</p>
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
          ) : null}
      </section>

      <footer className="app-footer">
        <small>
          Credentials are read from <code>VITE_LANGFUSE_*</code> environment variables. Avoid
          shipping production secrets to the browser—use this toolkit for local analysis or proxy the
          requests through a backend.
        </small>
      </footer>
    </div>
  )
}

export default App
