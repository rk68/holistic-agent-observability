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
    for (const item of messages) {
      if (!item || typeof item !== 'object' || !('content' in item)) {
        continue
      }
      const content = (item as { content?: unknown }).content
      if (typeof content === 'string') {
        return content
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

  if (typeof output === 'string') {
    return output
  }

  return null
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
    const depthCache = new Map<string, number>()

    const getDepth = (parentId: string | null): number => {
      if (!parentId) {
        return 1
      }
      const cached = depthCache.get(parentId)
      if (cached !== undefined) {
        return cached
      }
      const parent = observationMap.get(parentId)
      const depth = parent ? getDepth(parent.parentObservationId) + 1 : 1
      depthCache.set(parentId, depth)
      return depth
    }

    orderedObservations.forEach((observation, index) => {
      const depth = getDepth(observation.parentObservationId)
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

    const edges: GraphEdge[] = orderedObservations.map((observation) => {
      const source = observation.parentObservationId ?? traceDetail.id
      return {
        id: `${source}→${observation.id}`,
        source,
        target: observation.id,
        animated: observation.type === 'TOOL',
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
              <a
                className="secondary-button"
                href={selectedTrace.url}
                target="_blank"
                rel="noreferrer"
              >
                Open in Langfuse
              </a>
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

                    {highlight && highlight.value ? (
                      <div className="observation-highlight">
                        <h4>{highlight.label}</h4>
                        <p>{truncate(highlight.value, 240)}</p>
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
