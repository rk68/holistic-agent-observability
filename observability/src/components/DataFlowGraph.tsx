import { useMemo } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  Position,
  type Edge as ReactFlowEdge,
  type Node as ReactFlowNode,
} from 'reactflow'

import type { LangfuseObservation, LangfuseTraceDetail } from '../lib/langfuse'
import type {
  ArtefactInstance,
  StepVisibleData,
} from '../lib/dataflow'
import type { FailureSummary, ObservationFailureSummary } from '../lib/failureAnalysis'

type Stage = 'user' | 'retrieval' | 'sql' | 'tool' | 'llm' | 'agent' | 'output' | 'other'

type RiskLevel = 'none' | 'low' | 'medium' | 'high'

type LeakDetection = {
  level: RiskLevel
  sources: string[]
}

interface ObservationDataSummary {
  observationId: string
  stage: Stage
  sensitivityLevel: RiskLevel
  dataClasses: string[]
  isSource: boolean
  isSink: boolean
}

interface DataFlowJudgement {
  id: string
  sourceObservationId: string
  targetObservationId: string
  risk: RiskLevel
  dataClasses: string[]
  explanation: string
}

interface DataFlowGraphProps {
  trace: LangfuseTraceDetail
  observations: LangfuseObservation[]
  summaries?: ObservationDataSummary[]
  judgements?: DataFlowJudgement[]
  artefacts?: ArtefactInstance[]
  visibility?: StepVisibleData[]
  failures?: FailureSummary
  selectedObservationId?: string
  onSelectObservation?: (observationId: string) => void
}

type GraphNodeData = {
  label: string
  stage: Stage
  sensitivityLevel: RiskLevel
  visibleArtefactIds?: string[]
  highRiskVisibleCount?: number
  leakRiskLevel?: RiskLevel
  leakSources?: string[]
}

type GraphNode = ReactFlowNode<GraphNodeData>
type GraphEdge = ReactFlowEdge

const ROOT_OFFSET_Y = 60
const VERTICAL_SPACING = 110
const HORIZONTAL_SPACING = 260

const STAGE_ORDER: Stage[] = ['user', 'retrieval', 'sql', 'tool', 'llm', 'agent', 'output', 'other']

const LEAK_PATTERNS: { pattern: RegExp; source: string; level: RiskLevel }[] = [
  { pattern: /\b\d{3}-\d{2}-\d{4}\b/g, source: 'ssn', level: 'high' },
  { pattern: /\b(?:\d[ -]?){13,16}\b/g, source: 'credit_card', level: 'high' },
  { pattern: /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, source: 'email_address', level: 'medium' },
  {
    pattern: /\b(?:\+?\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}\b/g,
    source: 'phone_number',
    level: 'medium',
  },
  { pattern: /\b(account number|routing number|iban|iban number)\b/gi, source: 'account_identifier', level: 'low' },
]

const LEAK_STAGES: Stage[] = ['llm', 'agent', 'output']

function maxRisk(current: RiskLevel, next: RiskLevel): RiskLevel {
  const ordering: Record<RiskLevel, number> = { none: 0, low: 1, medium: 2, high: 3 }
  return ordering[next] > ordering[current] ? next : current
}

function coerceText(value: unknown): string {
  if (value == null) {
    return ''
  }
  if (typeof value === 'string') {
    return value
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value)
    } catch (error) {
      console.warn('[Observability] Failed to stringify value during leak detection', error)
      return ''
    }
  }
  return ''
}

function detectLeakFromObservation(
  observation: LangfuseObservation,
  stage: Stage,
  visibleArtefactIds: string[],
  artefactsById: Map<string, ArtefactInstance>,
): LeakDetection {
  const metadata = observation.metadata ?? {}
  const judge = (metadata as { leak_judge?: { risk?: RiskLevel; sources?: string[] } }).leak_judge

  const candidateTexts: string[] = []
  if (observation.output !== null && observation.output !== undefined) {
    candidateTexts.push(coerceText(observation.output))
  }

  const finalAnswer = (metadata as { final_answer?: unknown }).final_answer
  if (finalAnswer !== undefined) {
    candidateTexts.push(coerceText(finalAnswer))
  }

  if (candidateTexts.length === 0 && !judge) {
    return { level: 'none', sources: [] }
  }

  let level: RiskLevel = 'none'
  const sources = new Set<string>()

  const shouldInspectText = LEAK_STAGES.includes(stage) || observation.type === 'GENERATION'
  if (shouldInspectText) {
    const combinedText = candidateTexts.join('\n')
    if (combinedText) {
      for (const descriptor of LEAK_PATTERNS) {
        descriptor.pattern.lastIndex = 0
        if (descriptor.pattern.test(combinedText)) {
          level = maxRisk(level, descriptor.level)
          sources.add(descriptor.source)
        }
      }
    }
  }

  if (judge?.risk) {
    level = maxRisk(level, judge.risk)
    ;(judge.sources ?? []).forEach((source) => {
      if (source) {
        sources.add(source)
      }
    })
  }

  if (level !== 'none' && visibleArtefactIds.length > 0) {
    visibleArtefactIds.forEach((artefactId) => {
      const artefact = artefactsById.get(artefactId)
      if (artefact) {
        sources.add(artefact.id)
        if (artefact.sensitivity === 'high') {
          level = maxRisk(level, 'high')
        } else if (artefact.sensitivity === 'medium') {
          level = maxRisk(level, 'medium')
        }
      } else {
        sources.add(artefactId)
      }
    })
  }

  return { level, sources: Array.from(sources) }
}

function inferStage(observation: LangfuseObservation): Stage {
  const type = observation.type
  const name = (observation.name || '').toLowerCase()
  const metadata = observation.metadata ?? {}

  if (type === 'SPAN') {
    const fromUser = (metadata as { role?: string }).role === 'user'
    return fromUser ? 'user' : 'agent'
  }

  if (type === 'TOOL') {
    if (name.includes('sql')) return 'sql'
    if (name.includes('db')) return 'sql'
    if (name.includes('retriev') || name.includes('search')) return 'retrieval'
    return 'tool'
  }

  if (type === 'GENERATION') {
    return 'llm'
  }

  if (type === 'AGENT') {
    return 'agent'
  }

  return 'other'
}

function buildDefaultSummaries(observations: LangfuseObservation[]): ObservationDataSummary[] {
  return observations.map((observation, index) => {
    const stage = inferStage(observation)
    const isFirst = index === 0
    const isLast = index === observations.length - 1
    const isSource = stage === 'user' || stage === 'retrieval' || stage === 'sql' || isFirst
    const isSink = stage === 'output' || stage === 'llm' || isLast

    return {
      observationId: observation.id,
      stage,
      sensitivityLevel: 'none',
      dataClasses: [],
      isSource,
      isSink,
    }
  })
}

function buildVisibilityFromMetadata(observations: LangfuseObservation[]): StepVisibleData[] {
  const steps: StepVisibleData[] = []
  observations.forEach((observation) => {
    const metadata = observation.metadata ?? {}
    const raw = (metadata as { visible_data?: unknown }).visible_data

    if (!raw) {
      return
    }

    const visibleArtefactIds: string[] = []

    if (Array.isArray(raw)) {
      raw.forEach((value) => {
        if (typeof value === 'string' && value) {
          visibleArtefactIds.push(value)
        }
      })
    }

    if (visibleArtefactIds.length > 0) {
      steps.push({ observationId: observation.id, visibleArtefactIds })
    }
  })

  return steps
}

function riskColor(risk: RiskLevel): string {
  switch (risk) {
    case 'high':
      return '#dc2626'
    case 'medium':
      return '#ea580c'
    case 'low':
      return '#16a34a'
    default:
      return '#64748b'
  }
}

function stageColor(stage: Stage): string {
  switch (stage) {
    case 'user':
      return '#e0f2fe'
    case 'retrieval':
      return '#fef9c3'
    case 'sql':
      return '#fee2e2'
    case 'tool':
      return '#f3e8ff'
    case 'llm':
      return '#e0f2f1'
    case 'agent':
      return '#ede9fe'
    case 'output':
      return '#cffafe'
    default:
      return '#f9fafb'
  }
}

function DataFlowGraph({
  trace,
  observations,
  summaries: summariesProp,
  judgements: judgementsProp,
  artefacts: artefactsProp,
  visibility: visibilityProp,
  failures,
  selectedObservationId,
  onSelectObservation,
}: DataFlowGraphProps) {
  const summaries = useMemo(
    () => summariesProp ?? buildDefaultSummaries(observations),
    [observations, summariesProp],
  )

  const summariesById = useMemo(() => {
    const map = new Map<string, ObservationDataSummary>()
    summaries.forEach((summary) => {
      map.set(summary.observationId, summary)
    })
    return map
  }, [summaries])

  const stageIndex = useMemo(() => {
    const index = new Map<Stage, number>()
    STAGE_ORDER.forEach((stage, idx) => {
      index.set(stage, idx)
    })
    return index
  }, [])

  const judgements = judgementsProp ?? []

  const artefactsById = useMemo(() => {
    const map = new Map<string, ArtefactInstance>()
    ;(artefactsProp ?? []).forEach((artefact) => {
      map.set(artefact.id, artefact)
    })
    return map
  }, [artefactsProp])

  const visibilityByObservationId = useMemo(() => {
    const visibility = visibilityProp ?? buildVisibilityFromMetadata(observations)
    const map = new Map<string, StepVisibleData>()
    visibility.forEach((step) => {
      map.set(step.observationId, step)
    })
    return map
  }, [observations, visibilityProp])

  const failuresByObservationId = useMemo(() => {
    const map = new Map<string, ObservationFailureSummary>()
    if (!failures) {
      return map
    }
    const raw = failures.per_observation_failures ?? {}
    Object.entries(raw).forEach(([observationId, entry]) => {
      if (!entry || typeof entry !== 'object') {
        return
      }
      const maxSeverity = (entry as { max_severity?: string }).max_severity
      const codes = (entry as { codes?: unknown }).codes
      const normalizedSeverity =
        maxSeverity === 'HIGH' || maxSeverity === 'MEDIUM' || maxSeverity === 'LOW'
          ? maxSeverity
          : 'LOW'
      const normalizedCodes = Array.isArray(codes)
        ? codes.filter((code): code is string => typeof code === 'string')
        : []
      map.set(observationId, { max_severity: normalizedSeverity, codes: normalizedCodes })
    })
    return map
  }, [failures])

  const graph = useMemo(() => {
    if (!trace || observations.length === 0) {
      return { nodes: [] as GraphNode[], edges: [] as GraphEdge[] }
    }

    const nodes: GraphNode[] = [
      {
        id: trace.id,
        data: {
          label: trace.name || 'Trace',
          stage: 'other',
          sensitivityLevel: 'none',
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
          background: '#0f172a',
          color: '#ffffff',
        },
      },
    ]

    observations.forEach((observation, index) => {
      const summary = summariesById.get(observation.id)
      const stage = summary?.stage ?? inferStage(observation)
      const stageIdx = stageIndex.get(stage) ?? stageIndex.get('other') ?? STAGE_ORDER.length - 1

      const labelParts: string[] = [`#${index + 1}`, stage]
      if (observation.name) {
        labelParts.push(`• ${observation.name}`)
      }

      const visibility = visibilityByObservationId.get(observation.id)
      const visibleArtefactIds = visibility?.visibleArtefactIds ?? []

      const highRiskVisibleCount = visibleArtefactIds.reduce((count, artefactId) => {
        const artefact = artefactsById.get(artefactId)
        if (artefact && artefact.sensitivity === 'high') {
          return count + 1
        }
        return count
      }, 0)

      if (visibleArtefactIds.length > 0) {
        labelParts.push(`• ${visibleArtefactIds.length} artefact${visibleArtefactIds.length === 1 ? '' : 's'} visible`)
      }

      const leakDetection = detectLeakFromObservation(observation, stage, visibleArtefactIds, artefactsById)
      if (leakDetection.level !== 'none') {
        labelParts.push(`• leak risk: ${leakDetection.level.toUpperCase()}`)
      }

      const failure = failuresByObservationId.get(observation.id)
      if (failure && failure.codes.length > 0) {
        labelParts.push(`• failures: ${failure.codes.join(', ')}`)
      }

      const isSelected = selectedObservationId === observation.id

      let borderColor = '#cbd5f5'
      // Failure severity takes precedence for border colour.
      if (failure) {
        if (failure.max_severity === 'HIGH') {
          borderColor = '#dc2626'
        } else if (failure.max_severity === 'MEDIUM') {
          borderColor = '#ea580c'
        } else if (failure.max_severity === 'LOW') {
          borderColor = '#16a34a'
        }
      } else if (leakDetection.level === 'high') {
        borderColor = '#dc2626'
      } else if (leakDetection.level === 'medium') {
        borderColor = '#ea580c'
      }

      nodes.push({
        id: observation.id,
        position: {
          // Flow left-to-right by step index, grouped vertically by stage lane.
          x: (index + 1) * HORIZONTAL_SPACING,
          y: ROOT_OFFSET_Y + stageIdx * VERTICAL_SPACING,
        },
        data: {
          label: labelParts.join(' '),
          stage,
          sensitivityLevel: summary?.sensitivityLevel ?? 'none',
          visibleArtefactIds,
          highRiskVisibleCount,
          leakRiskLevel: leakDetection.level,
          leakSources: leakDetection.sources,
        },
        draggable: false,
        selectable: true,
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        style: {
          borderRadius: 10,
          padding: 10,
          background: stageColor(stage),
          border: isSelected ? `2px solid ${borderColor}` : `1px solid ${borderColor}`,
          boxShadow:
            highRiskVisibleCount > 0
              ? '0 0 0 2px rgba(220, 38, 38, 0.4)'
              : undefined,
          color: '#0f172a',
          fontWeight: 500,
        },
      })
    })

    const controlEdges: GraphEdge[] = observations.map((observation, index) => {
      let source = observation.parentObservationId ?? null
      if (!source) {
        source = index > 0 ? observations[index - 1]?.id ?? trace.id : trace.id
      }

      return {
        id: `control-${source}→${observation.id}`,
        source,
        target: observation.id,
        animated: false,
        style: {
          stroke: '#cbd5f5',
          strokeWidth: 1.5,
        },
      }
    })

    const dataEdges: GraphEdge[] = judgements.map((judgement) => ({
      id: `data-${judgement.id}`,
      source: judgement.sourceObservationId,
      target: judgement.targetObservationId,
      animated: judgement.risk === 'high',
      label: judgement.dataClasses.length > 0 ? judgement.dataClasses.join(', ') : undefined,
      style: {
        stroke: riskColor(judgement.risk),
        strokeWidth: judgement.risk === 'high' ? 3 : 2,
      },
      labelBgStyle: {
        fill: '#0f172a',
        color: '#f9fafb',
      },
    }))

    return { nodes, edges: [...controlEdges, ...dataEdges] }
  }, [
    trace,
    observations,
    summariesById,
    stageIndex,
    judgements,
    artefactsById,
    visibilityByObservationId,
    failuresByObservationId,
    selectedObservationId,
  ])

  return (
    <ReactFlowProvider>
      <ReactFlow
        nodes={graph.nodes}
        edges={graph.edges}
        style={{ width: '100%', height: '100%' }}
        defaultViewport={{ x: 0, y: 0, zoom: 1 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        panOnScroll
        zoomOnScroll={false}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, node) => {
          if (node.id === trace.id) {
            return
          }
          onSelectObservation?.(node.id)
        }}
        onInit={(instance) => {
          window.requestAnimationFrame(() => {
            instance.fitView({ padding: 0.25 })
          })
        }}
      >
        <Background gap={24} />
        <MiniMap zoomable pannable />
        <Controls showInteractive={false} position="bottom-right" />
      </ReactFlow>
    </ReactFlowProvider>
  )
}

export type { ObservationDataSummary, DataFlowJudgement, RiskLevel, Stage }

export default DataFlowGraph
