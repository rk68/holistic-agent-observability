import type { NodeProps } from 'reactflow'

export type TimelineNodeData = {
  step: number
  title: string
  subtitle?: string
  isActive: boolean
}

export default function TimelineNode({ data }: NodeProps<TimelineNodeData>) {
  return (
    <div className={`timeline-node${data.isActive ? ' is-selected' : ''}`}>
      <div className="timeline-node__step">Step {data.step}</div>
      <div className="timeline-node__title">{data.title}</div>
      {data.subtitle ? <div className="timeline-node__subtitle">{data.subtitle}</div> : null}
    </div>
  )
}
