import type { LangfuseTrace } from '../lib/langfuse'
import './TraceTable.css'

interface TraceTableProps {
  traces: LangfuseTrace[]
  onSelect?(trace: LangfuseTrace): void
  selectedTraceId?: string | null
  loadingTraceId?: string | null
}

function formatTimestamp(timestamp: string): string {
  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: 'medium',
      timeStyle: 'medium',
    }).format(new Date(timestamp))
  } catch (error) {
    console.warn('Unable to format timestamp', { timestamp, error })
    return timestamp
  }
}

function formatLatency(latency: number | null): string {
  if (latency === null) {
    return '—'
  }
  const rounded = Math.round(latency * 1000) / 1000
  return `${rounded.toLocaleString()} s`
}

export default function TraceTable({
  traces,
  onSelect,
  selectedTraceId,
  loadingTraceId,
}: TraceTableProps) {
  return (
    <div className="trace-table-wrapper">
      <table className="trace-table">
        <thead>
          <tr>
            <th scope="col">Name</th>
            <th scope="col">Environment</th>
            <th scope="col">Timestamp</th>
            <th scope="col">Latency</th>
            <th scope="col">Tags</th>
            <th scope="col" className="actions-column">
              Details
            </th>
          </tr>
        </thead>
        <tbody>
          {traces.map((trace) => {
            const isSelected = trace.id === selectedTraceId
            const isLoading = loadingTraceId === trace.id

            return (
              <tr key={trace.id} className={isSelected ? 'is-selected' : ''}>
                <th scope="row">{trace.name || trace.id}</th>
                <td>{trace.environment}</td>
                <td>{formatTimestamp(trace.timestamp)}</td>
                <td>{formatLatency(trace.latency)}</td>
                <td>
                  {trace.tags.length > 0 ? (
                    <ul className="tags-list">
                      {trace.tags.map((tag) => (
                        <li key={tag}>{tag}</li>
                      ))}
                    </ul>
                  ) : (
                    '—'
                  )}
                </td>
                <td className="actions-column">
                  {onSelect ? (
                    <button
                      type="button"
                      className={`trace-action ${isSelected ? 'is-active' : ''}`}
                      onClick={() => onSelect(trace)}
                      disabled={isLoading}
                    >
                      {isLoading ? 'Loading…' : isSelected ? 'Viewing flow' : 'Visualize flow'}
                    </button>
                  ) : null}
                  <a
                    href={trace.url}
                    target="_blank"
                    rel="noreferrer"
                    className="trace-action trace-action--external"
                  >
                    Open in Langfuse
                  </a>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
