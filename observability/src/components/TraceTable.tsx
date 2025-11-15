import { useMemo, useState } from 'react'
import type { LangfuseTrace } from '../lib/langfuse'
import './TraceTable.css'

interface TraceTableProps {
  traces: LangfuseTrace[]
  onSelect?(trace: LangfuseTrace): void
  selectedTraceId?: string | null
  loadingTraceId?: string | null
}

const PAGE_SIZE = 10

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
  const [page, setPage] = useState(0)

  const { pageTraces, totalPages, currentPage, from, to } = useMemo(() => {
    const total = traces.length
    const size = PAGE_SIZE
    const pages = Math.max(1, Math.ceil(total / size))
    const clampedPage = Math.min(page, pages - 1)
    const start = clampedPage * size
    const end = Math.min(start + size, total)

    return {
      pageTraces: traces.slice(start, end),
      totalPages: pages,
      currentPage: clampedPage,
      from: total === 0 ? 0 : start + 1,
      to: end,
    }
  }, [traces, page])

  const canPrevious = currentPage > 0
  const canNext = currentPage < totalPages - 1

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
          {pageTraces.map((trace) => {
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
      {totalPages > 1 ? (
        <div className="trace-table-pagination">
          <button
            type="button"
            className="pagination-button"
            onClick={() => canPrevious && setPage((p) => Math.max(0, p - 1))}
            disabled={!canPrevious}
            aria-label="Previous page"
          >
            ‹
          </button>
          <span className="pagination-info">
            Page {currentPage + 1} of {totalPages} • Showing {from}-{to}
          </span>
          <button
            type="button"
            className="pagination-button"
            onClick={() => canNext && setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={!canNext}
            aria-label="Next page"
          >
            ›
          </button>
        </div>
      ) : null}
    </div>
  )
}
