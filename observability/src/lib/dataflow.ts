export type DataSensitivity = 'none' | 'low' | 'medium' | 'high'

export type DataArtefactKind =
  | 'user_message'
  | 'retrieved_chunk'
  | 'sql_rowset'
  | 'http_response'
  | 'file_contents'
  | 'tool_output'
  | 'model_output'

export interface DataArtefactDefinition {
  kind: DataArtefactKind
  label: string
  defaultSensitivity: DataSensitivity
  defaultTags: string[]
  idFieldPaths?: string[]
  contentFieldPaths?: string[]
}

export type ToolType = 'retrieval' | 'sql_db' | 'http_client' | 'file_search' | 'math' | 'generic'

export interface ToolTypeDefinition {
  toolType: ToolType
  label: string
  description: string
  matchNames: {
    exact?: string[]
    contains?: string[]
    prefix?: string[]
  }
  artefact: DataArtefactDefinition
  metadataTagKeys: string[]
}

export interface ArtefactInstance {
  id: string
  kind: DataArtefactKind
  sourceToolType: ToolType
  sourceObservationId?: string
  sensitivity: DataSensitivity
  tags?: string[]
}

export interface StepVisibleData {
  observationId: string
  visibleArtefactIds: string[]
}

const RETRIEVAL_ARTEFACT: DataArtefactDefinition = {
  kind: 'retrieved_chunk',
  label: 'Retrieved document chunk',
  defaultSensitivity: 'medium',
  defaultTags: ['retrieval', 'knowledge'],
  idFieldPaths: ['document_id', 'id', 'metadata.id'],
  contentFieldPaths: ['page_content', 'content', 'text'],
}

const SQL_ARTEFACT: DataArtefactDefinition = {
  kind: 'sql_rowset',
  label: 'SQL result set',
  defaultSensitivity: 'high',
  defaultTags: ['sql', 'database'],
  idFieldPaths: ['rows.*.id', 'rows.*.primary_key'],
  contentFieldPaths: ['rows'],
}

const HTTP_ARTEFACT: DataArtefactDefinition = {
  kind: 'http_response',
  label: 'HTTP response payload',
  defaultSensitivity: 'medium',
  defaultTags: ['http', 'web'],
  idFieldPaths: ['url', 'request.url'],
  contentFieldPaths: ['body', 'data'],
}

const FILE_ARTEFACT: DataArtefactDefinition = {
  kind: 'file_contents',
  label: 'File contents',
  defaultSensitivity: 'medium',
  defaultTags: ['file'],
  idFieldPaths: ['path', 'file_path'],
  contentFieldPaths: ['contents', 'text'],
}

const MATH_ARTEFACT: DataArtefactDefinition = {
  kind: 'tool_output',
  label: 'Math/tool result',
  defaultSensitivity: 'low',
  defaultTags: ['math', 'computed'],
  contentFieldPaths: ['result', 'value'],
}

const GENERIC_ARTEFACT: DataArtefactDefinition = {
  kind: 'tool_output',
  label: 'Tool output',
  defaultSensitivity: 'low',
  defaultTags: ['tool'],
  contentFieldPaths: ['output', 'result'],
}

export const TOOL_TYPE_DEFINITIONS: ToolTypeDefinition[] = [
  {
    toolType: 'retrieval',
    label: 'Retrieval / RAG',
    description: 'Tools that fetch knowledge base or vector store content.',
    matchNames: {
      contains: ['retriever', 'search', 'rag', 'resource'],
    },
    artefact: RETRIEVAL_ARTEFACT,
    metadataTagKeys: ['data_source', 'collection', 'index_name', 'sensitivity', 'contains_pii'],
  },
  {
    toolType: 'sql_db',
    label: 'SQL / Database',
    description: 'Tools that execute SQL queries or read from databases.',
    matchNames: {
      contains: ['sql', 'database', 'db', 'postgres', 'mysql'],
    },
    artefact: SQL_ARTEFACT,
    metadataTagKeys: ['database', 'schema', 'table', 'columns', 'tenant_id', 'sensitivity'],
  },
  {
    toolType: 'http_client',
    label: 'HTTP client / API',
    description: 'Tools that call external HTTP APIs or web endpoints.',
    matchNames: {
      contains: ['http', 'request', 'fetch', 'web'],
    },
    artefact: HTTP_ARTEFACT,
    metadataTagKeys: ['url', 'service', 'sensitivity', 'contains_pii'],
  },
  {
    toolType: 'file_search',
    label: 'File search / file system',
    description: 'Tools that read from the local file system or document stores.',
    matchNames: {
      contains: ['file', 'filesystem', 'fs'],
    },
    artefact: FILE_ARTEFACT,
    metadataTagKeys: ['path', 'collection', 'sensitivity', 'contains_pii'],
  },
  {
    toolType: 'math',
    label: 'Math / deterministic utilities',
    description: 'Pure functions that compute results without accessing external data.',
    matchNames: {
      contains: ['math', 'calculator'],
    },
    artefact: MATH_ARTEFACT,
    metadataTagKeys: [],
  },
  {
    toolType: 'generic',
    label: 'Generic tool',
    description: 'Other tools that do not fit a more specific category.',
    matchNames: {
      contains: [],
    },
    artefact: GENERIC_ARTEFACT,
    metadataTagKeys: ['sensitivity', 'contains_pii'],
  },
]

export function inferToolTypeByName(toolName: string | null | undefined): ToolTypeDefinition {
  const normalized = (toolName ?? '').toLowerCase().trim()

  const generic =
    TOOL_TYPE_DEFINITIONS.find((definition) => definition.toolType === 'generic') ??
    TOOL_TYPE_DEFINITIONS[TOOL_TYPE_DEFINITIONS.length - 1]

  if (!normalized) {
    return generic
  }

  const candidates = TOOL_TYPE_DEFINITIONS.filter((definition) => definition.toolType !== 'generic')

  for (const definition of candidates) {
    const { exact = [], prefix = [], contains = [] } = definition.matchNames

    if (exact.some((name) => name.toLowerCase() === normalized)) {
      return definition
    }

    if (prefix.some((value) => normalized.startsWith(value.toLowerCase()))) {
      return definition
    }

    if (contains.some((value) => normalized.includes(value.toLowerCase()))) {
      return definition
    }
  }

  return generic
}
