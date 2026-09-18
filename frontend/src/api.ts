import type {
  KnowledgeChunkPage,
  KnowledgeSummary,
  ScenarioStarted,
  SessionSnapshot,
  SessionSummary,
  SqliteTableRows,
  SqliteTableSummary,
} from "./types";

const API_ROOT = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, init);
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Request failed with status ${response.status}.`);
  }
  return (await response.json()) as T;
}

export function listSessions(): Promise<SessionSummary[]> {
  return request<SessionSummary[]>("/sessions");
}

export function getSession(sessionId: string): Promise<SessionSnapshot> {
  return request<SessionSnapshot>(`/sessions/${sessionId}`);
}

export function getTrace(traceId: string): Promise<SessionSnapshot> {
  return request<SessionSnapshot>(`/traces/${traceId}`);
}

export function startScenario(profile: SessionSnapshot["client_profile"]): Promise<ScenarioStarted> {
  return request<ScenarioStarted>("/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });
}

export function traceEventStreamUrl(traceId: string, afterSequence: number): string {
  return `${API_ROOT}/traces/${traceId}/events/stream?after_sequence=${afterSequence}`;
}

export function listSqliteTables(): Promise<SqliteTableSummary[]> {
  return request<SqliteTableSummary[]>("/inspection/sqlite/tables");
}

export function getSqliteTableRows(tableName: string): Promise<SqliteTableRows> {
  return request<SqliteTableRows>(`/inspection/sqlite/tables/${tableName}/rows`);
}

export function getKnowledgeSummary(): Promise<KnowledgeSummary> {
  return request<KnowledgeSummary>("/inspection/knowledge/summary");
}

export function getKnowledgeChunks(filters: {
  publisher?: string;
  topic?: string;
  sourceId?: string;
  chunkId?: string;
  metadataField?: string;
  metadataValue?: string;
  minTokenCount?: number;
  maxTokenCount?: number;
  offset?: number;
}): Promise<KnowledgeChunkPage> {
  const params = new URLSearchParams();
  if (filters.publisher) params.set("publisher", filters.publisher);
  if (filters.topic) params.set("topic", filters.topic);
  if (filters.sourceId) params.set("source_id", filters.sourceId);
  if (filters.chunkId) params.set("chunk_id", filters.chunkId);
  if (filters.metadataField && filters.metadataValue) {
    params.set("metadata_field", filters.metadataField);
    params.set("metadata_value", filters.metadataValue);
  }
  if (filters.minTokenCount !== undefined) params.set("min_token_count", String(filters.minTokenCount));
  if (filters.maxTokenCount !== undefined) params.set("max_token_count", String(filters.maxTokenCount));
  params.set("offset", String(filters.offset ?? 0));
  params.set("limit", "50");
  return request<KnowledgeChunkPage>(`/inspection/knowledge/chunks?${params.toString()}`);
}
