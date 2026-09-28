import type {
  KnowledgeStoreInspection,
  SessionSnapshot,
  SessionSummary,
  SqliteTableRows,
} from "./types";

const API_ROOT = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, init);
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(
      payload?.detail ?? `Request failed with status ${response.status}.`,
    );
  }
  return (await response.json()) as T;
}

export function listSessions(): Promise<SessionSummary[]> {
  return request<SessionSummary[]>("/sessions");
}

export function getSession(sessionId: string): Promise<SessionSnapshot> {
  return request<SessionSnapshot>(`/sessions/${sessionId}`);
}

export function startConversation(): Promise<string> {
  return request<string>("/sessions", {
    method: "POST",
  });
}

export function sessionEventStreamUrl(sessionId: string): string {
  return `${API_ROOT}/sessions/${sessionId}/events/stream`;
}

export function listSqliteTables(): Promise<string[]> {
  return request<string[]>("/inspection/sqlite/tables");
}

export function getSqliteTableRows(
  tableName: string,
): Promise<SqliteTableRows> {
  return request<SqliteTableRows>(
    `/inspection/sqlite/tables/${tableName}/rows`,
  );
}

export function getKnowledgeStore(filters: {
  publisher?: string;
  topic?: string;
  metadataField?: string;
  metadataValue?: string;
  minTokenCount?: number;
  maxTokenCount?: number;
  offset?: number;
}): Promise<KnowledgeStoreInspection> {
  const params = new URLSearchParams();
  if (filters.publisher) params.set("publisher", filters.publisher);
  if (filters.topic) params.set("topic", filters.topic);
  if (filters.metadataField && filters.metadataValue) {
    params.set("metadata_field", filters.metadataField);
    params.set("metadata_value", filters.metadataValue);
  }
  if (filters.minTokenCount !== undefined) {
    params.set("min_token_count", String(filters.minTokenCount));
  }
  if (filters.maxTokenCount !== undefined) {
    params.set("max_token_count", String(filters.maxTokenCount));
  }
  params.set("offset", String(filters.offset ?? 0));
  params.set("limit", "50");
  return request<KnowledgeStoreInspection>(
    `/inspection/knowledge?${params.toString()}`,
  );
}
