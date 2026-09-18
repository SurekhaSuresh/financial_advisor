# Local Data Inspection

Financial Advisor includes a read-only development inspector for examining
persisted workflow state and the curated LanceDB knowledge store.

## SQLite session store

| Endpoint | Purpose |
| --- | --- |
| `GET /inspection/sqlite/tables` | Returns the approved SQLite tables and row counts. |
| `GET /inspection/sqlite/tables/{table_name}/rows` | Returns up to 100 raw rows from one approved table. |

The API does not accept arbitrary SQL or table names. The repository enforces a
fixed table allowlist, and the UI uses it to browse sessions, messages, research
plans, retrieval traces, briefs, recommendations, reviews, and ordered events.

## LanceDB knowledge store

| Endpoint | Purpose |
| --- | --- |
| `GET /inspection/knowledge/summary` | Returns chunk count plus publisher/topic distributions. |
| `GET /inspection/knowledge/chunks` | Returns one source-attributed readable chunk page, with optional metadata, publisher, topic, and token-window filters. |

The chunk inspector returns source title, publisher, URL, heading path, topics,
token count, and text. It accepts an `offset` and bounded `limit` so the UI can
page through the complete filtered result set in groups of 50. `min_token_count`
and `max_token_count` form an optional inclusive token-count window (for
example, at least 500 tokens, or 300–500 tokens). A missing minimum means 0;
a missing maximum means no upper bound.
It intentionally omits embedding vectors: vectors are
retrieval implementation data, not useful for a human inspection screen, and
would make each response unnecessarily large.

## UI use

Select **Observability** in the React client. It provides:

- **Trace Explorer**, where a session ID or trace ID loads the complete saved
  trajectory for replay;
- **SQLite Store**, a table-count and bounded raw-row browser for the actual
  local session database;
- **Knowledge Store**, a paginated LanceDB chunk browser with publisher, topic,
  metadata, and inclusive minimum/maximum-token filters;
- a clear separation between trusted persisted data and transient model state.

This screen is local development observability. It has no write operations and
cannot change a session, recommendation, knowledge chunk, or retrieval result.
