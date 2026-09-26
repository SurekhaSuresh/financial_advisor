import { useEffect, useRef, useState } from "react";

import {
  getKnowledgeChunks,
  getKnowledgeSummary,
  getSession,
  getSqliteTableRows,
  listSessions,
  listSqliteTables,
  sessionEventStreamUrl,
  startConversation,
} from "./api";
import type {
  CitedText,
  ClientProfile,
  KnowledgeChunkPage,
  KnowledgeSummary,
  Recommendation,
  SessionSnapshot,
  SessionSummary,
  SqliteTableRows,
  SqliteTableSummary,
} from "./types";

const MAYA_PROFILE: ClientProfile = {
  name: "Maya Chen",
  age: 38,
  risk_tolerance: "moderate",
  emergency_fund_months: 6,
  retirement_savings: "120000.00",
  brokerage_savings: "35000.00",
  student_loan_balance: "18000.00",
  student_loan_rate_percent: "5.8",
  primary_goal: "Buy a home",
  goal_time_horizon_years: 5,
};

const INTERNAL_EVIDENCE_ID_LIST =
  /\s*\[(?:[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,})(?:\s*,\s*[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,})*\]/g;

function money(value: string): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(Number(value));
}

function statusLabel(status: string): string {
  return status.replaceAll("_", " ");
}

function formatSessionUpdatedAt(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function clientVisibleClaim(text: string): string {
  return text.replace(INTERNAL_EVIDENCE_ID_LIST, "").trim();
}

async function copyText(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
    return;
  } catch {
    // Some browser privacy settings deny the Clipboard API even on localhost.
    const fallback = document.createElement("textarea");
    fallback.value = text;
    fallback.setAttribute("readonly", "");
    fallback.style.position = "fixed";
    fallback.style.opacity = "0";
    document.body.appendChild(fallback);
    fallback.select();
    const copied = document.execCommand("copy");
    document.body.removeChild(fallback);
    if (!copied) throw new Error("The browser denied clipboard access.");
  }
}

export function App() {
  const [history, setHistory] = useState<SessionSummary[]>([]);
  const [session, setSession] = useState<SessionSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [view, setView] = useState<"conversation" | "inspector">(
    "conversation",
  );
  const streamRef = useRef<EventSource | null>(null);

  const refreshHistory = async () => setHistory(await listSessions());
  const loadSession = async (sessionId: string) =>
    setSession(await getSession(sessionId));

  useEffect(() => {
    void refreshHistory().catch((reason: unknown) => setError(String(reason)));
    return () => streamRef.current?.close();
  }, []);

  const followSession = (sessionId: string) => {
    streamRef.current?.close();
    const source = new EventSource(sessionEventStreamUrl(sessionId, 0));
    streamRef.current = source;
    source.addEventListener("session_event", () => {
      void refreshHistory();
      void loadSession(sessionId);
    });
    source.addEventListener("session_complete", () => {
      source.close();
      void refreshHistory();
      void loadSession(sessionId);
    });
    source.onerror = () => source.close();
  };

  const start = async () => {
    setError(null);
    setIsStarting(true);
    try {
      const started = await startConversation(MAYA_PROFILE);
      await loadSession(started.session_id);
      await refreshHistory();
      followSession(started.session_id);
      setView("conversation");
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Unable to start the scenario.",
      );
    } finally {
      setIsStarting(false);
    }
  };

  const selectSession = async (sessionId: string) => {
    setError(null);
    streamRef.current?.close();
    setView("conversation");
    try {
      await loadSession(sessionId);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Unable to load this session.",
      );
    }
  };

  return (
    <main
      className="app-shell"
      style={{
        height: "100vh",
        overflow: "hidden",
        fontFamily: '"Söhne", Arial, sans-serif',
      }}
    >
      <aside className="sidebar" style={{ minHeight: 0 }}>
        <h1>Financial Advisor</h1>
        <button
          className="start-button"
          disabled={isStarting}
          onClick={() => void start()}
        >
          {isStarting ? "Starting…" : "Start Scenario"}
        </button>
        <button
          className={
            view === "inspector"
              ? "inspector-button active"
              : "inspector-button"
          }
          onClick={() => setView("inspector")}
        >
          Observability
        </button>
        <div className="history-heading">
          <span>Conversation history</span>
        </div>
        <nav
          className="history"
          aria-label="Conversation history"
          style={{ minHeight: 0, flex: 1 }}
        >
          {history.length === 0 ? (
            <p className="muted">No stored sessions yet.</p>
          ) : (
            history.map((item) => (
              <button
                key={item.session_id}
                className={
                  session?.session_id === item.session_id
                    ? "history-item active"
                    : "history-item"
                }
                onClick={() => void selectSession(item.session_id)}
                title={item.client_question ?? "Waiting for client question…"}
              >
                <span className="history-question">
                  {item.client_question ?? "Waiting for client question…"}
                </span>
                <span className="history-meta">
                  <span>{formatSessionUpdatedAt(item.updated_at)}</span>
                  <em
                    style={{
                      color:
                        item.status === "escalated"
                          ? "#ff958c"
                          : item.status === "resolved"
                            ? "#7dd3a7"
                            : undefined,
                    }}
                  >
                    {statusLabel(item.status)}
                  </em>
                </span>
              </button>
            ))
          )}
        </nav>
      </aside>

      <section
        className="workspace"
        style={{ overflowY: "auto", background: "#1e1e1e" }}
      >
        {view === "inspector" && (
          <header className="topbar">
            <div>
              <h2>Observability</h2>
            </div>
          </header>
        )}
        {error && <div className="error-banner">{error}</div>}
        {view === "inspector" ? (
          <Inspector onError={setError} />
        ) : !session ? (
          <EmptyState />
        ) : (
          <SessionView session={session} />
        )}
      </section>
    </main>
  );
}

function EmptyState() {
  return (
    <div
      style={{
        minHeight: "calc(100vh - 68px)",
        display: "grid",
        placeItems: "center",
      }}
    >
      <section
        className="empty-state card"
        style={{
          width: "min(780px, 100%)",
          maxWidth: "780px",
          padding: "42px",
        }}
      >
        <h3>
          A guided financial-planning conversation for a simulated client.
        </h3>
        <p>
          Run Maya Chen’s scenario to see how her stated goal, time horizon,
          risk preferences, assets, and liabilities inform a bounded educational
          analysis. The workflow uses curated public guidance and, where
          relevant, current web sources.
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
            gap: "12px",
            marginTop: "26px",
          }}
        >
          <div>
            <strong>Client context</strong>
            <p style={{ marginTop: "6px" }}>
              The analysis considers the profile details relevant to the
              question.
            </p>
          </div>
          <div>
            <strong>Evidence and sources</strong>
            <p style={{ marginTop: "6px" }}>
              Research is selected, validated, and cited before it informs the
              response.
            </p>
          </div>
          <div>
            <strong>Scope and next steps</strong>
            <p style={{ marginTop: "6px" }}>
              The response presents educational options, assumptions, and
              material tradeoffs.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

function SessionView({ session }: { session: SessionSnapshot }) {
  const latest = session.recommendations.at(-1);
  return (
    <div className="content-grid">
      <div className="primary-column">
        <ClientProfileCard profile={session.client_profile} />
        <Conversation session={session} latest={latest} />
      </div>
      <aside className="inspector">
        <ProgressUpdates session={session} />
        <Sources recommendation={latest} />
      </aside>
    </div>
  );
}

function ClientProfileCard({ profile }: { profile: ClientProfile }) {
  return (
    <section className="card profile-card">
      <div>
        <p className="eyebrow">SYNTHETIC CLIENT PROFILE</p>
        <h3>
          {profile.name}, {profile.age}
        </h3>
      </div>
      <dl>
        <div>
          <dt>Goal</dt>
          <dd>
            {profile.primary_goal} in {profile.goal_time_horizon_years} years
          </dd>
        </div>
        <div>
          <dt>Risk</dt>
          <dd>{profile.risk_tolerance}</dd>
        </div>
        <div>
          <dt>Emergency fund</dt>
          <dd>{profile.emergency_fund_months} months</dd>
        </div>
        <div>
          <dt>Retirement savings</dt>
          <dd>{money(profile.retirement_savings)}</dd>
        </div>
        <div>
          <dt>Brokerage savings</dt>
          <dd>{money(profile.brokerage_savings)}</dd>
        </div>
        <div>
          <dt>Student loan</dt>
          <dd>
            {money(profile.student_loan_balance)} at{" "}
            {profile.student_loan_rate_percent}%
          </dd>
        </div>
      </dl>
    </section>
  );
}

function Conversation({
  session,
  latest,
}: {
  session: SessionSnapshot;
  latest: Recommendation | undefined;
}) {
  const terminalOutcome = session.status === "active" ? null : session.status;
  const latestProgressUpdate = session.progress_updates.at(-1);
  return (
    <section className="conversation">
      <div className="conversation-heading">
        <p className="eyebrow">CONVERSATION</p>
        {terminalOutcome && (
          <span
            className={`conversation-outcome ${terminalOutcome}`}
            role="status"
          >
            <span aria-hidden="true">
              {terminalOutcome === "resolved" ? "✓" : "!"}
            </span>
            {statusLabel(terminalOutcome)}
          </span>
        )}
      </div>
      {session.client_results.length === 0 && !terminalOutcome && (
        <article
          className="message client client-opening-pending"
          aria-live="polite"
        >
          <span>Client</span>
          <p className="client-opening-pending-text">
            <span className="pulse" />
            Maya is preparing her question…
          </p>
        </article>
      )}
      {session.client_results.map((result, index) => (
        <div className="turn" key={`${index}-${result.message}`}>
          <article className="message client">
            <span>Client</span>
            <p>{result.message}</p>
          </article>
          {session.recommendations[index] && (
            <RecommendationCard
              recommendation={session.recommendations[index]}
            />
          )}
        </div>
      ))}
      {session.status === "active" && latestProgressUpdate && (
        <p className="active-workflow-status" role="status">
          <span className="pulse" />
          {latestProgressUpdate}
        </p>
      )}
      {session.status === "escalated" && (
        <article className="message advisor">
          <span>Advisor</span>
          <p>
            Research is currently unavailable. Please try again later or consult
            a qualified financial professional.
          </p>
        </article>
      )}
      {latest && session.client_results.length === 0 && (
        <RecommendationCard recommendation={latest} />
      )}
      {terminalOutcome && (
        <div className="conversation-end">End of Conversation</div>
      )}
    </section>
  );
}

function RecommendationCard({
  recommendation,
}: {
  recommendation: Recommendation;
}) {
  return (
    <article className="message advisor">
      <span>Advisor</span>
      <section className="advisor-summary">
        <p>
          <CitedClaim
            claim={recommendation.summary}
            recommendation={recommendation}
          />
        </p>
      </section>
      {recommendation.options.length > 0 && (
        <section className="response-section options-section">
          <p className="response-heading">Options to consider</p>
          <div className="option-list">
            {recommendation.options.map((option) => (
              <section key={option.title}>
                <p className="response-heading">{option.title}</p>
                <p>
                  <CitedClaim
                    claim={option.description}
                    recommendation={recommendation}
                  />
                </p>
              </section>
            ))}
          </div>
        </section>
      )}
      <Details title="Assumptions" items={recommendation.assumptions} />
      <CitedDetails
        title="Risks"
        claims={recommendation.risks}
        recommendation={recommendation}
      />
      <Details title="Next steps" items={recommendation.next_steps} />
      <Details title="Limitations" items={recommendation.limitations} />
    </article>
  );
}

function CitedClaim({
  claim,
  recommendation,
}: {
  claim: CitedText;
  recommendation: Recommendation;
}) {
  return (
    <>
      {clientVisibleClaim(claim.text)}
      <sup className="citation-markers">
        {claim.evidence_ids.map((evidenceId) => {
          const index = recommendation.citations.findIndex(
            (citation) => citation.evidence_id === evidenceId,
          );
          const citation = recommendation.citations[index];
          return citation?.url ? (
            <a
              key={evidenceId}
              href={citation.url}
              target="_blank"
              rel="noreferrer"
            >
              [{index + 1}]
            </a>
          ) : (
            <span key={evidenceId}>[{index + 1}]</span>
          );
        })}
      </sup>
    </>
  );
}

function CitedDetails({
  title,
  claims,
  recommendation,
}: {
  title: string;
  claims: CitedText[];
  recommendation: Recommendation;
}) {
  if (claims.length === 0) return null;
  return (
    <section className="details response-section">
      <p className="response-heading">{title}</p>
      <ul>
        {claims.map((claim) => (
          <li key={`${claim.text}-${claim.evidence_ids.join()}`}>
            <CitedClaim claim={claim} recommendation={recommendation} />
          </li>
        ))}
      </ul>
    </section>
  );
}

function Details({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <section className="details response-section">
      <p className="response-heading">{title}</p>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </section>
  );
}

function Sources({
  recommendation,
}: {
  recommendation: Recommendation | undefined;
}) {
  return (
    <details className="card sources-card">
      <summary>
        <h3>Citations</h3>
      </summary>
      {recommendation === undefined ? (
        <p className="muted">Sources appear after evidence selection.</p>
      ) : (
        <ul>
          {recommendation.citations.map((citation, index) => (
            <li key={citation.evidence_id}>
              <strong className="source-number">[{index + 1}]</strong>
              {citation.url ? (
                <a href={citation.url} target="_blank" rel="noreferrer">
                  {citation.title}
                </a>
              ) : (
                citation.title
              )}
              <span>{citation.publisher}</span>
            </li>
          ))}
        </ul>
      )}
    </details>
  );
}

function ProgressUpdates({ session }: { session: SessionSnapshot }) {
  if (session.progress_updates.length === 0) return null;
  return (
    <section className="card workflow-updates">
      <h3>Progress Updates</h3>
      {session.progress_updates.map((update, index) => (
        <p key={`${index}-${update}`}>
          {index + 1}. {update}
        </p>
      ))}
    </section>
  );
}

function Inspector({ onError }: { onError: (message: string | null) => void }) {
  const [inspectorPage, setInspectorPage] = useState<
    "trajectory" | "sqlite" | "chunks"
  >("trajectory");
  const [tables, setTables] = useState<SqliteTableSummary[]>([]);
  const [selectedTable, setSelectedTable] = useState("sessions");
  const [tableRows, setTableRows] = useState<SqliteTableRows | null>(null);
  const [knowledge, setKnowledge] = useState<KnowledgeSummary | null>(null);
  const [knowledgeError, setKnowledgeError] = useState<string | null>(null);
  const [sessionIdentifier, setSessionIdentifier] = useState("");
  const [inspectedSession, setInspectedSession] =
    useState<SessionSnapshot | null>(null);
  const [isSessionLoading, setIsSessionLoading] = useState(false);

  useEffect(() => {
    void listSqliteTables()
      .then(setTables)
      .catch((reason: unknown) =>
        onError(
          reason instanceof Error
            ? reason.message
            : "Unable to inspect the session store.",
        ),
      );
  }, [onError]);

  useEffect(() => {
    if (inspectorPage !== "sqlite") return;
    void getSqliteTableRows(selectedTable)
      .then(setTableRows)
      .catch((reason: unknown) =>
        onError(
          reason instanceof Error
            ? reason.message
            : "Unable to load table rows.",
        ),
      );
  }, [inspectorPage, onError, selectedTable]);

  useEffect(() => {
    if (inspectorPage !== "chunks") return;
    setKnowledgeError(null);
    void getKnowledgeSummary()
      .then(setKnowledge)
      .catch((reason: unknown) =>
        setKnowledgeError(
          reason instanceof Error
            ? reason.message
            : "Local knowledge is unavailable.",
        ),
      );
  }, [inspectorPage]);

  const loadSessionTrajectory = async () => {
    const identifier = sessionIdentifier.trim();
    if (!identifier) return;
    onError(null);
    setIsSessionLoading(true);
    try {
      setInspectedSession(await getSession(identifier));
    } catch (reason) {
      onError(
        reason instanceof Error
          ? reason.message
          : "Unable to load the persisted session.",
      );
      setInspectedSession(null);
    } finally {
      setIsSessionLoading(false);
    }
  };

  return (
    <div className="inspector-view">
      <div
        className="table-tabs"
        role="tablist"
        aria-label="Data inspector pages"
      >
        <button
          className={inspectorPage === "trajectory" ? "active" : ""}
          onClick={() => setInspectorPage("trajectory")}
        >
          Session Trajectory
        </button>
        <button
          className={inspectorPage === "sqlite" ? "active" : ""}
          onClick={() => setInspectorPage("sqlite")}
        >
          SQLite Store
        </button>
        <button
          className={inspectorPage === "chunks" ? "active" : ""}
          onClick={() => setInspectorPage("chunks")}
        >
          Knowledge Store
        </button>
      </div>
      {inspectorPage === "trajectory" && (
        <SessionTrajectory
          identifier={sessionIdentifier}
          session={inspectedSession}
          loading={isSessionLoading}
          onIdentifierChange={setSessionIdentifier}
          onLoad={() => void loadSessionTrajectory()}
        />
      )}
      {inspectorPage === "sqlite" && (
        <section
          className="card table-browser"
          style={{ padding: "28px", minHeight: "820px" }}
        >
          <p className="eyebrow">SQLITE SESSION STORE</p>
          <div className="table-tabs">
            {tables.map((table) => (
              <button
                key={table.table_name}
                className={selectedTable === table.table_name ? "active" : ""}
                onClick={() => setSelectedTable(table.table_name)}
              >
                {table.table_name}
                <span>{table.row_count}</span>
              </button>
            ))}
          </div>
          <RawTable rows={tableRows} />
        </section>
      )}
      {inspectorPage === "chunks" &&
        (knowledgeError ? (
          <section className="card chunk-browser">
            <p>{knowledgeError}</p>
          </section>
        ) : knowledge ? (
          <KnowledgeChunkBrowser knowledge={knowledge} onError={onError} />
        ) : (
          <p className="muted">Loading the knowledge store…</p>
        ))}
    </div>
  );
}

function KnowledgeChunkBrowser({
  knowledge,
  onError,
}: {
  knowledge: KnowledgeSummary | null;
  onError: (message: string | null) => void;
}) {
  const [chunks, setChunks] = useState<KnowledgeChunkPage | null>(null);
  const [publisher, setPublisher] = useState("");
  const [topic, setTopic] = useState("");
  const [metadataField, setMetadataField] = useState("source_title");
  const [metadataValue, setMetadataValue] = useState("");
  const [minimumTokenCount, setMinimumTokenCount] = useState("");
  const [maximumTokenCount, setMaximumTokenCount] = useState("");
  const [offset, setOffset] = useState(0);
  const parsedMinimumTokenCount = Number.parseInt(minimumTokenCount, 10);
  const minTokenCount =
    Number.isFinite(parsedMinimumTokenCount) && parsedMinimumTokenCount > 0
      ? parsedMinimumTokenCount
      : undefined;
  const parsedMaximumTokenCount = Number.parseInt(maximumTokenCount, 10);
  const maxTokenCount =
    Number.isFinite(parsedMaximumTokenCount) && parsedMaximumTokenCount > 0
      ? parsedMaximumTokenCount
      : undefined;
  const usesTokenRange = metadataField === "token_count";

  useEffect(() => {
    void getKnowledgeChunks({
      publisher,
      topic,
      metadataField,
      metadataValue: usesTokenRange ? undefined : metadataValue,
      minTokenCount: usesTokenRange ? minTokenCount : undefined,
      maxTokenCount: usesTokenRange ? maxTokenCount : undefined,
      offset,
    })
      .then(setChunks)
      .catch((reason: unknown) =>
        onError(
          reason instanceof Error
            ? reason.message
            : "Unable to load knowledge chunks.",
        ),
      );
  }, [
    maxTokenCount,
    metadataField,
    metadataValue,
    minTokenCount,
    offset,
    onError,
    publisher,
    topic,
    usesTokenRange,
  ]);

  const resetOffset = () => setOffset(0);
  const firstVisible =
    chunks && chunks.chunks.length > 0 ? chunks.offset + 1 : 0;
  const lastVisible = chunks ? chunks.offset + chunks.chunks.length : 0;
  const hasNextPage =
    chunks !== null && lastVisible < chunks.total_matching_chunks;

  return (
    <section
      className="card chunk-browser"
      style={{ padding: "28px", minHeight: "620px" }}
    >
      <p className="eyebrow">KNOWLEDGE STORE EXPLORER</p>
      <div className="filters knowledge-filters">
        <label>
          Publisher
          <select
            value={publisher}
            onChange={(event) => {
              setPublisher(event.target.value);
              resetOffset();
            }}
          >
            <option value="">All publishers</option>
            {knowledge &&
              Object.keys(knowledge.publishers).map((name) => (
                <option key={name} value={name}>
                  {name} ({knowledge.publishers[name]})
                </option>
              ))}
          </select>
        </label>
        <label>
          Topic
          <select
            value={topic}
            onChange={(event) => {
              setTopic(event.target.value);
              resetOffset();
            }}
          >
            <option value="">All topics</option>
            {knowledge &&
              Object.keys(knowledge.topics).map((name) => (
                <option key={name} value={name}>
                  {name} ({knowledge.topics[name]})
                </option>
              ))}
          </select>
        </label>
        <label>
          Metadata field
          <select
            value={metadataField}
            onChange={(event) => {
              setMetadataField(event.target.value);
              resetOffset();
            }}
          >
            <option value="chunk_id">Chunk ID</option>
            <option value="canonical_candidate_id">Canonical ID</option>
            <option value="source_id">Source ID</option>
            <option value="source_title">Source title</option>
            <option value="publisher">Publisher</option>
            <option value="source_url">Source URL</option>
            <option value="topics">Topics</option>
            <option value="heading_path">Heading path</option>
            <option value="section_position">Section position</option>
            <option value="chunk_position">Chunk position</option>
            <option value="token_count">Token count</option>
          </select>
        </label>
        {usesTokenRange ? (
          <>
            <label>
              Minimum tokens
              <input
                type="number"
                min="1"
                step="1"
                value={minimumTokenCount}
                onChange={(event) => {
                  setMinimumTokenCount(event.target.value);
                  resetOffset();
                }}
              />
            </label>
            <label>
              Maximum tokens
              <input
                type="number"
                min="1"
                step="1"
                value={maximumTokenCount}
                onChange={(event) => {
                  setMaximumTokenCount(event.target.value);
                  resetOffset();
                }}
              />
            </label>
          </>
        ) : (
          <label>
            Contains
            <input
              value={metadataValue}
              onChange={(event) => {
                setMetadataValue(event.target.value);
                resetOffset();
              }}
              placeholder="Filter metadata"
            />
          </label>
        )}
      </div>
      <p className="muted knowledge-page-status">
        {chunks
          ? `${chunks.total_matching_chunks} matching chunks; showing ${firstVisible}–${lastVisible}.`
          : "Loading chunks…"}
      </p>
      <div className="chunk-list" style={{ maxHeight: "620px" }}>
        {chunks?.chunks.map((chunk) => (
          <article key={chunk.chunk_id}>
            <div>
              <strong>{chunk.source_title}</strong>
              <span>
                {chunk.publisher} · {chunk.token_count} tokens
              </span>
            </div>
            <a href={chunk.source_url} target="_blank" rel="noreferrer">
              {chunk.heading_path.at(-1) ?? chunk.source_title}
            </a>
            <p>{chunk.text}</p>
            <details className="metadata">
              <summary>Chunk metadata</summary>
              <dl>
                <Metadata label="Chunk ID" value={chunk.chunk_id} />
                <Metadata
                  label="Canonical ID"
                  value={chunk.canonical_candidate_id}
                />
                <Metadata label="Source ID" value={chunk.source_id} />
                <Metadata
                  label="Heading path"
                  value={chunk.heading_path.join(" > ")}
                />
                <Metadata label="Topics" value={chunk.topics.join(" · ")} />
                <Metadata
                  label="Section / chunk"
                  value={`${chunk.section_position} / ${chunk.chunk_position}`}
                />
                <Metadata
                  label="Embedding"
                  value={`${chunk.embedding_dimension} dimensions`}
                />
              </dl>
            </details>
          </article>
        ))}
      </div>
      <div className="knowledge-pagination" aria-label="Knowledge Store pages">
        <button
          onClick={() => setOffset(Math.max(0, offset - 50))}
          disabled={offset === 0}
          aria-label="Previous 50 chunks"
          title="Previous 50 chunks"
        >
          ‹
        </button>
        <button
          onClick={() => setOffset(offset + 50)}
          disabled={!hasNextPage}
          aria-label="Next 50 chunks"
          title="Next 50 chunks"
        >
          ›
        </button>
      </div>
    </section>
  );
}

function Metadata({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd title={value}>{value}</dd>
    </div>
  );
}

function SessionTrajectory({
  identifier,
  session,
  loading,
  onIdentifierChange,
  onLoad,
}: {
  identifier: string;
  session: SessionSnapshot | null;
  loading: boolean;
  onIdentifierChange: (value: string) => void;
  onLoad: () => void;
}) {
  return (
    <section
      className="card trace-explorer"
      style={{ padding: "28px", minHeight: "820px" }}
    >
      <p className="eyebrow">SESSION TRAJECTORY</p>
      <h3>Inspect a persisted ADK session</h3>
      <div className="trace-lookup">
        <input
          value={identifier}
          onChange={(event) => onIdentifierChange(event.target.value)}
          placeholder="Session ID"
          aria-label="Session ID"
        />
        <button onClick={onLoad} disabled={loading || !identifier.trim()}>
          {loading ? "Loading…" : "Inspect"}
        </button>
      </div>
      {session && (
        <Artifact
          title="Complete persisted trajectory"
          value={session}
          open
          large
        />
      )}
    </section>
  );
}

function Artifact({
  title,
  value,
  open = false,
  large = false,
}: {
  title: string;
  value: unknown;
  open?: boolean;
  large?: boolean;
}) {
  const json = JSON.stringify(value, null, 2);
  return (
    <details className="artifact" open={open}>
      <summary>{title}</summary>
      <button className="copy-button" onClick={() => void copyText(json)}>
        Copy JSON
      </button>
      <pre style={large ? { maxHeight: "620px" } : undefined}>{json}</pre>
    </details>
  );
}

function RawTable({ rows }: { rows: SqliteTableRows | null }) {
  const [selectedValue, setSelectedValue] = useState<{
    column: string;
    value: string;
  } | null>(null);
  if (rows === null) return <p className="muted">Loading table values…</p>;
  if (rows.rows.length === 0)
    return <p className="muted">This table has no rows yet.</p>;
  return (
    <div style={{ display: "grid", gap: "14px" }}>
      <div className="raw-table-wrap">
        <table>
          <thead>
            <tr>
              {rows.columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.rows.map((row, index) => (
              <tr key={index}>
                {rows.columns.map((column) => {
                  const value = formatValue(row[column]);
                  return (
                    <td key={column}>
                      <button
                        className="raw-value"
                        title="View complete value"
                        onClick={() => setSelectedValue({ column, value })}
                      >
                        {value}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {selectedValue && (
        <section
          style={{
            border: "1px solid #d9e2ec",
            borderRadius: "8px",
            background: "#f8fafc",
            padding: "14px",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: "12px",
              marginBottom: "9px",
            }}
          >
            <strong style={{ color: "#334e68", fontSize: ".8rem" }}>
              {selectedValue.column}
            </strong>
            <button
              onClick={() => setSelectedValue(null)}
              style={{
                border: 0,
                background: "transparent",
                color: "#486581",
                padding: "3px 0",
              }}
            >
              Close
            </button>
          </div>
          <pre
            style={{
              margin: 0,
              maxHeight: "300px",
              overflow: "auto",
              whiteSpace: "pre-wrap",
              overflowWrap: "anywhere",
              color: "#243b53",
              fontSize: ".76rem",
              userSelect: "text",
            }}
          >
            {formatInspectableValue(selectedValue.value)}
          </pre>
        </section>
      )}
    </div>
  );
}

function formatValue(value: unknown): string {
  const rendered =
    typeof value === "string" ? value : (JSON.stringify(value) ?? "");
  return rendered;
}

function formatInspectableValue(value: string): string {
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}
