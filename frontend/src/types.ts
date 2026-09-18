export type SessionState =
  | "new"
  | "client_opens"
  | "advisor_assesses"
  | "advisor_delegates"
  | "analyst_researches"
  | "advisor_proposes"
  | "client_reviews"
  | "resolved"
  | "escalated";

export interface ClientProfile {
  client_id: string;
  name: string;
  age: number;
  risk_tolerance: string;
  emergency_fund_months: number;
  retirement_savings: string;
  brokerage_savings: string;
  student_loan_balance: string;
  student_loan_rate_percent: string;
  primary_goal: string;
  goal_time_horizon_years: number;
}

export interface SessionEvent {
  event_id: string;
  sequence: number;
  actor: string;
  event_type: string;
  summary: string;
  created_at: string;
}

export interface ClientMessage {
  message_id: string;
  text: string;
  follow_up_number: number;
}

export interface CitedText {
  text: string;
  evidence_ids: string[];
}

export interface Citation {
  evidence_id: string;
  title: string;
  publisher: string;
  url: string | null;
}

export interface Recommendation {
  recommendation_id: string;
  summary: CitedText;
  options: Array<{ title: string; description: CitedText; suitability: string }>;
  rationale: CitedText[];
  assumptions: string[];
  risks: CitedText[];
  next_steps: string[];
  citations: Citation[];
  educational_disclaimer: string;
}

export interface SessionSummary {
  session_id: string;
  trace_id: string;
  client_name: string;
  primary_goal: string;
  opening_question: string | null;
  started_at: string;
  state: SessionState;
}

export interface SessionSnapshot {
  session_id: string;
  trace_id: string;
  client_profile: ClientProfile;
  state: SessionState;
  terminal_response: string | null;
  follow_up_count: number;
  events: SessionEvent[];
  messages: ClientMessage[];
  research_plans: unknown[];
  analyst_tasks: unknown[];
  retrieval_traces: unknown[];
  research_briefs: unknown[];
  recommendations: Recommendation[];
  client_reviews: unknown[];
}

export interface ScenarioStarted {
  session_id: string;
  trace_id: string;
  state: SessionState;
}

export interface SqliteTableSummary {
  table_name: string;
  row_count: number;
}

export interface SqliteTableRows {
  table_name: string;
  columns: string[];
  rows: Array<Record<string, unknown>>;
}

export interface KnowledgeSummary {
  table_name: string;
  chunk_count: number;
  publishers: Record<string, number>;
  topics: Record<string, number>;
  embedding_model_name: string;
  embedding_dimension: number | null;
}

export interface KnowledgeChunk {
  chunk_id: string;
  canonical_candidate_id: string;
  source_id: string;
  source_title: string;
  publisher: string;
  source_url: string;
  topics: string[];
  heading: string;
  heading_path: string[];
  section_position: number;
  chunk_position: number;
  token_count: number;
  body_start_token: number | null;
  body_end_token_exclusive: number | null;
  source_content_hash: string | null;
  retrieved_at: string | null;
  embedding_dimension: number;
  text: string;
}

export interface KnowledgeChunkPage {
  total_matching_chunks: number;
  offset: number;
  limit: number;
  chunks: KnowledgeChunk[];
}
