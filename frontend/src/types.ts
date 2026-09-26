export type ConversationStatus = "active" | "resolved" | "escalated";

export interface ClientProfile {
  name: string;
  age: number;
  risk_tolerance: "low" | "moderate" | "high";
  emergency_fund_months: number;
  retirement_savings: string;
  brokerage_savings: string;
  student_loan_balance: string;
  student_loan_rate_percent: string;
  primary_goal: string;
  goal_time_horizon_years: number;
}

export interface ClientResult {
  action: "question" | "accept";
  message: string;
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
  options: Array<{ title: string; description: CitedText }>;
  assumptions: string[];
  risks: CitedText[];
  next_steps: string[];
  citations: Citation[];
  limitations: string[];
}

export interface PersistedSessionEvent {
  sequence: number;
  event_id: string;
  timestamp: string;
  author: string;
  event: Record<string, unknown>;
}

export interface SessionSummary {
  session_id: string;
  client_question: string | null;
  updated_at: string;
  status: ConversationStatus;
}

export interface SessionSnapshot {
  session_id: string;
  client_profile: ClientProfile;
  status: ConversationStatus;
  updated_at: string;
  client_results: ClientResult[];
  recommendations: Recommendation[];
  progress_updates: string[];
  state: Record<string, unknown>;
  events: PersistedSessionEvent[];
}

export interface ConversationStarted {
  session_id: string;
  status: ConversationStatus;
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
  heading_path: string[];
  section_position: number;
  chunk_position: number;
  token_count: number;
  embedding_dimension: number;
  text: string;
}

export interface KnowledgeChunkPage {
  total_matching_chunks: number;
  offset: number;
  limit: number;
  chunks: KnowledgeChunk[];
}
