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
  summary: CitedText;
  options: Record<string, CitedText>;
  assumptions: string[];
  risks: CitedText[];
  next_steps: string[];
  citations: Citation[];
  limitations: string[];
}

export interface SessionSummary {
  session_id: string;
  client_question: string | null;
  updated_at: string;
  status: ConversationStatus;
}

export interface SessionSnapshot {
  session_id: string;
  status: ConversationStatus;
  client_results: ClientResult[];
  recommendations: Recommendation[];
  progress_updates: string[];
  state: Record<string, unknown>;
  events: Array<Record<string, unknown>>;
}

export type SqliteTableRows = Array<Record<string, unknown>>;

export interface DocumentChunk {
  canonical_candidate_id: string;
  source_title: string;
  publisher: string;
  source_url: string;
  topics: string[];
  section_position: number;
  chunk_position: number;
  token_count: number;
  text: string;
}

export interface KnowledgeStoreInspection {
  publishers: string[];
  topics: string[];
  total_matching_chunks: number;
  offset: number;
  chunks: DocumentChunk[];
}
