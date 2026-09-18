# Financial Advisor Domain Contracts

## Purpose

This document defines the data exchanged by Financial Advisor components and the allowed workflow states. It is the design source for the first backend implementation and its tests.

## Implementation status

The enumerations and core Pydantic models are implemented in `src/financial_advisor/domain.py` and validated by `tests/unit/test_domain.py`. The deterministic transition engine is implemented in `src/financial_advisor/workflow.py` and validated by `tests/unit/test_workflow.py`. SQLite session persistence is implemented in `src/financial_advisor/persistence.py` and validated by `tests/unit/test_persistence.py`. Agent contracts and workflow adapters are implemented under `src/financial_advisor/agents/` and validated by focused unit and integration tests.

## Design principles

- Every inter-component message is a validated Pydantic model, not an untyped dictionary.
- The Advisor is the only communication bridge between Client and Analyst.
- The workflow engine enforces permissions and termination; agents do not bypass it.
- Agent outputs are structured and validated before downstream use.
- The first version uses only synthetic client data.

## Enumerations

### Risk tolerance

```python
RiskTolerance = Literal["conservative", "moderate", "aggressive"]
```

### Session state

```python
class SessionState(str, Enum):
    NEW = "new"
    CLIENT_OPENS = "client_opens"
    ADVISOR_ASSESSES = "advisor_assesses"
    ADVISOR_DELEGATES = "advisor_delegates"
    ANALYST_RESEARCHES = "analyst_researches"
    ADVISOR_PROPOSES = "advisor_proposes"
    CLIENT_REVIEWS = "client_reviews"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
```

`RESOLVED` and `ESCALATED` are terminal states.

## Core models

### ClientProfile

Owned by the Client Agent. It describes the synthetic client facts that may be relevant to a recommendation.

```python
class ClientProfile(BaseModel):
    client_id: str
    name: str
    age: int = Field(ge=18, le=120)
    risk_tolerance: RiskTolerance
    emergency_fund_months: int = Field(ge=0)
    retirement_savings: Decimal = Field(ge=0)
    brokerage_savings: Decimal = Field(ge=0)
    student_loan_balance: Decimal = Field(ge=0)
    student_loan_rate_percent: Decimal = Field(ge=0, le=100)
    primary_goal: str
    goal_time_horizon_years: int = Field(ge=0)
```

### ClientMessage

A profile-consistent message sent by the Client Agent to the Advisor. The Client Agent cannot target any other recipient.

```python
class ClientMessage(BaseModel):
    session_id: UUID
    message_id: UUID
    text: str = Field(min_length=1, max_length=2_000)
    follow_up_number: int = Field(ge=0, le=2)
```

The Workflow Engine loads the full `ClientProfile` from session state before the Advisor assesses a message. A Client message does not repeat a fixed subset of profile fields.

### AnalystProfileContext

`AnalystProfileContext` is an immutable, identity-free snapshot of the financial and goal fields needed for research. It intentionally excludes `client_id` and `name`.

The deterministic `to_analyst_profile_context` mapper creates this context from `ClientProfile`. When a new `ClientProfile` field is introduced, it is not exposed to the Analyst unless it is explicitly added to this mapper and context model.

### AnalystTask

Created by the Advisor. It converts a client need into an explicit, bounded research request.

```python
class AnalystTask(BaseModel):
    task_id: UUID
    session_id: UUID
    question: str = Field(min_length=1, max_length=2_000)
    research_plan: AdvisorResearchPlan
    client_context: AnalystProfileContext
    prohibited_actions: list[str]
```

`AdvisorResearchPlan` is a trusted, persisted authorization record. It binds the
task to a Client message, a bounded plan attempt, a research question, selected
local hybrid retrieval, and zero to two selected web scopes. The Advisor creates
the task with an `AnalystProfileContext` snapshot from session state. The
Analyst receives client context only through this Advisor-created task; it does
not communicate with the Client directly.

The `ResearchBrief` contract defines the required Analyst output. The Analyst must not create a client-facing recommendation.

### Evidence

An evidence record returned from the vector knowledge store or approved research adapter.

```python
class Evidence(BaseModel):
    evidence_id: UUID
    title: str
    publisher: str
    url: HttpUrl | None
    published_at: date | None
    retrieved_at: datetime
    excerpt: str = Field(min_length=1, max_length=1_000)
    source_type: Literal["vector_store", "web"]
```

### ResearchBrief

The Analyst's validated response to the Advisor. It is an internal artifact, never shown raw to the Client.

```python
class ResearchBrief(BaseModel):
    task_id: UUID
    findings: list[Finding] = Field(min_length=1)
    scenario_comparison: list[ScenarioComparison] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)
    caveats: list[str] = Field(min_length=1)
```

Every factual `Finding` references one or more `evidence_id` values from `evidence`.
`scenario_comparison` is populated only when the research task needs alternatives to be compared.

### Recommendation

The Advisor's client-facing response. This is the only advice-shaped object allowed to be sent to the Client Agent and UI.

```python
class Recommendation(BaseModel):
    recommendation_id: UUID
    session_id: UUID
    summary: CitedText
    options: list[RecommendationOption] = Field(min_length=1)
    rationale: list[CitedText] = Field(min_length=1)
    assumptions: list[str] = Field(min_length=1)
    risks: list[CitedText] = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1)
    evidence_ids: list[UUID] = Field(min_length=1)
    citations: list[EvidenceCitation] = Field(min_length=1)
    limitations: list[str]
    educational_disclaimer: str
```

`CitedText` contains client-facing text and the evidence IDs supporting that
claim. The server validates those IDs against the `ResearchBrief`, then creates
`EvidenceCitation` records from the original stored source metadata and URL.

### SessionEvent

An immutable, safe-to-display event for the UI and audit trail. Each session
has one root `trace_id`; every event in that session carries it, allowing trace
retrieval independent of a session lookup. Events contain a redacted summary,
not hidden reasoning or raw sensitive data.

```python
class SessionEvent(BaseModel):
    event_id: UUID
    session_id: UUID
    trace_id: UUID
    sequence: int = Field(ge=1)
    actor: Literal["client", "advisor", "analyst", "workflow_engine", "system"]
    event_type: EventType
    summary: str
    created_at: datetime
```

### ApplicationError

`ApplicationError` is a safe structured failure result for expected validation, workflow, research-provider, or model-provider failures. It contains a controlled `ErrorCode`, a displayable message, a recoverability flag, and an optional session identifier. It never carries raw exception data or credentials.

## Agent interfaces

```python
class ClientAgent(Protocol):
    def open_conversation(self, profile: ClientProfile) -> ClientMessage: ...
    def review_recommendation(
        self, recommendation: Recommendation, profile: ClientProfile
    ) -> ClientReview: ...

class AdvisorAgent(Protocol):
    def assess(self, message: ClientMessage, state: SessionContext) -> AdvisorDecision: ...
    def create_plan(self, decision: AdvisorDecision, state: SessionContext) -> AdvisorResearchPlan: ...
    def create_task(self, plan: AdvisorResearchPlan, state: SessionContext) -> AnalystTask: ...
    def propose(self, brief: ResearchBrief, state: SessionContext) -> Recommendation: ...

class AnalystAgent(Protocol):
    def research(self, task: AnalystTask) -> ResearchBrief: ...
```

The workflow engine invokes these interfaces. No `ClientAgent -> AnalystAgent` method exists.

## Allowed state transitions

| From | Allowed next states | Who triggers it |
| --- | --- | --- |
| `NEW` | `CLIENT_OPENS`, `ESCALATED` | Workflow Engine |
| `CLIENT_OPENS` | `ADVISOR_ASSESSES`, `ESCALATED` | Workflow Engine after Client message or the safe runtime boundary |
| `ADVISOR_ASSESSES` | `ADVISOR_DELEGATES`, `ADVISOR_PROPOSES`, `ESCALATED` | Advisor decision, validated by Workflow Engine |
| `ADVISOR_DELEGATES` | `ANALYST_RESEARCHES`, `ESCALATED` | Workflow Engine |
| `ANALYST_RESEARCHES` | `ADVISOR_ASSESSES`, `ADVISOR_PROPOSES`, `ESCALATED` | One local-only replan, brief validation, or safe failure |
| `ADVISOR_PROPOSES` | `CLIENT_REVIEWS`, `ESCALATED` | Workflow Engine |
| `CLIENT_REVIEWS` | `ADVISOR_ASSESSES`, `RESOLVED`, `ESCALATED` | Client review, validated by Workflow Engine |
| `RESOLVED` | none | Terminal |
| `ESCALATED` | none | Terminal |

## Routing rules

The Advisor returns one of three decisions:

```python
class AdvisorDecisionType(str, Enum):
    ANSWER_FROM_STATE = "answer_from_state"
    DELEGATE_RESEARCH = "delegate_research"
    ESCALATE = "escalate"
```

```text
If profile facts or prior validated research support the response:
    ANSWER_FROM_STATE

If fresh evidence, calculation, or comparison is required:
    DELEGATE_RESEARCH

If the request is out of scope, unsupported, or unsafe:
    ESCALATE
```

## Invariants

- `MAX_FOLLOW_UPS = 2`; the Workflow Engine rejects a third follow-up.
- An Analyst task is created only by the Advisor.
- An Analyst research brief must include at least one evidence record.
- A recommendation can cite only evidence present in the session.
- The Client Agent receives only `Recommendation`, never `ResearchBrief`.
- No agent action occurs after a terminal state.
- When persistence is enabled, every completed workflow action is saved atomically. The event trace is append-only: existing event IDs are never rewritten.

## Initial acceptance tests

1. A seeded Maya session reaches `RESOLVED`.
2. The Client cannot send a message to the Analyst.
3. The Advisor can answer a time-horizon follow-up from existing state without an Analyst call.
4. A request for a new comparison produces an `AnalystTask`.
5. A brief without evidence causes `ESCALATED`.
6. A third Client follow-up is rejected and the session closes safely.
